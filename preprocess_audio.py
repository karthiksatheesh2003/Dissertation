#!/usr/bin/env python3
"""
Preprocess FLAC audio to 2‑channel log‑mel spectrograms + LAeq (dBA) target.
- Assumes each input FLAC is ~10 s total.
- Forces exactly one 10 s segment per file by pad/truncate after resampling.
- Saves compact binary .npz: spec[2, n_mels, T], laeq_db (float32), and metadata.
- Prints a line per file so you can see progress in SLURM logs.

Usage (example):
python preprocess_audio.py \
  --base /mnt/parscratch/users/$USER/extracted/partID299_1 \
  --out  /mnt/parscratch/users/$USER/features_npz \
  --sr 32000 --segment_seconds 10 --n_mels 128 --n_fft 1024 --hop_length 320 --win_length 1024
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import librosa
from scipy import signal

# ---------- A-weighting (IEC 61672) ----------
def _a_weighting_filter(sr: int):
    # Analog A-weighting constants
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997
    nums = [(2*np.pi*f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    dens = np.polymul([1, 4*np.pi*f4, (2*np.pi*f4)**2],
                      [1, 4*np.pi*f1, (2*np.pi*f1)**2])
    dens = np.polymul(np.polymul(dens, [1, 2*np.pi*f3]),
                      [1, 2*np.pi*f2])
    b, a = signal.bilinear(nums, dens, fs=sr)
    return b, a

def a_weight_audio(x: np.ndarray, sr: int):
    b, a = _a_weighting_filter(sr)
    return signal.lfilter(b, a, x)

PA_REF = 20e-6  # 20 microPascal

def laeq_dba(x: np.ndarray, sr: int) -> float:
    """LAeq in dBA for the given segment."""
    xa = a_weight_audio(x, sr)
    rms = np.sqrt(np.mean(np.square(xa) + 1e-12))
    return float(20.0 * np.log10(rms / PA_REF + 1e-12))

# ---------- Spectrogram helpers ----------
def logmel(y, sr, n_mels=128, n_fft=1024, hop_length=320, win_length=1024):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
        power=2.0, center=True
    )
    S = np.log10(S + 1e-10).astype(np.float32)  # log-power
    return S  # [n_mels, T]

# ---------- Core per-file processing ----------
def process_file(path: Path, out_root: Path, args) -> int:
    try:
        y, sr0 = sf.read(path, always_2d=False)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}", flush=True)
        return 0

    # Mixdown if stereo
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    # If read returns scalar or empty, skip
    if not isinstance(y, np.ndarray) or y.size == 0:
        print(f"[WARN] Empty audio {path}", flush=True)
        return 0

    # Resample to target sr
    if sr0 != args.sr:
        try:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr0, target_sr=args.sr, res_type="soxr_hq")
        except Exception:
            # fallback
            y = librosa.resample(y.astype(np.float32), orig_sr=sr0, target_sr=args.sr)
    else:
        y = y.astype(np.float32)

    sr = args.sr
    seg_len = int(args.segment_seconds * sr)

    # --- IMPORTANT: Force exactly one 10 s window (pad/truncate) ---
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)), mode="constant")
    elif len(y) > seg_len:
        y = y[:seg_len]

    # Output subdir mirrors the parent folder name of the input file
    rel_dir = path.parent.name
    out_dir = out_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute 2-channel spectrograms:
    # Ch0: log-mel of raw audio
    S0 = logmel(y, sr, n_mels=args.n_mels, n_fft=args.n_fft,
                hop_length=args.hop_length, win_length=args.win_length)

    # Ch1: log-mel of A-weighted audio (perceptual emphasis)
    y_aw = a_weight_audio(y, sr).astype(np.float32)
    S1 = logmel(y_aw, sr, n_mels=args.n_mels, n_fft=args.n_fft,
                hop_length=args.hop_length, win_length=args.win_length)

    # Align time dim in case of off-by-1
    T = min(S0.shape[1], S1.shape[1])
    spec = np.stack([S0[:, :T], S1[:, :T]], axis=0).astype(np.float32)  # [2, M, T]

    # Target: LAeq,10s (dBA)
    laeq = np.float32(laeq_dba(y, sr))

    meta = dict(
        file=str(path),
        start_sec=0,
        sr=int(sr),
        n_mels=int(args.n_mels),
        hop_length=int(args.hop_length),
        win_length=int(args.win_length),
        n_fft=int(args.n_fft)
    )

    out_name = f"{path.stem}__s000000.npz"
    out_path = out_dir / out_name
    np.savez_compressed(out_path, spec=spec, laeq_db=laeq, **meta)

    print(f"[OK] {path.name}  ->  {out_path.name}  |  spec {spec.shape}  |  LAeq={laeq:.2f} dBA", flush=True)
    return 1

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True,
                    help="Folder containing .flac files (or a parent with many subfolders).")
    ap.add_argument("--out", type=str, required=True, help="Output root for .npz features.")
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--segment_seconds", type=int, default=10)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=320)   # ~10 ms @ 32 kHz
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--max_files", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    base = Path(args.base)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Accept either: a directory of FLACs OR a parent with many subfolders
    if any(p.suffix.lower() == ".flac" for p in base.iterdir() if p.is_file()):
        files = sorted(base.glob("*.flac"))
    else:
        files = sorted(base.rglob("*.flac"))

    if args.max_files > 0:
        files = files[:args.max_files]

    if not files:
        print(f"[WARN] No .flac files found under {base}", flush=True)
        sys.exit(0)

    total = 0
    for p in files:
        try:
            total += process_file(p, out_root, args)
        except Exception as e:
            print(f"[WARN] {p}: {e}", flush=True)

    print(f"Done. Files processed: {total}", flush=True)

if __name__ == "__main__":
    main()
