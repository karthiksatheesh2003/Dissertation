#!/bin/bash
#SBATCH --job-name=prep_npz
#SBATCH --output=/mnt/parscratch/users/%u/logs/prep_npz_%A_%a.out
#SBATCH --error=/mnt/parscratch/users/%u/logs/prep_npz_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ksatheesh1@sheffield.ac.uk

set -euo pipefail

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

ROOT=/mnt/parscratch/users/$USER/extracted
OUT=/mnt/parscratch/users/$USER/features_npz
MANI=/mnt/parscratch/users/$USER/manifests

mkdir -p "$OUT" "$MANI" "/mnt/parscratch/users/$USER/logs"

# Ensure manifest exists (dirs that actually contain .flac)
if [ ! -s "$MANI/flac_dirs.txt" ]; then
  find "$ROOT" -type f -name '*.flac' -printf '%h\n' | sort -u > "$MANI/flac_dirs.txt"
fi

# Map array index -> one directory (1-based for sed)
IDX=$((SLURM_ARRAY_TASK_ID + 1))
TARGET_DIR=$(sed -n "${IDX}p" "$MANI/flac_dirs.txt" || true)

echo "ArrayTaskID=$SLURM_ARRAY_TASK_ID  IDX=$IDX"
echo "TARGET_DIR=$TARGET_DIR"

if [ -z "${TARGET_DIR:-}" ]; then
  echo "No directory for this index. Exiting."
  exit 0
fi

conda run -n audioenv python /mnt/parscratch/users/$USER/audio_pipeline/preprocess.py \
  --root "$TARGET_DIR" \
  --out "$OUT" \
  --sr 16000 \
  --file_seconds 10 \
  --n_mels 64 \
  --n_fft 1024 \
  --hop_length 256 \
  --patch_seconds 2.0 \
  --patch_hop_seconds 1.0
