#!/bin/bash
#SBATCH --job-name=cnn_improved
#SBATCH --output=/mnt/parscratch/users/%u/logs/cnn_improved_%A.out
#SBATCH --error=/mnt/parscratch/users/%u/logs/cnn_improved_%A.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ksatheesh1@sheffield.ac.uk

set -u

echo "=== IMPROVED CNN TRAINING ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Paths
DATA_DIR="/mnt/parscratch/users/$USER/features_npz"
RESULTS_DIR="/mnt/parscratch/users/$USER/cnn_results_improved"
SCRIPT_DIR="/mnt/parscratch/users/$USER/audio_pipeline"

mkdir -p "$RESULTS_DIR"

# Load environment
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
set +u
conda activate cnnenv
set -u

echo "=== SOLUTION 1: SIMPLER MODEL + MORE DATA ==="
python "$SCRIPT_DIR/cnn_model.py" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_DIR/simple_model" \
    --task_type regression \
    --pair_strategy adjacent \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.0005 \
    --patience 20 \
    --val_split 0.2 \
    --dropout 0.6 \
    --max_files 15000 \
    --num_workers 8 \
    --create_plots \
    --seed 42

echo "=== SOLUTION 2: DIFFERENT PAIRING STRATEGY ==="
python "$SCRIPT_DIR/cnn_model.py" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_DIR/same_location" \
    --task_type regression \
    --pair_strategy same_location \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.0005 \
    --patience 20 \
    --dropout 0.6 \
    --max_files 15000 \
    --num_workers 8 \
    --create_plots \
    --seed 42

echo "=== SOLUTION 3: CLASSIFICATION INSTEAD ==="
# Convert to classification problem (easier to learn)
python "$SCRIPT_DIR/cnn_model.py" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_DIR/classification" \
    --task_type classification \
    --n_classes 5 \
    --pair_strategy random \
    --batch_size 32 \
    --epochs 80 \
    --lr 0.001 \
    --patience 15 \
    --dropout 0.5 \
    --max_files 15000 \
    --num_workers 8 \
    --create_plots \
    --seed 42

echo "=== CREATING COMPARISON REPORT ==="
python3 << 'EOF'
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

results_dir = Path("/mnt/parscratch/users/acr24ks/cnn_results_improved")

experiments = ["simple_model", "same_location", "classification"]
metrics = {}

for exp in experiments:
    exp_dirs = list(results_dir.glob(f"{exp}/experiment_*"))
    if exp_dirs:
        latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
        metrics_file = latest_exp / "metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics[exp] = json.load(f)

# Create comparison plot
if metrics:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: R2 scores (for regression)
    reg_experiments = [k for k in metrics if 'r2_score' in metrics[k]]
    if reg_experiments:
        r2_scores = [metrics[exp]['r2_score'] for exp in reg_experiments]
        ax1.bar(reg_experiments, r2_scores)
        ax1.set_title('R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: MAE (for regression)
    if reg_experiments:
        mae_scores = [metrics[exp]['mae'] for exp in reg_experiments]
        ax2.bar(reg_experiments, mae_scores)
        ax2.set_title('Mean Absolute Error')
        ax2.set_ylabel('MAE (dBA)')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'improvement_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Comparison plot saved")

print("=== IMPROVEMENT EXPERIMENTS COMPLETED ===")
EOF

echo "==============================="
echo "Improved training completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "✅ Check improvement_comparison.png for results"
echo "==============================="