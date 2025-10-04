MLP Model Pipeline – README
Overview

This repository contains a full pipeline for training, optimizing, and testing a Multi-Layer Perceptron (MLP) model for multi-label classification tasks. The workflow includes:

Downloading datasets

Training models with multiple seeds

Optimizing prediction thresholds

Testing models with blended thresholds

Generating final submission

The workflow is designed for code reusability and easy replication across different seeds or datasets.

1. Download Datasets

Run the following command to download all required datasets:

python download_data.py

2. Training Models

Train the upgraded MLP model using different random seeds. Update --data-dir and --out-dir as needed.

Training Commands
# Seed 42
python train_mlp_upgraded.py \
  --data-dir "D:/Kaggle/preprocessed" \
  --out-dir "D:/Kaggle/outputs/mlp_v1/seed_42" \
  --seed 42 \
  --epochs 30 \
  --batch-size 512

# Seed 123
python train_mlp_upgraded.py \
  --data-dir "D:/Kaggle/preprocessed" \
  --out-dir "D:/Kaggle/outputs/mlp_v1/seed_123" \
  --seed 123 \
  --epochs 30 \
  --batch-size 512

# Seed 2025
python train_mlp_upgraded.py \
  --data-dir "D:/Kaggle/preprocessed" \
  --out-dir "D:/Kaggle/outputs/mlp_v1/seed_2025" \
  --seed 2025 \
  --epochs 30 \
  --batch-size 512

# Seed 777
python train_mlp_upgraded.py \
  --data-dir "D:/Kaggle/preprocessed" \
  --out-dir "D:/Kaggle/outputs/mlp_v1/seed_777" \
  --seed 777 \
  --epochs 30 \
  --batch-size 512


Repeat this process for all seeds you plan to use.

3. Optimize Thresholds

After training, optimize the prediction thresholds using validation data for each seed. Adjust --init-threshold, --grid-start, --grid-end, --grid-steps, and --max-iters for fine-tuning.

python optimize_thresholds.py \
  --X-val "../outputs/mlp_v1/seed_42/splits_scaled/X_val.npy" \
  --Y-val "../outputs/mlp_v1/seed_42/splits_scaled/Y_val.npy" \
  --model-path "../outputs/mlp_v1/seed_42/best_mlp.pt" \
  --scaler-path "../outputs/mlp_v1/seed_42/splits_scaled/scaler.json" \
  --out-dir "../outputs/mlp_v1/seed_42/thresholds_greedy" \
  --mode greedy \
  --init-threshold 0.65 \
  --grid-start 0.30 --grid-end 0.70 --grid-steps 41 \
  --max-iters 3 \
  --batch-size 512


Repeat this step for each seed.

4. Blend Thresholds

Once all thresholds are optimized, blend them to improve final predictions:

python threshold_blender.py


This script generates blended thresholds for testing.

5. Test Models

Test all upgraded models using the blended thresholds:

# Seed 42
python test_mlp_upgraded.py \
  --test-path "../data/test_data.npy" \
  --codes-path "../data/cleaned_unique_icd10_codes.txt" \
  --model-path "../outputs/mlp_v1/seed_42/best_mlp.pt" \
  --scaler-path "../outputs/mlp_v1/seed_42/splits_scaled/scaler.json" \
  --thresholds-path "../outputs/mlp_v1/seed_42/thresholds_greedy/thresholds_blend.npy" \
  --out-csv "../outputs/mlp_v1/seed_42/blendpred_seed_42.csv"


Repeat for all seeds: 123, 2025, 777, 999, etc.

6. Generate Submission

Finally, generate the submission file:

python generate_submission.py

Notes

Ensure all directory paths exist before running scripts.

Recommended batch size: 512. Adjust depending on available GPU/CPU memory.

The workflow is seed-agnostic: simply add or remove seeds as needed.

Maintain consistent directory structure to ensure reusability.
