# 🩺 ICD-10 Multi-Label Text Embedding Classification  
**PALS TurboTech Build Hackathon 2025**  
**Team AURA – KCG College of Technology, Karapakkam**  
**Team Members:** Raji S | Rahul R  
**Rank 1 | Micro-F₂ = 0.533**

---

## 📄 Abstract
We addressed the ICD-10 medical coding challenge as a multi-label text embedding classification problem. Each electronic health record (EHR) embedding represents one or more diagnostic conditions mapped to ICD-10 codes. Our solution leverages a deep residual multi-layer perceptron (MLP) optimized with Asymmetric Loss (ASL) to effectively handle severe label imbalance and capture non-linear relationships across medical embeddings. Through systematic preprocessing, normalization, and stratified data splitting, we ensured reproducibility and robust validation. Our model achieved a final **micro-F₂ score of 0.533**, securing **Rank 1 on the official leaderboard**, demonstrating superior generalization and performance on unseen clinical embeddings.

---

## 🧠 Methodology Overview
1. **Data Preparation**  
   - Preprocessed embeddings and multi-hot ICD-10 label vectors.  
   - Normalization with `StandardScaler` (parameters saved for reproducibility).  
   - Stratified train–validation split preserving label distribution.  

2. **Model Architecture**  
   - Deep Residual MLP with 4 layers: `[1024 → 512 → 512 → 256]` + skip connections.  
   - `ReLU` activations and dropout (0.4) after each layer for regularization.  
   - Sigmoid output for multi-label classification.  

3. **Loss Function**  
   - **Asymmetric Loss (ASL)** to handle extreme label imbalance and boost rare-label recall.  

4. **Multi-Seed Training & Ensembling**  
   - Trained 10 models with different random seeds for robustness.  
   - Per-label threshold optimization and blending for stable predictions.  

---

## 📂 Repository Structure
├── src/
│ ├── train_mlp_upgraded.py
│ ├── test_mlp_upgraded.py
│ ├── optimize_thresholds.py
│ ├── threshold_blender.py
│ ├── generate_submission.py
│ └── download_data.py
└── README.md



> **Note:**  
> Datasets, preprocessed embeddings, and trained models are hosted on Google Drive.

---

## 📥 Downloading Data & Models
Before running any training or testing, download all required folders (data, preprocessed, models):


python src/download_data.py
This script automatically downloads the shared Google Drive folder:


https://drive.google.com/drive/folders/1q_LOeGmRC0b33faVXxJIQUMmpODC9Ayh?usp=sharing
All files will be stored in your current working directory.
---
🚀 Quick Start
1️⃣ Test Pre-trained Models
After downloading data:


python src/test_mlp_upgraded.py \
  --test-path "../data/test_data.npy" \
  --codes-path "../data/cleaned_unique_icd10_codes.txt" \
  --model-path "../outputs/mlp_v1/seed_42/best_mlp.pt" \
  --scaler-path "../outputs/mlp_v1/seed_42/splits_scaled/scaler.json" \
  --thresholds-path "../outputs/mlp_v1/seed_42/thresholds_greedy/thresholds_blend.npy" \
  --out-csv "../outputs/mlp_v1/seed_42/blendpred_seed_42.csv"
Then ensemble all prediction CSVs:


python src/generate_submission.py
---
2️⃣ Train From Scratch (Optional)
Example for seed 42:

```bash
python src/train_mlp_upgraded.py \
  --data-dir "D:/Kaggle/preprocessed" \
  --out-dir "D:/Kaggle/outputs/mlp_v1/seed_42" \
  --seed 42 \
  --epochs 30 \
  --batch-size 512
  ---
3️⃣ Optimize Thresholds
Run threshold optimization for each seed:


python src/optimize_thresholds.py \
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
  ---
4️⃣ Blend Thresholds
Combine optimized thresholds across seeds:


python src/threshold_blender.py
---
5️⃣ Final Testing with Blended Thresholds

python src/test_mlp_upgraded.py \
  --test-path "../data/test_data.npy" \
  --codes-path "../data/cleaned_unique_icd10_codes.txt" \
  --model-path "../outputs/mlp_v1/seed_42/best_mlp.pt" \
  --scaler-path "../outputs/mlp_v1/seed_42/splits_scaled/scaler.json" \
  --thresholds-path "../outputs/mlp_v1/seed_42/thresholds_greedy/thresholds_blend.npy" \
  --out-csv "../outputs/mlp_v1/seed_42/blendpred_seed_42.csv"
Then ensemble all blended predictions:


python src/generate_submission.py
📊 Results
Metric	Score
Micro-F₂	0.533
Rank	1 / Leaderboard

⚙️ Requirements

python >= 3.10
torch >= 2.0
numpy, pandas, scikit-learn
gdown, tqdm
Install:

pip install -r requirements.txt
🏁 Conclusion
This repository demonstrates that deep residual MLPs, when combined with asymmetric loss and threshold ensembling, can outperform heavier transformer models on structured clinical embeddings while remaining reproducible and lightweight.

🔗 References
ICD-10 Challenge Dataset (Kaggle Platform)

PALS TurboTech Build Hackathon 2025

KCG College of Technology
