import pandas as pd
import os
from collections import Counter

# ====== Config ======
csv_paths = [
    "../outputs/mlp_v1/seed_42/blendpred_seed_42.csv",
    "../outputs/mlp_v1/seed_123/blendpred_seed_123.csv",
    "../outputs/mlp_v1/seed_2025/blendpred_seed_2025.csv",
    "../outputs/mlp_v1/seed_777/blendpred_seed_777.csv",
    "../outputs/mlp_v1/seed_999/blendpred_seed_999.csv",
    "../outputs/mlp_v1/seed_111/blendpred_seed_111.csv",
    "../outputs/mlp_v1/seed_222/blendpred_seed_222.csv",
    "../outputs/mlp_v1/seed_333/blendpred_seed_333.csv",
    "../outputs/mlp_v1/seed_444/blendpred_seed_444.csv",
    "../outputs/mlp_v1/seed_555/blendpred_seed_555.csv",
]

# Weights based on f1_micro (from your training logs)
weights = [
    0.8114,  # seed_42
    0.8129,  # seed_123
    0.8136,  # seed_2025
    0.8098,  # seed_222
    0.8128,  # seed_333
    0.8101,  # seed_111
    0.8147,  # seed_444
    0.8118,  # seed_555
    0.8126,  # seed_777
    0.8172,  # seed_999
]

output_dir = "../outputs/mlp_v1"

# ====== Load predictions ======
dfs = [pd.read_csv(path) for path in csv_paths]
print(f"✅ Loaded {len(dfs)} blended prediction CSVs.")

ids = dfs[0]['id'].values
num_samples = len(ids)

# ====== Weighted Ensemble Function ======
def weighted_ensemble(threshold, save_path):
    final_labels = []

    for i in range(num_samples):
        counter = Counter()

        for df, w in zip(dfs, weights):
            label_str = df.loc[i, 'labels']
            if pd.isna(label_str):
                continue
            labels = str(label_str).split()
            for lab in labels:
                counter[lab] += w  # add weight instead of +1

        # Pick labels above threshold
        chosen = [lab for lab, c in counter.items() if c >= threshold]

        if not chosen and len(counter) > 0:
            # fallback: pick labels with max weighted vote
            max_vote = max(counter.values())
            chosen = [lab for lab, c in counter.items() if c == max_vote]

        final_labels.append(" ".join(chosen) if chosen else "")

    submission = pd.DataFrame({"id": ids, "labels": final_labels})
    submission['labels'] = submission['labels'].fillna("").astype(str)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)

    print(f"✅ Weighted Ensemble saved → {save_path}")
    print(f"📊 Total samples: {len(submission)}")

# ====== Run weighted ensemble ======
# Threshold can be tuned, e.g., 0.5 * sum(weights)/10 ≈ average weight
weighted_threshold = sum(weights) / len(weights) * 0.5
save_path = os.path.join(output_dir, f"submission_weighted.csv")
weighted_ensemble(weighted_threshold, save_path)
