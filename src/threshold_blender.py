import numpy as np
import os

seeds = ["seed_111", "seed_222", "seed_333", "seed_444", "seed_555"]  # include all seeds here
base_dir = "../outputs/mlp_v1"

for s in seeds:
    th_path = os.path.join(base_dir, s, "thresholds_greedy", "thresholds.npy")
    th = np.load(th_path)
    
    blended = 0.5 * th + 0.5 * 0.65  # blend with baseline 0.65
    np.save(os.path.join(base_dir, s, "thresholds_greedy", "thresholds_blend.npy"), blended)
    print(f"✅ Saved blended thresholds for {s}")
