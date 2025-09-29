🏆 ICD10 Classification – Kaggle 1st Place Solution

We are proud to announce that our team secured 🥇 1st Place on the Kaggle ICD10 Classification Competition leaderboard with a final micro-F₂ score of 0.533. 🎉

📌 Project Overview

Medical coding is the process of assigning ICD10 codes to electronic health records (EHRs). This task is usually manual, time-consuming, and error-prone.

In this project, we solved the ICD10 multi-label classification problem using GatorTron embeddings (1024-dimensional clinical language model embeddings) and machine learning models.

Dataset:

~200,000 training records

~99,500 test records

~1,400 unique ICD10 codes

Features: 1024-dim embeddings per record

Goal: Predict ICD10 codes for unseen patient records.

Evaluation Metric: Average Micro-F₂ score.

⚙️ Approach
🔹 Step 1 – Data Preprocessing

Normalized embeddings.

Converted ICD10 labels into multi-hot vectors.

🔹 Step 2 – Model Training

Base model: Multi-Layer Perceptron (MLP) with dropout and hidden layers.

Loss functions explored:

Binary Cross-Entropy (BCE) → baseline

Focal Loss → moderate improvement

Asymmetric Loss (ASL) → best performance

🔹 Step 3 – Threshold Tuning

Global threshold sweeping (0.32–0.45).

Per-label thresholds improved recall on rare ICD codes.

🔹 Step 4 – Ensembling

Trained 10 MLP models with different random seeds.

Voting ensembles improved robustness.

Weighted ensemble (based on validation F1) gave the best final score.

📊 Results
Model Type	Public Score	Final Score
Single MLP (BCE)	0.494	0.496
Single MLP (ASL)	0.506	0.509
Ensemble (Voting, 10 seeds)	0.525	0.525
Weighted Ensemble (Final)	0.531	0.533

✅ Final Leaderboard Rank: #1

🚀 Key Highlights

First-place solution among all competing teams.

Demonstrated effectiveness of ASL loss + ensembling + threshold tuning.

Automated ICD10 classification using embeddings (without raw text).

Scalable, robust, and applicable to real-world medical coding.



📖 References

Kaggle Competition: ICD10 Classification

ICD-10 Official Documentation

GatorTron: Large Clinical Language Model

✨ Team Victory

This project is a testament to smart experimentation, rigorous validation, and teamwork.
We not only built a robust machine learning pipeline but also delivered a winning solution to a real-world medical coding challenge.
