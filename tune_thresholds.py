# This script loads the validation set predictions and true labels from the trained symptom model,
# and performs threshold tuning for each symptom label to optimize F1 score.

import numpy as np
from sklearn.metrics import f1_score

# Load saved validation outputs
probs = np.load("artifacts/symptom_model/val_probs.npy")
labels = np.load("artifacts/symptom_model/val_labels.npy")

num_labels = labels.shape[1]

best_thresholds = []
best_f1_scores = []

print("Tuning thresholds per label...\n")

for i in range(num_labels):
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (probs[:, i] > threshold).astype(int)
        f1 = f1_score(labels[:, i], preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    best_thresholds.append(best_threshold)
    best_f1_scores.append(best_f1)

    print(f"Label {i}: Best Threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")

print("\nOptimal Thresholds:", best_thresholds)

# Compute overall F1 with optimized thresholds
optimized_preds = np.zeros_like(probs)

for i in range(num_labels):
    optimized_preds[:, i] = (probs[:, i] > best_thresholds[i]).astype(int)

micro_f1 = f1_score(labels, optimized_preds, average="micro")
macro_f1 = f1_score(labels, optimized_preds, average="macro")

print(f"\nOptimized Micro F1: {micro_f1:.4f}")
print(f"Optimized Macro F1: {macro_f1:.4f}")
