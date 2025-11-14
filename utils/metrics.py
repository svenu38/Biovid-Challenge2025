# utils/metrics.py

import numpy as np
from sklearn.metrics import roc_curve, accuracy_score

def calculate_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    eer_idx = np.nanargmin(abs_diffs)
    eer = np.mean((fpr[eer_idx], fnr[eer_idx]))
    eer_thresh = thresholds[eer_idx]

    y_pred = (y_score >= eer_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)

    # APCER = False Accept Rate (accept impostors)
    apcer = np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1)

    # BPCER = False Reject Rate (reject genuine users)
    bpcer = np.sum((y_pred == 0) & (y_true == 1)) / max(np.sum(y_true == 1), 1)

    return acc, eer, apcer, bpcer
