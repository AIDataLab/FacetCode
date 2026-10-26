import numpy as np


def find_threshold_micro(yhat_raw, y):
    best_f1 = 0
    best_thresh = 0
    for thresh in np.arange(0.05, 0.95, 0.05):
        yhat = (yhat_raw >= thresh).astype(int)
        f1 = f1_score_micro(yhat, y)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def find_threshold_micro_v2(yhat_raw, y):
    best_f1 = 0
    best_thresh = 0
    for thresh in np.arange(0.1, 0.9, 0.02):
        yhat = (yhat_raw >= thresh).astype(int)
        f1 = f1_score_micro(yhat, y)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def f1_score_micro(yhat, y):
    tp = np.sum((yhat == 1) & (y == 1))
    fp = np.sum((yhat == 1) & (y == 0))
    fn = np.sum((yhat == 0) & (y == 1))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1