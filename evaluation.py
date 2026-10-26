import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)


def all_metrics(yhat, y, yhat_raw=None):
    metrics = {}

    metrics['auc_macro'] = roc_auc_score(y, yhat_raw, average='macro')
    metrics['auc_micro'] = roc_auc_score(y, yhat_raw, average='micro')

    metrics['f1_macro'] = f1_score(y, yhat, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y, yhat, average='micro', zero_division=0)

    metrics['precision_macro'] = precision_score(y, yhat, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y, yhat, average='micro', zero_division=0)

    metrics['recall_macro'] = recall_score(y, yhat, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y, yhat, average='micro', zero_division=0)

    return metrics


def print_metrics(metrics, prefix=None, log_file=None):
    if prefix:
        print(f"{prefix}:")
    if log_file:
        with open(log_file, 'a') as f:
            if prefix:
                f.write(f"{prefix}:\n")
                print(f"{prefix}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
                f.write(f"  {key}: {value:.4f}\n")
    else:
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")