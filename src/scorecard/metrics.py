import pandas as pd

from scipy.stats import ks_2samp

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

def evaluate_model(y_true, probabilities):

    auc = roc_auc_score(y_true, probabilities)
    gini = 2 * auc - 1
    brier = brier_score_loss(y_true, probabilities)
    loss = log_loss(y_true, probabilities)
    ks = ks_2samp(
        probabilities[y_true == 0],
        probabilities[y_true == 1]
    ).statistic

    return {
        "roc_auc": auc,
        "gini": gini,
        "ks": ks,
        "brier_score": brier,
        "log_loss": loss,
    }

def print_metrics(metrics):

    print("\n================ MODEL PERFORMANCE ================\n")

    print(f"ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"Gini           : {metrics['gini']:.4f}")
    print(f"KS Statistic   : {metrics['ks']:.4f}")
    print(f"Brier Score    : {metrics['brier_score']:.4f}")
    print(f"Log Loss       : {metrics['log_loss']:.4f}")

    print("\n===================================================\n")


def save_metrics(metrics, output_path):

    serializable = {
        "roc_auc": round(float(metrics["roc_auc"]), 4),
        "gini": round(float(metrics["gini"]), 4),
        "ks": round(float(metrics["ks"]), 4),
        "brier_score": round(float(metrics["brier_score"]), 4),
        "log_loss": round(float(metrics["log_loss"]), 4),
    }

    pd.DataFrame([serializable]).to_csv(output_path, index=False)