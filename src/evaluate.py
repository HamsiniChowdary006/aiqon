"""Metrics, confidence labels and plotting helpers."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             brier_score_loss, accuracy_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.calibration import calibration_curve


def clf_metrics(y, p, thr=0.5):
    """Classification metrics from true labels y and predicted probabilities p."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    return dict(
        auc=roc_auc_score(y, p),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        brier=brier_score_loss(y, p),
        accuracy=accuracy_score(y, pred),
    )


def reg_metrics(y, pred):
    return dict(mae=mean_absolute_error(y, pred),
                rmse=float(np.sqrt(mean_squared_error(y, pred))),
                r2=r2_score(y, pred))


def confidence_label(p):
    """Confidence = how far the probability is from 0.5 (the 'coin flip' point)."""
    d = np.abs(np.asarray(p, dtype=float) - 0.5)
    return np.select([d < 0.15, d < 0.30], ["Low", "Medium"], default="High")


def plot_calibration(curves, path, title="Reliability curve"):
    """curves: {label: (y_true, p_pred)}"""
    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    for label, (y, p) in curves.items():
        frac, mean_p = calibration_curve(y, p, n_bins=6, strategy="quantile")
        plt.plot(mean_p, frac, marker="o", label=label)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed congestion rate")
    plt.title(title); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(path, dpi=140); plt.close()


def plot_confusion(y, p, path, title="Confusion matrix", thr=0.5):
    cm = confusion_matrix(y, (np.asarray(p) >= thr).astype(int))
    plt.figure(figsize=(4, 3.6))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Not congested", "Congested"]); plt.yticks([0, 1], ["Not congested", "Congested"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
