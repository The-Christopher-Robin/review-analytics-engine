import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


LABEL_NAMES = ["very_neg", "negative", "neutral", "positive", "very_pos"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "matrix": cm.tolist(),
        "labels": LABEL_NAMES[:len(set(y_true))],
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return report


def compare_models(results: dict[str, dict]) -> list[dict]:
    """Takes {model_name: {accuracy, ...}} and returns sorted comparison."""
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "accuracy": metrics.get("accuracy", 0),
            "f1_macro": metrics.get("f1_macro", 0),
            "f1_weighted": metrics.get("f1_weighted", 0),
        })
    return sorted(rows, key=lambda x: x["accuracy"], reverse=True)
