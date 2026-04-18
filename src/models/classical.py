import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


class ClassicalModels:
    """Trains and evaluates sklearn baselines: LogReg, Random Forest, MLP."""

    def __init__(self, config: dict):
        model_cfg = config.get("models", {})

        lr_cfg = model_cfg.get("logistic_regression", {})
        rf_cfg = model_cfg.get("random_forest", {})
        mlp_cfg = model_cfg.get("mlp", {})

        self.models = {
            "logistic_regression": LogisticRegression(
                max_iter=lr_cfg.get("max_iter", 1000),
                C=lr_cfg.get("C", 1.0),
                solver="lbfgs",
                n_jobs=-1,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 200),
                max_depth=rf_cfg.get("max_depth", 50),
                random_state=42,
                n_jobs=-1,
            ),
            "mlp": MLPClassifier(
                hidden_layer_sizes=tuple(mlp_cfg.get("hidden_layer_sizes", [256, 128])),
                max_iter=mlp_cfg.get("max_iter", 300),
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
        }

        # filter to only requested models
        requested = model_cfg.get("classical", list(self.models.keys()))
        self.models = {k: v for k, v in self.models.items() if k in requested}

        self._trained: dict[str, bool] = {k: False for k in self.models}

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self._trained[name] = True

            train_acc = accuracy_score(y_train, model.predict(X_train))
            results[name] = train_acc
            logger.info(f"  {name} train accuracy: {train_acc:.4f}")

        return results

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, dict]:
        results = {}
        label_names = ["very_neg", "negative", "neutral", "positive", "very_pos"]

        for name, model in self.models.items():
            if not self._trained[name]:
                logger.warning(f"Skipping {name} -- not trained yet")
                continue

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            report = classification_report(
                y_test, preds,
                target_names=label_names,
                output_dict=True,
                zero_division=0,
            )

            results[name] = {
                "accuracy": acc,
                "predictions": preds,
                "report": report,
            }
            logger.info(f"  {name} test accuracy: {acc:.4f}")

        return results

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        if not self._trained[model_name]:
            raise RuntimeError(f"{model_name} not trained")
        return self.models[model_name].predict(X)

    def predict_proba(self, model_name: str, X: np.ndarray) -> Optional[np.ndarray]:
        model = self.models.get(model_name)
        if model is None or not self._trained[model_name]:
            return None
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None
