import logging
from typing import Any, Optional

import mlflow

logger = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, config: dict):
        tracking_cfg = config.get("tracking", {})
        self.experiment_name = tracking_cfg.get("experiment_name", "review-analytics")
        self.tracking_uri = tracking_cfg.get("tracking_uri", "mlruns")

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow experiment: {self.experiment_name} @ {self.tracking_uri}")

        self._active_run = None

    def start_run(self, run_name: str, tags: Optional[dict] = None):
        self._active_run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started run: {run_name}")
        return self._active_run

    def end_run(self):
        if self._active_run:
            mlflow.end_run()
            self._active_run = None

    def log_params(self, params: dict[str, Any]):
        for key, val in params.items():
            mlflow.log_param(key, val)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        for key, val in metrics.items():
            mlflow.log_metric(key, val, step=step)

    def log_artifact(self, filepath: str):
        mlflow.log_artifact(filepath)

    def log_model_run(
        self,
        model_name: str,
        params: dict,
        metrics: dict,
        tags: Optional[dict] = None,
    ):
        """Convenience method: one run per model evaluation."""
        run_tags = {"model_type": model_name}
        if tags:
            run_tags.update(tags)

        self.start_run(run_name=model_name, tags=run_tags)
        try:
            self.log_params(params)
            self.log_metrics(metrics)
        finally:
            self.end_run()

        logger.info(f"Logged run for {model_name}: acc={metrics.get('accuracy', 'N/A')}")
