import logging
import json
from pathlib import Path

import numpy as np
import yaml

from .data import DataLoader, TextPreprocessor
from .models import ClassicalModels, BertSentimentClassifier, ExtractiveSummarizer
from .evaluation.metrics import compute_metrics, confusion_matrix_data, per_class_report
from .tracking import ExperimentTracker

logger = logging.getLogger(__name__)


class ReviewAnalyticsPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.loader = DataLoader(self.config)
        self.preprocessor = TextPreprocessor(self.config)
        self.classical = ClassicalModels(self.config)
        self.summarizer = ExtractiveSummarizer(self.config)
        self.tracker = ExperimentTracker(self.config)

        self.results: dict = {}

    def run(self, skip_bert: bool = False):
        logger.info("=== Starting review analytics pipeline ===")

        # 1. load data
        batch = self.loader.load()
        reviews = batch.reviews

        # 2. preprocess
        df = self.preprocessor.prepare_dataframe(reviews)
        test_size = self.config["data"].get("test_split", 0.2)
        seed = self.config["data"].get("random_seed", 42)
        train_df, test_df = self.preprocessor.split_data(df, test_size=test_size, seed=seed)

        # 3. TF-IDF features for classical models
        X_train_tfidf = self.preprocessor.fit_tfidf(train_df["text"])
        X_test_tfidf = self.preprocessor.transform_tfidf(test_df["text"])
        y_train = train_df["label"].values
        y_test = test_df["label"].values

        # 4. train classical models
        logger.info("--- Training classical models ---")
        self.classical.train(X_train_tfidf, y_train)
        classical_results = self.classical.evaluate(X_test_tfidf, y_test)

        for model_name, res in classical_results.items():
            metrics = compute_metrics(y_test, res["predictions"])
            self.results[model_name] = metrics

            params = self._get_model_params(model_name)
            self.tracker.log_model_run(model_name, params, metrics)

        # 5. optionally train BERT
        if not skip_bert:
            logger.info("--- Training BERT classifier ---")
            bert = BertSentimentClassifier(self.config)

            train_texts = train_df["text"].tolist()
            train_labels = y_train.tolist()
            test_texts = test_df["text"].tolist()
            test_labels = y_test.tolist()

            history = bert.train(train_texts, train_labels)
            bert_eval = bert.evaluate(test_texts, test_labels)
            bert_metrics = compute_metrics(
                np.array(test_labels), bert_eval["predictions"]
            )
            self.results["bert"] = bert_metrics

            bert_params = {
                "model_name": self.config["models"]["bert"]["model_name"],
                "epochs": self.config["models"]["bert"]["epochs"],
                "batch_size": self.config["models"]["bert"]["batch_size"],
                "learning_rate": self.config["models"]["bert"]["learning_rate"],
            }
            self.tracker.log_model_run("bert", bert_params, bert_metrics)
        else:
            logger.info("Skipping BERT training (--skip-bert flag)")

        # 6. extractive summarization on a sample
        logger.info("--- Running extractive summarization ---")
        sample_texts = test_df["text"].head(100).tolist()
        summaries = self.summarizer.summarize_batch(sample_texts)

        avg_compression = np.mean([s["compression_ratio"] for s in summaries])
        logger.info(f"Average compression ratio: {avg_compression:.3f}")

        # 7. save results
        self._save_results(summaries[:20])

        logger.info("=== Pipeline complete ===")
        return self.results

    def _get_model_params(self, model_name: str) -> dict:
        if model_name == "logistic_regression":
            cfg = self.config["models"].get("logistic_regression", {})
            return {"max_iter": cfg.get("max_iter", 1000), "C": cfg.get("C", 1.0)}
        elif model_name == "random_forest":
            cfg = self.config["models"].get("random_forest", {})
            return {"n_estimators": cfg.get("n_estimators", 200), "max_depth": cfg.get("max_depth", 50)}
        elif model_name == "mlp":
            cfg = self.config["models"].get("mlp", {})
            return {
                "hidden_layers": str(cfg.get("hidden_layer_sizes", [256, 128])),
                "max_iter": cfg.get("max_iter", 300),
            }
        return {}

    def _save_results(self, sample_summaries: list[dict]):
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        results_path = out_dir / "model_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        summ_path = out_dir / "sample_summaries.json"
        with open(summ_path, "w") as f:
            json.dump(sample_summaries, f, indent=2)
        logger.info(f"Sample summaries saved to {summ_path}")
