#!/usr/bin/env python3
"""Main training entry point. Runs the full pipeline."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ReviewAnalyticsPipeline


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Train review classification models")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT training (faster)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting training pipeline")
    pipeline = ReviewAnalyticsPipeline(config_path=args.config)
    results = pipeline.run(skip_bert=args.skip_bert)

    logger.info("Final results:")
    for model, metrics in results.items():
        logger.info(f"  {model}: accuracy={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
