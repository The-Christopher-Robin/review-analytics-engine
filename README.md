# Review Analytics Engine

End-to-end NLP pipeline for classifying and summarizing product reviews. Compares classical ML baselines (Logistic Regression, Random Forest, MLP) against a fine-tuned BERT model, with MLflow experiment tracking and a Streamlit dashboard for exploring results.

## Tech Stack

- **Models**: PyTorch, Hugging Face Transformers (BERT), Scikit-learn
- **Tracking**: MLflow
- **Data**: Pandas, NumPy, Snowflake connector (CSV fallback for local dev)
- **Dashboard**: Streamlit, Matplotlib, Seaborn
- **Validation**: Pydantic

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

**1. Fetch data**

Downloads Amazon product reviews from HuggingFace Datasets:

```bash
python scripts/fetch_data.py --samples 50000
```

**2. Train models**

Runs classical baselines + optional BERT fine-tuning:

```bash
python scripts/train.py                # full pipeline (includes BERT)
python scripts/train.py --skip-bert    # classical models only (faster)
```

**3. View results**

```bash
python scripts/evaluate.py             # prints comparison table
streamlit run dashboard/app.py         # interactive dashboard
mlflow ui                              # experiment tracking UI
```

## Architecture

```
scripts/fetch_data.py  -->  data/raw/reviews.csv
                                  |
                          src/data/loader.py  (Snowflake or CSV)
                                  |
                        src/data/preprocessing.py  (cleaning, TF-IDF)
                                  |
                    +-------------+-------------+
                    |                           |
          src/models/classical.py     src/models/bert_classifier.py
            (LogReg, RF, MLP)           (BERT fine-tuning)
                    |                           |
                    +-------------+-------------+
                                  |
                      src/evaluation/metrics.py
                                  |
                      src/tracking/experiment.py  (MLflow)
                                  |
                        dashboard/app.py  (Streamlit)
```

## Configuration

All hyperparameters and data source settings are in `config/config.yaml`. To use Snowflake instead of local CSV, set `data.source: snowflake` and provide credentials via environment variables.

## Tests

```bash
pytest tests/ -v
```
