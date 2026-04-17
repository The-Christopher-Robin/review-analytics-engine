import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


st.set_page_config(page_title="Product Review Analytics", layout="wide")


@st.cache_data
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_results():
    path = Path("data/processed/model_results.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_summaries():
    path = Path("data/processed/sample_summaries.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_reviews():
    path = Path("data/raw/reviews.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def render_header():
    st.title("Product Review Analytics")
    st.markdown("Classification and summarization insights for product, marketing, and finance teams.")
    st.divider()


def render_data_overview(df: pd.DataFrame):
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Products", f"{df['product_id'].nunique():,}")
    col3.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    col4.metric("Categories", f"{df['category'].nunique()}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        rating_counts = df["rating"].value_counts().sort_index()
        colors = ["#d32f2f", "#f57c00", "#fbc02d", "#66bb6a", "#2e7d32"]
        ax.bar(rating_counts.index, rating_counts.values, color=colors)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_xticks([1, 2, 3, 4, 5])
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.subheader("Reviews by Category")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        cat_counts = df["category"].value_counts()
        ax.barh(cat_counts.index, cat_counts.values, color="#1976d2")
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    # rating trend by category
    st.subheader("Average Rating by Category")
    cat_ratings = df.groupby("category")["rating"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    st.dataframe(cat_ratings.rename(columns={"mean": "Avg Rating", "count": "Num Reviews"}), use_container_width=True)


def render_model_comparison(results: dict):
    st.header("Model Performance Comparison")

    rows = []
    for model, metrics in results.items():
        rows.append({
            "Model": model,
            "Accuracy": metrics["accuracy"],
            "Precision (macro)": metrics["precision_macro"],
            "Recall (macro)": metrics["recall_macro"],
            "F1 (macro)": metrics["f1_macro"],
            "F1 (weighted)": metrics["f1_weighted"],
        })

    comp_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)

    # highlight best
    best_model = comp_df.iloc[0]["Model"]
    best_acc = comp_df.iloc[0]["Accuracy"]
    st.success(f"Best model: **{best_model}** with {best_acc:.1%} accuracy")

    st.dataframe(
        comp_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision (macro)": "{:.4f}",
            "Recall (macro)": "{:.4f}",
            "F1 (macro)": "{:.4f}",
            "F1 (weighted)": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(comp_df))
    width = 0.25
    ax.bar(x - width, comp_df["Accuracy"], width, label="Accuracy", color="#1976d2")
    ax.bar(x, comp_df["F1 (macro)"], width, label="F1 (macro)", color="#f57c00")
    ax.bar(x + width, comp_df["F1 (weighted)"], width, label="F1 (weighted)", color="#66bb6a")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_ylabel("Score")
    plt.tight_layout()
    st.pyplot(fig)


def render_summaries(summaries: list[dict]):
    st.header("Review Summaries")
    st.markdown("Extractive summaries of sample reviews.")

    for i, s in enumerate(summaries[:10]):
        with st.expander(f"Review {i+1} (compression: {s['compression_ratio']:.0%})"):
            st.markdown("**Original:**")
            st.text(s["original"][:500] + ("..." if len(s["original"]) > 500 else ""))
            st.markdown("**Summary:**")
            st.info(s["summary"])


def render_insights(df: pd.DataFrame):
    st.header("Business Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Breakdown")
        sentiment_map = {1: "Very Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Very Positive"}
        df["sentiment"] = df["rating"].map(sentiment_map)

        fig, ax = plt.subplots(figsize=(6, 4))
        sentiment_order = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        sentiment_counts = df["sentiment"].value_counts().reindex(sentiment_order)
        colors = ["#d32f2f", "#f57c00", "#fbc02d", "#66bb6a", "#2e7d32"]
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, colors=colors,
               autopct="%1.1f%%", startangle=90)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Category Satisfaction")
        cat_sentiment = df.groupby("category")["rating"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(cat_sentiment.index, cat_sentiment.values)
        for bar, val in zip(bars, cat_sentiment.values):
            color = "#2e7d32" if val >= 3.5 else "#f57c00" if val >= 2.5 else "#d32f2f"
            bar.set_color(color)
        ax.set_xlabel("Average Rating")
        ax.set_xlim(0, 5)
        plt.tight_layout()
        st.pyplot(fig)

    # key metrics for stakeholders
    st.subheader("Key Metrics for Stakeholders")
    total = len(df)
    positive_pct = len(df[df["rating"] >= 4]) / total * 100
    negative_pct = len(df[df["rating"] <= 2]) / total * 100
    verified_pct = df["verified_purchase"].mean() * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positive Reviews", f"{positive_pct:.1f}%")
    m2.metric("Negative Reviews", f"{negative_pct:.1f}%")
    m3.metric("Verified Purchases", f"{verified_pct:.1f}%")
    m4.metric("Avg Review Length", f"{df['text'].str.len().mean():.0f} chars")


def main():
    render_header()

    reviews_df = load_reviews()
    results = load_results()
    summaries = load_summaries()

    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Model Comparison", "Summaries", "Business Insights"])

    with tab1:
        if reviews_df is not None:
            render_data_overview(reviews_df)
        else:
            st.warning("No review data found. Run `python scripts/fetch_data.py` first.")

    with tab2:
        if results:
            render_model_comparison(results)
        else:
            st.warning("No model results found. Run `python scripts/train.py` first.")

    with tab3:
        if summaries:
            render_summaries(summaries)
        else:
            st.warning("No summaries found. Run the training pipeline first.")

    with tab4:
        if reviews_df is not None:
            render_insights(reviews_df)
        else:
            st.warning("No data available for insights.")


if __name__ == "__main__":
    main()
