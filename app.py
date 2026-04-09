"""
Sentiment Analysis Web Application
====================================

A single-page, step-by-step ML pipeline built with Streamlit.
Guides the user through: Upload → Preprocess → Visualize → Train → Evaluate → Predict.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import time

import config
from src.data.data_loader import load_csv, load_uploaded_csv
from src.data.data_preprocessing import preprocess_dataframe
from src.features.feature_engineering import encode_labels, split_data
from src.models.train_model import train_all_models, select_best_model, save_trained_artifacts
from src.models.evaluate_model import evaluate_all_models, build_comparison_dataframe
from src.models.predict_model import load_trained_pipeline, load_label_encoder, predict_sentiment
from src.visualization.plots import (
    plot_sentiment_distribution,
    plot_wordclouds,
    plot_all_confusion_matrices,
    plot_model_comparison,
)
from src.utils.helpers import file_exists
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Web Application",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS — Light, clean, professional
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #F8FAFC;
    }

    /* ── Hide sidebar completely ─────────────────────────── */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    /* ── Hero Section ───────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 2.5rem 1rem 1rem 1rem;
    }
    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #111827;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .title-accent {
        color: #2563EB;
    }

    /* ── Section Headers ────────────────────────────────── */
    .section-wrapper {
        max-width: 1100px;
        margin: 0 auto 2rem auto;
        padding: 0 1rem;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .section-step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #2563EB;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.95rem;
        flex-shrink: 0;
    }
    .section-step-done {
        background: #10B981;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }

    /* ── Cards ───────────────────────────────────────────── */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease;
    }
    .stat-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
        margin: 0;
    }
    .stat-value-green {
        color: #10B981;
    }
    .stat-value-amber {
        color: #F59E0B;
    }
    .stat-value-red {
        color: #EF4444;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 4px;
        font-weight: 500;
    }

    /* ── Prediction Result ──────────────────────────────── */
    .predict-result {
        background: #FFFFFF;
        border: 2px solid #E5E7EB;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .predict-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.3rem;
    }
    .predict-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
    }
    .predict-conf {
        font-size: 1rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }
    .predict-conf strong {
        color: #2563EB;
        font-weight: 700;
    }

    /* ── Info Bar ────────────────────────────────────────── */
    .info-bar {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        color: #1E40AF;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* ── Technology Pill Badges ──────────────────────────── */
    .tech-pill {
        display: inline-block;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 20px;
        padding: 5px 14px;
        margin: 3px;
        font-size: 0.82rem;
        color: #2563EB;
        font-weight: 500;
    }

    /* ── Cleaned text sample row ─────────────────────────── */
    .sample-row {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .sample-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sample-text {
        font-size: 0.92rem;
        color: #111827;
        margin-top: 3px;
        line-height: 1.5;
    }

    /* ── Divider ─────────────────────────────────────────── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #D1D5DB, transparent);
        margin: 2.5rem auto;
        max-width: 1100px;
    }

    /* ── Footer ──────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 2rem 1rem;
        color: #9CA3AF;
        font-size: 0.82rem;
    }
    .app-footer a {
        color: #2563EB;
        text-decoration: none;
        font-weight: 500;
    }

    /* ── Fix Streamlit default spacing ───────────────────── */
    .block-container {
        padding-top: 1rem !important;
        max-width: 1200px;
    }

    /* ── Make dataframes readable ─────────────────────── */
    .stDataFrame {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# HELPER — section header
# ──────────────────────────────────────────────────────────────
def _section(step: int, title: str, done: bool = False):
    """Render a numbered section header."""
    css_class = "section-step-done" if done else ""
    st.markdown(
        f"""<div class="section-header">
            <span class="section-step {css_class}">{step}</span>
            <h2 class="section-title">{title}</h2>
        </div>""",
        unsafe_allow_html=True,
    )

def _divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# CACHED DATA LOADER
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_and_preprocess(path):
    """Load and preprocess the raw dataset."""
    df = load_csv(path, column_names=config.DATASET_COLUMNS)
    return df


@st.cache_data(show_spinner=False)
def _preprocess(df_raw):
    """Clean text and encode labels."""
    df = preprocess_dataframe(df_raw, text_col=config.TEXT_COLUMN)
    df, le = encode_labels(df)
    return df, le


# ══════════════════════════════════════════════════════════════
#  HERO — Title
# ══════════════════════════════════════════════════════════════
st.markdown(
    """<div class="app-header">
        <h1 class="app-title">🎯 Sentiment <span class="title-accent">Analysis</span> Web Application</h1>
        <p class="app-subtitle">
            A step-by-step ML pipeline — Upload data, preprocess, visualize, train models, and predict sentiment
        </p>
    </div>""",
    unsafe_allow_html=True,
)

# Quick info pills
techs = ["Python", "Streamlit", "scikit-learn", "NLTK", "TF-IDF", "Pandas", "Matplotlib"]
pills = " ".join(f'<span class="tech-pill">{t}</span>' for t in techs)
st.markdown(f'<div style="text-align:center; margin-bottom:2rem;">{pills}</div>', unsafe_allow_html=True)

_divider()


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — Upload Dataset
# ══════════════════════════════════════════════════════════════
_section(1, "Upload Dataset", done="df_raw" in st.session_state)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    source = st.radio(
        "Choose data source",
        ["📁  Use default dataset", "📤  Upload your own CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "📁  Use default dataset":
        if st.button("Load Default Dataset", type="primary"):
            if file_exists(config.RAW_DATASET_PATH):
                with st.spinner("Loading dataset …"):
                    df_raw = _load_and_preprocess(config.RAW_DATASET_PATH)
                    st.session_state["df_raw"] = df_raw
                st.success(f"✅ Loaded **{len(df_raw):,}** rows from `twitter_training.csv`")
            else:
                st.error("Default dataset not found. Place `twitter_training.csv` in `data/raw/`.")
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded:
            with st.spinner("Reading file …"):
                df_raw = load_uploaded_csv(uploaded, column_names=config.DATASET_COLUMNS)
                st.session_state["df_raw"] = df_raw
            st.success(f"✅ Uploaded **{len(df_raw):,}** rows")

with col_info:
    st.markdown(
        """<div class="info-bar">
            <strong>Supported format:</strong> CSV with columns — id, topic, sentiment, text
        </div>""",
        unsafe_allow_html=True,
    )
    if "df_raw" in st.session_state:
        df_raw = st.session_state["df_raw"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="stat-card"><p class="stat-value">{len(df_raw):,}</p>'
                f'<p class="stat-label">Total Rows</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            n_cols = len(df_raw.columns)
            st.markdown(
                f'<div class="stat-card"><p class="stat-value">{n_cols}</p>'
                f'<p class="stat-label">Columns</p></div>',
                unsafe_allow_html=True,
            )

# Data preview
if "df_raw" in st.session_state:
    with st.expander("Preview raw dataset", expanded=False):
        st.dataframe(st.session_state["df_raw"].head(8), use_container_width=True)


_divider()


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — Data Preprocessing
# ══════════════════════════════════════════════════════════════
_section(2, "Data Preprocessing", done="df" in st.session_state)

if "df_raw" not in st.session_state:
    st.info("⬆️  Upload a dataset in Step 1 to continue.")
else:
    if st.button("🔄 Clean & Preprocess Text", type="primary"):
        with st.spinner("Cleaning text — removing URLs, mentions, stop-words, applying lemmatization …"):
            df, le = _preprocess(st.session_state["df_raw"])
            st.session_state["df"] = df
            st.session_state["le"] = le

    if "df" in st.session_state:
        df = st.session_state["df"]
        le = st.session_state["le"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="stat-card"><p class="stat-value stat-value-green">{len(df):,}</p>'
                f'<p class="stat-label">Rows After Cleaning</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            n_classes = len(le.classes_)
            st.markdown(
                f'<div class="stat-card"><p class="stat-value">{n_classes}</p>'
                f'<p class="stat-label">Sentiment Classes</p></div>',
                unsafe_allow_html=True,
            )
        with c3:
            classes_str = " · ".join(le.classes_)
            st.markdown(
                f'<div class="stat-card"><p class="stat-value" style="font-size:1rem;">{classes_str}</p>'
                f'<p class="stat-label">Class Labels</p></div>',
                unsafe_allow_html=True,
            )

        # Show sample cleaned texts
        st.markdown("")
        st.markdown("**Sample cleaned texts:**")
        for i in range(min(3, len(df))):
            orig = str(df["text"].iloc[i])[:120]
            clean = str(df["clean_text"].iloc[i])[:120]
            st.markdown(
                f"""<div class="sample-row">
                    <p class="sample-label">Original</p>
                    <p class="sample-text">{orig}</p>
                    <p class="sample-label" style="margin-top:8px;">Cleaned</p>
                    <p class="sample-text" style="color:#2563EB;">{clean}</p>
                </div>""",
                unsafe_allow_html=True,
            )


_divider()


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — Data Visualization
# ══════════════════════════════════════════════════════════════
_section(3, "Data Visualization", done="df" in st.session_state)

if "df" not in st.session_state:
    st.info("⬆️  Complete preprocessing in Step 2 to unlock visualizations.")
else:
    df = st.session_state["df"]

    viz_tabs = st.tabs(["📊 Distribution", "☁️ Word Clouds"])

    with viz_tabs[0]:
        fig = plot_sentiment_distribution(df)
        st.pyplot(fig)

    with viz_tabs[1]:
        with st.spinner("Generating word clouds …"):
            fig = plot_wordclouds(df)
        st.pyplot(fig)


_divider()


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — Train Models
# ══════════════════════════════════════════════════════════════
_section(4, "Train Models", done="evaluations" in st.session_state)

if "df" not in st.session_state:
    st.info("⬆️  Complete Steps 1–2 before training.")
else:
    df = st.session_state["df"]
    le = st.session_state["le"]

    # Config summary
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="stat-card"><p class="stat-value">3</p>'
            '<p class="stat-label">Models</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><p class="stat-value">{config.CV_FOLDS}</p>'
            f'<p class="stat-label">CV Folds</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="stat-card"><p class="stat-value">Grid</p>'
            '<p class="stat-label">Search Strategy</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="stat-card"><p class="stat-value">{config.RANDOM_STATE}</p>'
            f'<p class="stat-label">Random Seed</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        progress = st.progress(0, text="Initializing …")

        def progress_cb(step, total, msg):
            progress.progress(step / total, text=msg)

        with st.spinner("Training with GridSearchCV — this may take a few minutes …"):
            results = train_all_models(X_train, y_train, progress_callback=progress_cb)

        progress.progress(1.0, text="✅ Training complete!")

        # Evaluate
        evaluations = evaluate_all_models(results, X_test, y_test, list(le.classes_))
        st.session_state["results"] = results
        st.session_state["evaluations"] = evaluations

        # Select & save best
        best_name, best_pipeline = select_best_model(results)
        save_trained_artifacts(best_pipeline, le)
        st.session_state["best_name"] = best_name

    # Show results if trained
    if "evaluations" in st.session_state:
        evaluations = st.session_state["evaluations"]
        best_name = st.session_state.get("best_name", "")

        st.markdown("")
        comparison_df = build_comparison_dataframe(evaluations)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.success(f"🏆 **Best Model: {best_name}** — saved to `models/`")

        # CV scores
        with st.expander("📋 Cross-Validation Details"):
            for name, res in st.session_state["results"].items():
                icon = "🥇" if name == best_name else "📌"
                st.markdown(
                    f"{icon} **{name}** — CV F1: `{res['best_cv_score']:.4f}` "
                    f"| Params: `{res['best_params']}`"
                )


_divider()


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — Model Performance
# ══════════════════════════════════════════════════════════════
_section(5, "Model Performance", done="evaluations" in st.session_state)

if "evaluations" not in st.session_state:
    st.info("⬆️  Train models in Step 4 to see performance metrics.")
else:
    evaluations = st.session_state["evaluations"]
    le = st.session_state["le"]
    best_name = st.session_state.get("best_name", "")
    best_m = evaluations.get(best_name, {})

    # Metric cards
    st.markdown(f"**Best model: {best_name}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card"><p class="stat-value">{best_m.get("accuracy", 0)*100:.2f}%</p>'
            f'<p class="stat-label">Accuracy</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><p class="stat-value stat-value-green">'
            f'{best_m.get("precision", 0)*100:.2f}%</p>'
            f'<p class="stat-label">Precision</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card"><p class="stat-value stat-value-amber">'
            f'{best_m.get("recall", 0)*100:.2f}%</p>'
            f'<p class="stat-label">Recall</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="stat-card"><p class="stat-value">'
            f'{best_m.get("f1", 0)*100:.2f}%</p>'
            f'<p class="stat-label">F1 Score</p></div>',
            unsafe_allow_html=True,
        )

    # Classification reports
    with st.expander("📑 Detailed Classification Reports"):
        for name, metrics in evaluations.items():
            icon = "🥇" if name == best_name else "📌"
            st.markdown(f"**{icon} {name}**")
            report_df = pd.DataFrame(metrics["classification_report"]).T
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
            st.markdown("")

    # Charts side by side
    perf_tabs = st.tabs(["📊 Model Comparison", "🔢 Confusion Matrices"])

    with perf_tabs[0]:
        fig = plot_model_comparison(evaluations)
        st.pyplot(fig)

    with perf_tabs[1]:
        fig = plot_all_confusion_matrices(evaluations, list(le.classes_))
        st.pyplot(fig)


_divider()


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — Predict Sentiment
# ══════════════════════════════════════════════════════════════
_section(6, "Predict Sentiment")

model_ready = file_exists(config.TRAINED_MODEL_PATH) and file_exists(config.LABEL_ENCODER_PATH)

if not model_ready:
    st.info("⬆️  Train a model in Step 4 to enable predictions.")
else:
    pipeline = load_trained_pipeline()
    label_encoder = load_label_encoder()

    col_input, col_result = st.columns([3, 2])

    with col_input:
        user_text = st.text_area(
            "Enter text to analyse",
            placeholder="e.g. This product is absolutely amazing, best purchase ever!",
            height=130,
        )

        predict_clicked = st.button("🔮 Predict Sentiment", type="primary", use_container_width=True)

        # Quick examples
        st.markdown("**Try an example:**")
        examples = [
            "This product is absolutely amazing!",
            "Terrible experience, total waste of money.",
            "The package arrived today, it's okay.",
            "Just saw a bird outside my window.",
        ]

        ex_cols = st.columns(2)
        for i, ex in enumerate(examples):
            with ex_cols[i % 2]:
                if st.button(f"💬 {ex[:45]}", key=f"ex_{i}", use_container_width=True):
                    st.session_state["quick_text"] = ex

    with col_result:
        text_to_predict = None
        if predict_clicked and user_text.strip():
            text_to_predict = user_text
        elif "quick_text" in st.session_state:
            text_to_predict = st.session_state.pop("quick_text")

        if text_to_predict:
            with st.spinner("Analysing …"):
                result = predict_sentiment(text_to_predict, pipeline, label_encoder)

            # Color by sentiment
            color_map = {
                "Positive": "#10B981",
                "Negative": "#EF4444",
                "Neutral": "#2563EB",
                "Irrelevant": "#F59E0B",
            }
            label_color = color_map.get(result["predicted_label"], "#111827")

            st.markdown(
                f"""<div class="predict-result">
                    <div class="predict-emoji">{result['emoji']}</div>
                    <div class="predict-label" style="color:{label_color};">
                        {result['predicted_label']}
                    </div>
                    <div class="predict-conf">
                        Confidence: <strong>{result['confidence']*100:.1f}%</strong>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            with st.expander("📝 Preprocessing details"):
                st.markdown(f"**Original:** {result['original_text']}")
                st.markdown(f"**Cleaned:** {result['cleaned_text']}")
        else:
            st.markdown(
                """<div class="predict-result" style="border-style:dashed; color:#9CA3AF;">
                    <div class="predict-emoji">🔮</div>
                    <div class="predict-label" style="color:#D1D5DB;">Awaiting input</div>
                    <div class="predict-conf">Enter text and click Predict</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("")
st.markdown("")
st.markdown(
    """<div class="app-footer">
        Sentiment Analysis Web Application — Project<br>
        Built with Python · Streamlit · scikit-learn
    </div>""",
    unsafe_allow_html=True,
)
