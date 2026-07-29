from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(
    page_title="AlphaForge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_data
def find_csv_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(directory.rglob("*.csv"))


@st.cache_data
def read_csv_safely(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📈 AlphaForge")

st.subheader(
    "Machine Learning for Financial Return Prediction and Portfolio Construction"
)

st.markdown(
    """
AlphaForge is an end-to-end quantitative research project combining
Fama–French asset-pricing factors, GARCH volatility estimates, and
path-signature features.

The project compares CNN, Random Forest, XGBoost, and ensemble models across
daily and monthly datasets, then converts predictions into portfolio strategies.
"""
)

# ---------------------------------------------------------
# Project status
# ---------------------------------------------------------

csv_files = find_csv_files(RESULTS_DIR)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Data Frequencies", "Daily + Monthly")
col2.metric("Factor Models", "FF3 + FF5")
col3.metric("Feature Families", "Signature + GARCH")
col4.metric("Result Files", len(csv_files))

st.divider()

# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

st.header("Research Pipeline")

st.code(
    """
Raw Portfolio and Factor Data
              |
              v
Data Cleaning and Alignment
              |
              v
Daily and Monthly Dataset Construction
              |
              v
Log-Signature and GARCH Feature Engineering
              |
              v
CNN / Random Forest / XGBoost / Ensemble Training
              |
              v
Out-of-Sample Model Evaluation
              |
              v
Portfolio Construction and Backtesting
              |
              v
Cross-Experiment Robustness Analysis
""",
    language="text",
)

# ---------------------------------------------------------
# Result status
# ---------------------------------------------------------

st.header("Project Status")

if not RESULTS_DIR.exists():
    st.error(f"Results directory not found: {RESULTS_DIR}")

elif not csv_files:
    st.warning(
        "The results folder exists, but no CSV output files were found. "
        "Run the evaluation, backtest, or robustness notebooks."
    )

else:
    st.success(
        f"Connected to {len(csv_files)} CSV result files inside the results directory."
    )

    with st.expander("View detected result files"):
        for file_path in csv_files:
            st.code(str(file_path.relative_to(PROJECT_ROOT)))

st.info("Use the sidebar to explore the project pages.")