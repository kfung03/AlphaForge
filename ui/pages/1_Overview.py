from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "robustness_analysis"
    / "tables"
)

CLASSIFICATION_PATH = TABLE_DIR / "all_classification_results.csv"
BACKTEST_PATH = TABLE_DIR / "all_backtest_results.csv"


st.title("Project Overview")

st.markdown(
    """
AlphaForge is an end-to-end quantitative machine-learning project designed to
predict high-return portfolios and evaluate whether those predictions can be
converted into useful investment strategies.
"""
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return load_csv(path)

    return pd.DataFrame()


classification = load_optional(CLASSIFICATION_PATH)
backtests = load_optional(BACKTEST_PATH)


st.subheader("Research Question")

st.markdown(
    """
Can traditional asset-pricing factors, conditional-volatility estimates, and
path-dependent time-series features be combined to improve financial return
classification?

The project compares results across:

- Daily and monthly datasets
- Fama–French three-factor and five-factor specifications
- Signature, GARCH, and combined feature sets
- CNN, XGBoost, Random Forest, and ensemble models
"""
)


st.divider()

st.subheader("Project Pipeline")

pipeline_col1, pipeline_col2, pipeline_col3 = st.columns(3)

with pipeline_col1:
    st.markdown(
        """
### 1. Data

- Clean portfolio returns
- Align Fama–French factors
- Build daily and monthly datasets
- Create train, validation, and test splits
"""
    )

with pipeline_col2:
    st.markdown(
        """
### 2. Modeling

- Generate log-signature features
- Estimate GARCH volatility
- Train multiple classifiers
- Tune classification thresholds
"""
    )

with pipeline_col3:
    st.markdown(
        """
### 3. Evaluation

- Compare classification metrics
- Construct portfolio signals
- Backtest model strategies
- Test robustness across experiments
"""
    )


st.divider()

st.subheader("Key Results")

metric1, metric2, metric3, metric4 = st.columns(4)

if not classification.empty and "Accuracy" in classification.columns:
    metric1.metric(
        "Best Accuracy",
        f"{classification['Accuracy'].max():.3f}",
    )
else:
    metric1.metric("Best Accuracy", "Pending")

if not classification.empty and "F1" in classification.columns:
    metric2.metric(
        "Best F1",
        f"{classification['F1'].max():.3f}",
    )
else:
    metric2.metric("Best F1", "Pending")

if not classification.empty and "ROC_AUC" in classification.columns:
    metric3.metric(
        "Best ROC-AUC",
        f"{classification['ROC_AUC'].max():.3f}",
    )
else:
    metric3.metric("Best ROC-AUC", "Pending")

strategy_results = backtests.copy()

if (
    not strategy_results.empty
    and "Is Benchmark" in strategy_results.columns
):
    strategy_results = strategy_results[
        ~strategy_results["Is Benchmark"].astype(bool)
    ]

if (
    not strategy_results.empty
    and "Sharpe Ratio" in strategy_results.columns
):
    metric4.metric(
        "Best Sharpe Ratio",
        f"{strategy_results['Sharpe Ratio'].max():.3f}",
    )
else:
    metric4.metric("Best Sharpe Ratio", "Pending")


st.divider()

left, right = st.columns([1.3, 1])

with left:
    st.subheader("Notebook Workflow")

    notebook_steps = [
        ("01", "Data preprocessing"),
        ("02", "Dataset construction"),
        ("03", "Feature engineering"),
        ("04", "Model training"),
        ("05", "Model evaluation"),
        ("06", "Portfolio backtesting"),
        ("07", "Robustness analysis"),
    ]

    for number, description in notebook_steps:
        st.markdown(f"**{number} — {description}**")

with right:
    st.subheader("Technologies")

    st.markdown(
        """
- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- TensorFlow or Keras
- ARCH / GARCH modeling
- Plotly
- Streamlit
"""
    )


st.divider()

st.subheader("Project Contribution")

st.markdown(
    """
This project demonstrates the ability to design a complete quantitative
research workflow, including data preparation, feature engineering, model
development, out-of-sample evaluation, and portfolio backtesting.

The Streamlit application presents the results in an interactive format for
recruiters, researchers, and other users.
"""
)


if classification.empty or backtests.empty:
    st.info(
        "Some result files are not connected yet. Run the robustness analysis "
        "notebook to populate the final model and backtest metrics."
    )