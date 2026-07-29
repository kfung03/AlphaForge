import streamlit as st


st.title("About the Project")

st.markdown(
    """
AlphaForge is a quantitative machine-learning research platform built to study
whether advanced time-series features can improve financial return prediction
and portfolio construction.

The project was developed as a complete research pipeline rather than as a
single modeling notebook. It includes data preparation, dataset construction,
feature engineering, model training, evaluation, backtesting, and robustness
analysis.
"""
)


# ---------------------------------------------------------
# Project objective
# ---------------------------------------------------------

st.divider()
st.subheader("Project Objective")

st.markdown(
    """
The central research question is:

> Can Fama–French factors, GARCH volatility statistics, and path-signature
> features be combined to improve the identification of high-return portfolios?

The analysis evaluates this question across both daily and monthly data and
under Fama–French three-factor and five-factor specifications.
"""
)


# ---------------------------------------------------------
# What was built
# ---------------------------------------------------------

st.divider()
st.subheader("What I Built")

build_col1, build_col2 = st.columns(2)

with build_col1:
    st.markdown(
        """
#### Research and modeling

- Processed financial return and factor datasets
- Constructed daily and monthly rolling samples
- Generated log-signature features
- Estimated GARCH volatility features
- Built combined feature matrices
- Trained CNN and tree-based classifiers
- Tuned classification thresholds
"""
    )

with build_col2:
    st.markdown(
        """
#### Evaluation and presentation

- Compared accuracy, precision, recall, F1, and ROC-AUC
- Converted predicted probabilities into portfolio signals
- Calculated return and risk statistics
- Compared strategies against benchmarks
- Performed cross-experiment robustness analysis
- Built this interactive Streamlit application
"""
    )


# ---------------------------------------------------------
# Technical skills
# ---------------------------------------------------------

st.divider()
st.subheader("Technical Skills Demonstrated")

skill_col1, skill_col2, skill_col3 = st.columns(3)

with skill_col1:
    st.markdown(
        """
### Financial Modeling

- Fama–French factors
- Pricing-error analysis
- Conditional volatility
- GARCH modeling
- Portfolio construction
- Backtesting
- Sharpe ratio
- Maximum drawdown
"""
    )

with skill_col2:
    st.markdown(
        """
### Machine Learning

- Binary classification
- CNN models
- Random Forest
- XGBoost
- Ensemble learning
- Threshold optimization
- Time-series validation
- Feature comparison
"""
    )

with skill_col3:
    st.markdown(
        """
### Engineering

- Python
- pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- Plotly
- Streamlit
- Git and GitHub
"""
    )


# ---------------------------------------------------------
# Repository structure
# ---------------------------------------------------------

st.divider()
st.subheader("Repository Structure")

st.code(
    """
alphaforge/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_build_datasets_parameterized.ipynb
│   ├── 03_feature_engineering_parameterized.ipynb
│   ├── 04_model_training_parameterized.ipynb
│   ├── 05_model_evaluation_parameterized.ipynb
│   ├── 06_portfolio_backtest_parameterized.ipynb
│   └── 07_robustness_analysis_expanded.ipynb
│
├── src/
│   └── alphaforge/
│
├── features/
├── models/
├── results/
│   └── robustness_analysis/
│
└── ui/
    ├── streamlit_app.py
    └── pages/
""",
    language="text",
)


# ---------------------------------------------------------
# Notebook workflow
# ---------------------------------------------------------

st.divider()
st.subheader("Research Workflow")

workflow = [
    {
        "step": "01",
        "title": "Data Preprocessing",
        "description": (
            "Clean and standardize the raw financial and factor datasets."
        ),
    },
    {
        "step": "02",
        "title": "Dataset Construction",
        "description": (
            "Build parameterized daily and monthly train, validation, and test datasets."
        ),
    },
    {
        "step": "03",
        "title": "Feature Engineering",
        "description": (
            "Generate log-signature, GARCH, and combined feature matrices."
        ),
    },
    {
        "step": "04",
        "title": "Model Training",
        "description": (
            "Train neural-network, tree-based, and ensemble classifiers."
        ),
    },
    {
        "step": "05",
        "title": "Model Evaluation",
        "description": (
            "Evaluate out-of-sample predictions and compare classification metrics."
        ),
    },
    {
        "step": "06",
        "title": "Portfolio Backtesting",
        "description": (
            "Convert model probabilities into systematic investment strategies."
        ),
    },
    {
        "step": "07",
        "title": "Robustness Analysis",
        "description": (
            "Aggregate and compare results across every experiment."
        ),
    },
]

for item in workflow:
    with st.expander(
        f"{item['step']} — {item['title']}"
    ):
        st.write(item["description"])


# ---------------------------------------------------------
# Resume section
# ---------------------------------------------------------

st.divider()
st.subheader("Resume Summary")

st.markdown(
    """
**AlphaForge — Quantitative Machine-Learning Research Platform**

- Developed an end-to-end financial machine-learning pipeline combining
  Fama–French factors, GARCH volatility estimates, and log-signature features
  to classify high-return portfolios using daily and monthly market data.
- Trained and evaluated CNN, Random Forest, XGBoost, and ensemble models using
  out-of-sample accuracy, F1 score, and ROC-AUC.
- Designed a portfolio-backtesting framework to convert model probabilities
  into investment signals and evaluate annualized return, volatility, Sharpe
  ratio, and maximum drawdown.
- Built an interactive Streamlit dashboard to communicate research methods,
  model comparisons, and portfolio results.
"""
)


# ---------------------------------------------------------
# Limitations
# ---------------------------------------------------------

st.divider()
st.subheader("Research Limitations")

st.markdown(
    """
This application presents historical research results and is not intended to
provide investment advice.

Relevant limitations include:

- Historical performance does not guarantee future results.
- Backtests may not fully capture transaction costs or market impact.
- Financial time series can experience structural changes.
- Model performance depends on the chosen sample period and target definition.
- Classification performance does not automatically imply economic value.
"""
)


# ---------------------------------------------------------
# Contact section
# ---------------------------------------------------------

st.divider()
st.subheader("Contact")

st.markdown(
    """
**Kyler Fung**

Finance and quantitative-machine-learning student

- GitHub: `https://github.com/kfung03`
- LinkedIn: `https://www.linkedin.com/in/kyler-fung`
- Email: `kylerfung03@gmail.com`
"""
)

st.caption(
    "Replace the contact placeholders above with your real information before deployment."
)