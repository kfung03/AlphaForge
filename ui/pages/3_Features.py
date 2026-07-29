from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIRS = [
    PROJECT_ROOT / "features",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "results",
]


st.title("Feature Engineering")

st.markdown(
    """
AlphaForge combines three complementary sources of predictive information:

1. **Fama–French factors** capture broad economic and style-related return drivers.
2. **GARCH features** measure changing volatility and market uncertainty.
3. **Path-signature features** summarize the ordered behavior of multivariate
   financial time series.

The combined feature set tests whether economic, volatility, and path-dependent
information improve classification performance when used together.
"""
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_data
def find_feature_files(directories: list[Path]) -> list[Path]:
    feature_files: list[Path] = []

    keywords = [
        "feature",
        "signature",
        "logsig",
        "garch",
        "combined",
        "train",
        "validation",
        "test",
    ]

    for directory in directories:
        if not directory.exists():
            continue

        for path in directory.rglob("*.csv"):
            filename = path.name.lower()

            if any(keyword in filename for keyword in keywords):
                feature_files.append(path)

    return sorted(set(feature_files))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


# ---------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------

st.divider()
st.subheader("Feature Pipeline")

pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = st.columns(4)

with pipeline_col1:
    st.markdown(
        """
### 1. Rolling Windows

Historical observations are grouped into rolling time windows while preserving
the order of returns and factor values.
"""
    )

with pipeline_col2:
    st.markdown(
        """
### 2. Log-Signatures

Log-signature transforms summarize the shape, ordering, and interactions of the
multivariate path inside each window.
"""
    )

with pipeline_col3:
    st.markdown(
        """
### 3. GARCH Statistics

GARCH models estimate conditional volatility and produce features related to
time-varying risk.
"""
    )

with pipeline_col4:
    st.markdown(
        """
### 4. Feature Fusion

Signature and GARCH outputs are combined into a single matrix for CNN,
XGBoost, Random Forest, and ensemble models.
"""
    )


# ---------------------------------------------------------
# Feature-family explanation
# ---------------------------------------------------------

st.divider()
st.subheader("Feature Families")

selected_family = st.radio(
    "Select a feature family",
    [
        "Fama–French Factors",
        "Log-Signature Features",
        "GARCH Features",
        "Combined Features",
    ],
    horizontal=True,
)


if selected_family == "Fama–French Factors":
    st.markdown(
        """
### Fama–French Factors

The project uses Fama–French factor data as the economic foundation of the
modeling pipeline.

The three-factor specification contains:

- Market excess return
- Size
- Value

The five-factor specification adds:

- Profitability
- Investment

These factors provide an economically interpretable description of systematic
return variation.
"""
    )

elif selected_family == "Log-Signature Features":
    st.markdown(
        """
### Log-Signature Features

Log-signatures encode the ordered path followed by a multivariate time series.

Unlike summary statistics that ignore sequencing, signature-based features can
capture:

- Directional movements
- Interactions between variables
- Lead-lag structure
- Nonlinear path behavior
- Higher-order temporal information

The signature depth controls the level of interaction represented in the final
feature vector. Greater depth creates richer features but also increases model
complexity.
"""
    )

elif selected_family == "GARCH Features":
    st.markdown(
        """
### GARCH Features

GARCH models estimate conditional volatility rather than assuming that return
variance stays constant.

Potential GARCH-derived features include:

- Conditional volatility
- Conditional variance
- Standardized residuals
- Lagged volatility
- Volatility changes
- Model parameters and summary statistics

These features provide information about changing risk conditions that may not
be captured by factor returns alone.
"""
    )

else:
    st.markdown(
        """
### Combined Features

The combined feature matrix joins log-signature features with GARCH-derived
statistics.

This feature-fusion approach tests whether the two methods contain complementary
information:

- Log-signatures describe the structure of the historical path.
- GARCH describes the evolving level of conditional risk.

The combined representation is evaluated against the individual feature
families to determine whether fusion improves out-of-sample performance.
"""
    )


# ---------------------------------------------------------
# Experimental configurations
# ---------------------------------------------------------

st.divider()
st.subheader("Experimental Configurations")

config_col1, config_col2 = st.columns(2)

with config_col1:
    st.markdown(
        """
#### Dataset configurations

- Monthly Fama–French three-factor
- Monthly Fama–French five-factor
- Daily Fama–French three-factor
- Daily Fama–French five-factor
"""
    )

with config_col2:
    st.markdown(
        """
#### Feature comparisons

- Signature-only features
- GARCH-only features
- Combined signature and GARCH features
- Multiple signature-depth configurations
"""
    )


# ---------------------------------------------------------
# Interactive feature-file explorer
# ---------------------------------------------------------

st.divider()
st.subheader("Feature Data Explorer")

feature_files = find_feature_files(FEATURE_DIRS)

if not feature_files:
    st.info(
        "No feature CSV files were detected. Run the feature-engineering "
        "notebook or confirm that generated features are stored inside the "
        "`features`, `data/processed`, or `results` directories."
    )

else:
    file_options = {
        str(path.relative_to(PROJECT_ROOT)): path
        for path in feature_files
    }

    selected_filename = st.selectbox(
        "Select a generated feature file",
        list(file_options.keys()),
    )

    selected_path = file_options[selected_filename]

    try:
        feature_df = load_csv(selected_path)
    except Exception as exc:
        st.error(f"Unable to load the selected file: {exc}")
        st.stop()

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Rows", f"{len(feature_df):,}")
    metric2.metric("Columns", f"{len(feature_df.columns):,}")
    metric3.metric(
        "Numeric Features",
        f"{len(get_numeric_columns(feature_df)):,}",
    )
    metric4.metric(
        "Missing Values",
        f"{int(feature_df.isna().sum().sum()):,}",
    )

    st.caption(f"Source: `{selected_path.relative_to(PROJECT_ROOT)}`")

    st.subheader("Feature Preview")

    st.dataframe(
        feature_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    numeric_columns = get_numeric_columns(feature_df)

    if numeric_columns:
        st.subheader("Feature Distribution")

        selected_feature = st.selectbox(
            "Choose a numeric feature",
            numeric_columns,
        )

        histogram = px.histogram(
            feature_df,
            x=selected_feature,
            nbins=50,
            title=f"Distribution of {selected_feature}",
        )

        histogram.update_layout(
            xaxis_title=selected_feature,
            yaxis_title="Observations",
            height=450,
        )

        st.plotly_chart(
            histogram,
            use_container_width=True,
        )

        if len(numeric_columns) >= 2:
            st.subheader("Feature Relationship")

            relationship_col1, relationship_col2 = st.columns(2)

            x_feature = relationship_col1.selectbox(
                "X-axis feature",
                numeric_columns,
                index=0,
            )

            y_feature = relationship_col2.selectbox(
                "Y-axis feature",
                numeric_columns,
                index=min(1, len(numeric_columns) - 1),
            )

            scatter = px.scatter(
                feature_df,
                x=x_feature,
                y=y_feature,
                opacity=0.65,
                title=f"{y_feature} vs {x_feature}",
            )

            scatter.update_layout(height=480)

            st.plotly_chart(
                scatter,
                use_container_width=True,
            )

        st.subheader("Descriptive Statistics")

        summary = (
            feature_df[numeric_columns]
            .describe()
            .transpose()
            .reset_index()
            .rename(columns={"index": "Feature"})
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.warning(
            "The selected file does not contain numeric feature columns."
        )


# ---------------------------------------------------------
# Modeling connection
# ---------------------------------------------------------

st.divider()
st.subheader("Connection to the Models")

st.markdown(
    """
The engineered feature matrices are passed into several model families:

| Model | Role |
|---|---|
| CNN | Learns nonlinear patterns from structured feature inputs |
| Random Forest | Provides a robust tree-based baseline |
| XGBoost | Captures nonlinear interactions through boosted trees |
| Ensemble | Combines model probabilities to improve stability |

All feature families are evaluated using the same out-of-sample test framework,
allowing a direct comparison between signature, GARCH, and combined inputs.
"""
)