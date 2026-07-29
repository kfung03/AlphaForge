from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

st.title("Portfolio Backtesting")


@st.cache_data
def find_backtest_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    keywords = [
        "backtest",
        "wealth",
        "performance",
        "benchmark",
        "strategy",
        "equity",
    ]

    return sorted(
        path
        for path in directory.rglob("*.csv")
        if any(keyword in path.name.lower() for keyword in keywords)
    )


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


files = find_backtest_files(RESULTS_DIR)

if not files:
    st.warning(
        "No backtesting CSV files were found inside the results directory."
    )
    st.stop()


file_options = {
    str(path.relative_to(PROJECT_ROOT)): path
    for path in files
}

selected_name = st.selectbox(
    "Backtest file",
    list(file_options.keys()),
)

df = load_csv(file_options[selected_name])

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

numeric_columns = df.select_dtypes(include="number").columns.tolist()

date_candidates = [
    column
    for column in df.columns
    if "date" in column.lower() or "time" in column.lower()
]

value_candidates = [
    column
    for column in numeric_columns
    if any(
        term in column.lower()
        for term in [
            "wealth",
            "value",
            "cumulative",
            "return",
            "equity",
        ]
    )
]

if date_candidates and value_candidates:
    date_column = st.selectbox(
        "Date column",
        date_candidates,
    )

    value_column = st.selectbox(
        "Portfolio value column",
        value_candidates,
    )

    plot_df = df.copy()
    plot_df[date_column] = pd.to_datetime(
        plot_df[date_column],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[date_column, value_column],
    )

    figure = px.line(
        plot_df,
        x=date_column,
        y=value_column,
        title="Portfolio Performance",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )
else:
    st.info(
        "The selected file does not contain an obvious date and portfolio-value pair."
    )