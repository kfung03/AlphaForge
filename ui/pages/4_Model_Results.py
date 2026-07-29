from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

st.title("Model Evaluation")


@st.cache_data
def find_result_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    keywords = [
        "classification",
        "model_comparison",
        "evaluation",
        "metrics",
        "results",
    ]

    files = []

    for path in directory.rglob("*.csv"):
        name = path.name.lower()

        if any(keyword in name for keyword in keywords):
            files.append(path)

    return sorted(files)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


result_files = find_result_files(RESULTS_DIR)

if not result_files:
    st.warning(
        "No model-evaluation CSV files were found inside the results folder."
    )
    st.stop()


file_options = {
    str(path.relative_to(PROJECT_ROOT)): path
    for path in result_files
}

selected_file = st.selectbox(
    "Evaluation file",
    list(file_options.keys()),
)

df = load_csv(file_options[selected_file])

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

numeric_columns = df.select_dtypes(include="number").columns.tolist()
text_columns = df.select_dtypes(exclude="number").columns.tolist()

metric_candidates = [
    column
    for column in numeric_columns
    if any(
        word in column.lower()
        for word in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "score",
        ]
    )
]

if not metric_candidates:
    st.info(
        "The selected file does not contain recognizable evaluation metrics."
    )
    st.stop()


selected_metric = st.selectbox(
    "Metric",
    metric_candidates,
)

best_index = df[selected_metric].idxmax()
best_row = df.loc[best_index]

st.metric(
    f"Best {selected_metric}",
    f"{best_row[selected_metric]:.4f}",
)

if text_columns:
    category_column = st.selectbox(
        "Category displayed on the chart",
        text_columns,
    )

    color_options = ["None"] + [
        column
        for column in text_columns
        if column != category_column
    ]

    selected_color = st.selectbox(
        "Group by",
        color_options,
    )

    figure = px.bar(
        df,
        x=category_column,
        y=selected_metric,
        color=None if selected_color == "None" else selected_color,
        title=f"{selected_metric} Comparison",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )