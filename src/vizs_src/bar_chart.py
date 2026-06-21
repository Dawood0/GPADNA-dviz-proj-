"""Bar chart visualization comparing habits by grade category."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.preprocessing import detect_columns, prepare_radar_data


TITLE = "Average Normalized Habit Values by Grade Category"
EXPLANATION = (
    "Compares average normalized habit values for high, average, and low grade "
    "students. Bars show how habits differ across performance groups."
)
COLORS = {
    "High grade": "#0f9f6e",
    "Medium/Average grade": "#d97706",
    "Low grade": "#dc2626",
}
TEMPLATE = "plotly_white"


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16, "color": "#64748b"})
    fig.update_layout(
        template=TEMPLATE,
        height=520,
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _threshold_labels(high_threshold: float, low_threshold: float, display_mode: str) -> dict[str, str]:
    labels = {
        "Low grade": f"Low grade: score <= {low_threshold:g}",
        "Medium/Average grade": f"Medium grade: {low_threshold:g} < score < {high_threshold:g}",
        "High grade": f"High grade: score >= {high_threshold:g}",
    }
    if display_mode == "high_low":
        labels.pop("Medium/Average grade", None)
    return labels


def _create_bar_chart(
    bar_data: pd.DataFrame,
    high_threshold: float,
    low_threshold: float,
    display_mode: str,
) -> go.Figure:
    if bar_data.empty:
        return _empty_figure("Bar chart needs a detected grade column and at least one selected feature.")

    fig = go.Figure()
    labels = _threshold_labels(high_threshold, low_threshold, display_mode)
    feature_labels = bar_data["feature_label"].drop_duplicates().tolist()
    
    # Add traces for each grade category
    for category in ["Low grade", "Medium/Average grade", "High grade"]:
        group = bar_data.loc[bar_data["grade_category"].eq(category)]
        if group.empty:
            continue
        
        group = group.set_index("feature_label").reindex(feature_labels).reset_index()
        values = group["normalized_average"].fillna(0).tolist()
        
        fig.add_trace(
            go.Bar(
                name=labels[category],
                x=feature_labels,
                y=values,
                marker={"color": COLORS[category]},
                hovertemplate="<b>%{x}</b><br>Normalized avg: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        height=520,
        margin={"l": 30, "r": 30, "t": 50, "b": 150},
        xaxis={"showgrid": False, "tickangle": -30},
        yaxis={"showgrid": True, "gridcolor": "lightgrey"},
        barmode="group",
        legend={"orientation": "v", "x": 1.01, "y": 1},
        font={"family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"},
    )
    return fig


def create_visual(df: pd.DataFrame, **kwargs) -> tuple[go.Figure, str, str]:
    detected = detect_columns(df)
    features = kwargs.get("features", detected["features"])
    target_col = kwargs.get("target_col", detected["target_col"])
    high_threshold = float(kwargs.get("high_threshold", 80))
    low_threshold = float(kwargs.get("low_threshold", 50))
    display_mode = kwargs.get("display_mode", "all")

    bar_data = prepare_radar_data(
        df,
        features,
        target_col,
        high_threshold,
        low_threshold,
        display_mode,
    )
    figure = _create_bar_chart(bar_data, high_threshold, low_threshold, display_mode)
    return figure, TITLE, EXPLANATION


def make_bar_chart(df: pd.DataFrame, **kwargs) -> go.Figure:
    figure, _, _ = create_visual(df, **kwargs)
    return figure