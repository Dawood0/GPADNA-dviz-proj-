"""Radar visualization for student habits and academic performance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from preprocessing import detect_columns, prepare_radar_data


TITLE = "Average Normalized Habit Values by Grade Category"
EXPLANATION = (
    "Compares average normalized habit values for high, average, and low grade "
    "students. Larger gaps indicate habits that differ more strongly by grade category."
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


def _create_radar_chart(
    radar_data: pd.DataFrame,
    high_threshold: float,
    low_threshold: float,
    display_mode: str,
) -> go.Figure:
    if radar_data.empty:
        return _empty_figure("Radar chart needs a detected grade column and at least one selected feature.")

    fig = go.Figure()
    labels = _threshold_labels(high_threshold, low_threshold, display_mode)
    features = radar_data["feature_label"].drop_duplicates().tolist()
    for category in labels:
        group = radar_data.loc[radar_data["grade_category"].eq(category)]
        if group.empty:
            continue
        group = group.set_index("feature_label").reindex(features).reset_index()
        values = group["normalized_average"].fillna(0).tolist()
        custom = list(
            zip(
                group["actual_label"].fillna("Unavailable"),
                group["student_count"].fillna(0).astype(int),
                group["feature"].fillna(""),
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=features + features[:1],
                customdata=custom + custom[:1],
                mode="lines+markers",
                fill="toself",
                name=labels[category],
                line={"color": COLORS[category], "width": 3},
                marker={"size": 6},
                hovertemplate=(
                    "<b>%{theta}</b><br>"
                    "Normalized average: %{r:.2f}<br>"
                    "%{customdata[0]}<br>"
                    "Students: %{customdata[1]}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        title=TITLE,
        height=620,
        margin={"l": 40, "r": 40, "t": 70, "b": 40},
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        legend={"orientation": "h", "y": -0.08, "x": 0},
        font={"family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"},
    )
    return fig


def create_visual(df: pd.DataFrame, **kwargs) -> tuple[go.Figure, str, str]:
    """Return the radar figure, title, and explanation for the supplied data."""
    detected = detect_columns(df)
    features = kwargs.get("features", detected["features"])
    target_col = kwargs.get("target_col", detected["target_col"])
    high_threshold = float(kwargs.get("high_threshold", 80))
    low_threshold = float(kwargs.get("low_threshold", 50))
    display_mode = kwargs.get("display_mode", "all")

    radar_data = prepare_radar_data(
        df,
        features,
        target_col,
        high_threshold,
        low_threshold,
        display_mode,
    )
    figure = _create_radar_chart(radar_data, high_threshold, low_threshold, display_mode)
    return figure, TITLE, EXPLANATION
