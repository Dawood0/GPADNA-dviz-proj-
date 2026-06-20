"""Heatmap visualizations for student habits and performance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from itertools import combinations
from plotly.subplots import make_subplots

from src.preprocessing import detect_columns

AXIS_OPTIONS = [
    {"label": "Exercise Frequency", "value": "exercise_frequency"},
    {"label": "Mental Health Rating", "value": "mental_health_rating"},
    {"label": "Diet Quality", "value": "diet_quality"},
    {"label": "Sleep Hours", "value": "sleep_hours"},
]

CATEGORY_ORDERS = {
    "diet_quality": ["Poor", "Fair", "Good"],
    "parental_education_level": ["High School", "Bachelor", "Master"],
    "internet_quality": ["Poor", "Average", "Good"],
}

BINS_MAP = {
    "sleep_hours": ([0, 4, 6, 8, 24], ["0-4", "4-6", "6-8", "8+"]),
    "study_hours_per_day": ([0, 1, 2, 4, 6, 24], ["0-1", "1-2", "2-4", "4-6", "6+"]),
    "social_media_hours": ([0, 1, 2, 3, 4, 24], ["0-1", "1-2", "2-3", "3-4", "4+"]),
    "netflix_hours": ([0, 1, 2, 3, 4, 24], ["0-1", "1-2", "2-3", "3-4", "4+"]),
    "attendance_percentage": ([0, 70, 80, 90, 100], ["<70%", "70-80%", "80-90%", "90-100%"]),
}

DEFAULT_HEALTH_VARS = [option["value"] for option in AXIS_OPTIONS]
PAIRWISE_NUMERIC_VARS = [
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
]

TITLE = "Comparison of the impacts of different factors on students' median scores"
EXPLANATION = (
    "Displays heatmaps comparing the median exam scores across different combinations of habits. "
    "Use this chart to understand which factor has a significant impact on student performance when compared to any other factor."
)


def format_label(variable_name):
    return variable_name.replace("_", " ").title()


def prepare_axis(df, variable_name):
    if variable_name in BINS_MAP:
        bins, labels = BINS_MAP[variable_name]
        return pd.cut(df[variable_name], bins=bins, labels=labels, include_lowest=True, right=False)
    if variable_name in CATEGORY_ORDERS:
        return pd.Categorical(df[variable_name], categories=CATEGORY_ORDERS[variable_name], ordered=True)
    return df[variable_name]


def _target_column(df, target_col):
    if target_col and target_col in df.columns:
        return target_col
    detected = detect_columns(df)
    return detected["target_col"] if detected["target_col"] in df.columns else "exam_score"


def make_heatmap_trace(df, x_axis, y_axis, target_col):
    df = df.assign(**{x_axis: prepare_axis(df, x_axis), y_axis: prepare_axis(df, y_axis)})
    pivot = df.pivot_table(index=y_axis, columns=x_axis, values=target_col, aggfunc="median")

    return go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="Blues",
        zmin=45,
        zmax=100,
        colorbar=dict(title="Median Score"),
        hovertemplate=(
            f"<b>{format_label(x_axis)}:</b> %{{x}}<br>"
            f"<b>{format_label(y_axis)}:</b> %{{y}}<br>"
            "<b>Median Exam Score:</b> %{z:.2f}<extra></extra>"
        ),
        name=f"{format_label(x_axis)} vs {format_label(y_axis)}",
    )


def make_grid_figure(df, variables, target_col):
    combos = list(combinations(variables, 2))
    cols = min(2, len(combos))
    rows = (len(combos) + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{format_label(x)} vs {format_label(y)}" for x, y in combos],
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )

    for idx, (x_axis, y_axis) in enumerate(combos):
        row, col = idx // cols + 1, idx % cols + 1
        fig.add_trace(make_heatmap_trace(df, x_axis, y_axis, target_col), row=row, col=col)
        fig.update_xaxes(title_text=format_label(x_axis), row=row, col=col)
        fig.update_yaxes(title_text=format_label(y_axis), row=row, col=col)

    fig.update_layout(
        height=780,
        autosize=True,
        dragmode=False,
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    return fig


def create_visual(df, target_col, mode):
    target_column = _target_column(df, target_col)
    variables = PAIRWISE_NUMERIC_VARS if mode == "pairwise" else DEFAULT_HEALTH_VARS
    figure = make_grid_figure(df, variables, target_column)
    return figure, TITLE, EXPLANATION