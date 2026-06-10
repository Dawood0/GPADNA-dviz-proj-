"""Heatmap visualizations for student habits and performance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from itertools import combinations
from plotly.subplots import make_subplots

from preprocessing import detect_columns

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

SLEEP_HOURS_BINS = [0, 4, 6, 8, 24]
SLEEP_HOURS_LABELS = ["0-4", "4-6", "6-8", "8+"]

EXAM_SCORE_BINS = [0, 50, 60, 70, 80, 90, 100]
EXAM_SCORE_LABELS = ["0-50", "50-60", "60-70", "70-80", "80-90", "90-100"]

SOCIAL_MEDIA_BINS = [0, 1, 2, 3, 4, 24]
SOCIAL_MEDIA_LABELS = ["0-1", "1-2", "2-3", "3-4", "4+"]

ATTENDANCE_BINS = [0, 70, 80, 90, 100]
ATTENDANCE_LABELS = ["<70%", "70-80%", "80-90%", "90-100%"]

STUDY_BINS = [0, 1, 2, 4, 6, 24]
STUDY_LABELS = ["0-1", "1-2", "2-4", "4-6", "6+"]

NETFLIX_BINS = [0, 1, 2, 3, 4, 24]
NETFLIX_LABELS = ["0-1", "1-2", "2-3", "3-4", "4+"]

DEFAULT_HEALTH_VARS = [option["value"] for option in AXIS_OPTIONS]
PAIRWISE_NUMERIC_VARS = [
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
]

BINS_MAP = {
    "study_hours_per_day": (STUDY_BINS, STUDY_LABELS),
    "social_media_hours": (SOCIAL_MEDIA_BINS, SOCIAL_MEDIA_LABELS),
    "netflix_hours": (NETFLIX_BINS, NETFLIX_LABELS),
    "attendance_percentage": (ATTENDANCE_BINS, ATTENDANCE_LABELS),
}

TITLE = "Comparison of the impacts of different factors on students' median scores"
EXPLANATION = (
    "Displays heatmaps comparing the median exam scores across different combinations of habits."
    "Use this chart to understand which factor has a significant impact on student performance when compared to any other factor."
)


def format_label(variable_name: str) -> str:
    return variable_name.replace("_", " ").title()


def prepare_categorical_ordering(df: pd.DataFrame, variable_name: str) -> pd.Series:
    if variable_name in CATEGORY_ORDERS:
        return pd.Categorical(df[variable_name], categories=CATEGORY_ORDERS[variable_name], ordered=True)
    return df[variable_name]


def _target_column(df: pd.DataFrame, target_col: str | None) -> str | None:
    if target_col and target_col in df.columns:
        return target_col
    detected = detect_columns(df)
    if detected["target_col"] in df.columns:
        return detected["target_col"]
    return "exam_score" if "exam_score" in df.columns else None


def make_heatmap_trace(df: pd.DataFrame, x_axis: str, y_axis: str, target_col: str) -> go.Heatmap | None:
    df = df.copy()
    if x_axis == "sleep_hours":
        df[x_axis] = pd.cut(
            df[x_axis],
            bins=SLEEP_HOURS_BINS,
            labels=SLEEP_HOURS_LABELS,
            include_lowest=True,
            right=False,
        )
    else:
        df[x_axis] = prepare_categorical_ordering(df, x_axis)

    if y_axis == "sleep_hours":
        df[y_axis] = pd.cut(
            df[y_axis],
            bins=SLEEP_HOURS_BINS,
            labels=SLEEP_HOURS_LABELS,
            include_lowest=True,
            right=False,
        )
    else:
        df[y_axis] = prepare_categorical_ordering(df, y_axis)

    pivot = df.pivot_table(
        index=y_axis,
        columns=x_axis,
        values=target_col,
        aggfunc="median",
    )

    if pivot.empty:
        return None

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


def make_all_heatmaps_figure(df: pd.DataFrame, variables: list[str], target_col: str) -> go.Figure:
    combos = list(combinations(variables, 2))
    cols = min(2, len(combos)) if combos else 1
    rows = (len(combos) + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{format_label(x)} vs {format_label(y)}" for x, y in combos],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, (x_axis, y_axis) in enumerate(combos):
        trace = make_heatmap_trace(df, x_axis, y_axis, target_col)
        if trace is None:
            continue
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(trace, row=row, col=col)
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


def make_pairwise_numeric_figure(df: pd.DataFrame, vars_list: list[str], target_col: str) -> go.Figure:
    combos = list(combinations(vars_list, 2))
    cols = min(2, len(combos)) if combos else 1
    rows = (len(combos) + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{format_label(x)} vs {format_label(y)}" for x, y in combos],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    for idx, (x, y) in enumerate(combos):
        xb, xl = BINS_MAP.get(x, ([], []))
        yb, yl = BINS_MAP.get(y, ([], []))
        sub_df = df.dropna(subset=[x, y]).copy()
        if not xb or not yb or sub_df.empty:
            continue
        sub_df[x] = pd.cut(sub_df[x], bins=xb, labels=xl, include_lowest=True, right=False)
        sub_df[y] = pd.cut(sub_df[y], bins=yb, labels=yl, include_lowest=True, right=False)
        pivot = sub_df.pivot_table(index=y, columns=x, values=target_col, aggfunc="median")
        if pivot.empty:
            continue

        trace = go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Blues",
            zmin=45,
            zmax=100,
            colorbar=dict(title="Median Score"),
            hovertemplate=(
                f"<b>{format_label(x)}:</b> %{{x}}<br>"
                f"<b>{format_label(y)}:</b> %{{y}}<br>"
                "<b>Median Exam Score:</b> %{z:.2f}<extra></extra>"
            ),
        )
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(title_text=format_label(x), row=row, col=col)
        fig.update_yaxes(title_text=format_label(y), row=row, col=col)

    fig.update_layout(
        height=780,
        autosize=True,
        dragmode=False,
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
    )
    return fig


def create_visual(df: pd.DataFrame, target_col: str | None = None, mode: str = "health") -> tuple[go.Figure, str, str]:
    target_column = _target_column(df, target_col)
    if not target_column:
        fig = go.Figure()
        fig.add_annotation(text="Heatmap requires a numeric target column like exam_score.", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", xaxis={"visible": False}, yaxis={"visible": False})
        return fig, TITLE, EXPLANATION

    if mode == "pairwise":
        figure = make_pairwise_numeric_figure(df, PAIRWISE_NUMERIC_VARS, target_column)
    else:
        figure = make_all_heatmaps_figure(df, DEFAULT_HEALTH_VARS, target_column)

    return figure, TITLE, EXPLANATION
