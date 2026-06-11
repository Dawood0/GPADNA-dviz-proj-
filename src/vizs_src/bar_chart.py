"""Bar chart visualization comparing habits by performance group."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

TITLE = "Average Habits by Performance Group"
EXPLANATION = (
    "Compares average habit values across top performers (score >= 80%), "
    "low performers (score <= 40%), and all students."
)
TEMPLATE = "plotly_white"
COLORS = {
    "Top Performers (Score above 80%)": "#0f9f6e",
    "Low Performers (Score below 40%)": "#dc2626",
    "All Students": "#d97706",
}

HABIT_MAPPING = {
    "study_hours_per_day": "Study Hours/Day",
    "sleep_hours": "Sleep Hours",
    "social_media_hours": "Social Media Hours",
    "netflix_hours": "Netflix Hours",
    "exercise_frequency": "Exercise Frequency",
    "mental_health_rating": "Mental Health Rating",
    "diet_quality_num": "Diet Quality (1=Poor, 3=Good)",
    "part_time_job_num": "Part-Time Job (0=No, 1=Yes)",
    "extracurricular_num": "Extracurricular (0=No, 1=Yes)",
    "internet_quality_num": "Internet Quality (1=Poor, 3=Good)",
}


def _prepare_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numeric values."""
    df_copy = df.copy()
    df_copy["diet_quality_num"] = df_copy["diet_quality"].map({"Poor": 1, "Fair": 2, "Good": 3})
    df_copy["part_time_job_num"] = df_copy["part_time_job"].map({"No": 0, "Yes": 1})
    df_copy["extracurricular_num"] = df_copy["extracurricular_participation"].map({"No": 0, "Yes": 1})
    df_copy["internet_quality_num"] = df_copy["internet_quality"].map({"Poor": 1, "Fair": 2, "Good": 3})
    df_copy["parental_education_num"] = df_copy["parental_education_level"].map({
        "None": 0, "High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4
    })
    df_copy["gender_num"] = df_copy["gender"].map({"Male": 0, "Female": 1, "Other": 2})
    return df_copy


def _extract_habit_keys_and_labels() -> tuple[list[str], list[str]]:
    """Extract habit column names and their display labels."""
    habit_keys = list(HABIT_MAPPING.keys())
    habit_labels = list(HABIT_MAPPING.values())
    return habit_keys, habit_labels


def make_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Create bar chart comparing habit averages by performance group.
    
    Args:
        df: DataFrame containing student habits and exam scores.
        
    Returns:
        A Plotly Figure object displaying the bar chart.
    """
    df = _prepare_categorical_data(df)
    habit_keys, habit_labels = _extract_habit_keys_and_labels()
    
    # Filter by performance groups
    top = df[df["exam_score"] >= df["exam_score"].quantile(0.80)]
    low = df[df["exam_score"] <= df["exam_score"].quantile(0.40)]
    
    # Calculate averages
    top_avg = top[habit_keys].mean()
    low_avg = low[habit_keys].mean()
    all_avg = df[habit_keys].mean()
    
    # Build chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Top Performers (Score above 80%)",
        x=habit_labels,
        y=top_avg,
        marker_color=COLORS["Top Performers (Score above 80%)"],
        hovertemplate="<b>%{x}</b><br>Top avg: %{y:.2f}<extra></extra>",
    ))
    
    fig.add_trace(go.Bar(
        name="Low Performers (Score below 40%)",
        x=habit_labels,
        y=low_avg,
        marker_color=COLORS["Low Performers (Score below 40%)"],
        hovertemplate="<b>%{x}</b><br>Low avg: %{y:.2f}<extra></extra>",
    ))
    
    fig.add_trace(go.Bar(
        name="All Students",
        x=habit_labels,
        y=all_avg,
        marker_color=COLORS["All Students"],
        hovertemplate="<b>%{x}</b><br>Overall avg: %{y:.2f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=TITLE,
        template=TEMPLATE,
        xaxis_title="Habit",
        yaxis_title="Average Value",
        barmode="group",
        height=520,
        margin={"l": 30, "r": 30, "t": 50, "b": 150},
        xaxis={"showgrid": False, "tickangle": -30},
        yaxis={"showgrid": True, "gridcolor": "lightgrey"},
        legend={"orientation": "v", "x": 1.01, "y": 1},
    )
    
    return fig