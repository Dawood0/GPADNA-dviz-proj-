from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "student_habits_performance.csv"

POSITIVE_KEYWORDS = [
    "study",
    "sleep",
    "attendance",
    "exercise",
    "diet",
    "internet",
    "education",
    "parental_education",
    "extracurricular",
    "participation",
    "health_score",
    "health_rating",
    "mental_health_rating",
    "motivation",
]
NEGATIVE_KEYWORDS = [
    "social_media",
    "netflix",
    "tv",
    "part_time_job",
    "job",
    "stress",
    "absence",
    "absences",
    "mental_health_problem",
    "mental_health_issue",
    "procrastination",
]
TARGET_KEYWORDS = ["exam_score", "final_score", "test_score", "grade", "score", "performance"]
ID_KEYWORDS = ["student_id", "studentid", "id"]
ORDINAL_CATEGORY_MAPS = {
    "diet_quality": {"poor": 0, "fair": 1, "average": 1, "good": 2},
    "internet_quality": {"poor": 0, "fair": 1, "average": 1, "good": 2},
    "parental_education_level": {"high school": 0, "bachelor": 1, "bachelors": 1, "master": 2, "masters": 2, "phd": 3},
    "extracurricular_participation": {"no": 0, "yes": 1},
    "part_time_job": {"no": 0, "yes": 1},
}
NOMINAL_FEATURES = {"gender"}


def clean_column_name(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())).strip("_")


def pretty_name(name: str | None) -> str:
    return "Not detected" if not name else name.replace("_", " ").title()


def load_data(path: str | Path = DATA_FILE) -> pd.DataFrame:
    """Load, clean column names, strip text values, and coerce numeric-like columns."""
    df = pd.read_csv(path)
    df = df.rename(columns={col: clean_column_name(col) for col in df.columns})
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().mean() >= 0.9:
                df[col] = numeric
    return df


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, object]:
    """Detect the grade, student id, and usable feature columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target_col = _first_existing(numeric_cols, TARGET_KEYWORDS)
    if target_col is None:
        scored = [col for col in numeric_cols if any(token in col for token in TARGET_KEYWORDS)]
        target_col = scored[0] if scored else None

    id_col = _first_existing(df.columns.tolist(), ID_KEYWORDS)
    if id_col is None:
        id_matches = [col for col in df.columns if "student" in col and "id" in col]
        id_col = id_matches[0] if id_matches else None

    excluded = {target_col, id_col}
    features = [
        col
        for col in df.columns
        if col not in excluded and not col.endswith("_id") and df[col].nunique(dropna=True) > 1
    ]
    numeric_features = [
        col
        for col in numeric_cols
        if col in features
    ]
    categorical_features = [col for col in features if col not in numeric_features]
    return {
        "target_col": target_col,
        "student_id_col": id_col,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_cols": numeric_cols,
    }


def classify_features_positive_negative(features: list[str]) -> dict[str, list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    positive_exact = {
        "study_hours_per_day",
        "attendance_percentage",
        "sleep_hours",
        "diet_quality",
        "exercise_frequency",
        "parental_education_level",
        "internet_quality",
        "mental_health_rating",
        "extracurricular_participation",
    }
    negative_exact = {
        "social_media_hours",
        "netflix_hours",
        "part_time_job",
    }
    for feature in features:
        compact = feature.lower()
        is_positive = feature in positive_exact or any(token in compact for token in POSITIVE_KEYWORDS)
        is_negative = feature in negative_exact or any(token in compact for token in NEGATIVE_KEYWORDS)
        if feature in positive_exact:
            positive.append(feature)
        elif feature in negative_exact:
            negative.append(feature)
        elif is_positive and not is_negative:
            positive.append(feature)
        elif is_negative and not is_positive:
            negative.append(feature)
        elif "mental_health_rating" in compact or ("health" in compact and ("score" in compact or "rating" in compact)):
            positive.append(feature)
    return {"all": features, "positive": positive, "negative": negative}


def classify_features_by_grade_pattern(
    df: pd.DataFrame,
    features: list[str],
    target_col: str | None,
    high_threshold: float,
    low_threshold: float,
) -> dict[str, list[str]]:
    categorized = create_grade_categories(df, target_col, high_threshold, low_threshold, "all")
    if not target_col or target_col not in categorized.columns or high_threshold <= low_threshold:
        return {"all": features, "positive": [], "negative": []}

    normalized = normalize_features(categorized, [feature for feature in features if feature in categorized.columns])
    positive: list[str] = []
    negative: list[str] = []
    for feature in features:
        if feature not in normalized.columns or feature in NOMINAL_FEATURES:
            continue
        green_values = normalized.loc[categorized["grade_category"].eq("High grade"), feature].dropna()
        red_values = normalized.loc[categorized["grade_category"].eq("Low grade"), feature].dropna()
        if green_values.empty or red_values.empty:
            continue
        green_average = float(green_values.mean())
        red_average = float(red_values.mean())
        if green_average > red_average:
            positive.append(feature)
        elif red_average > green_average:
            negative.append(feature)

    return {"all": features, "positive": positive, "negative": negative}


def select_features(feature_groups: dict[str, list[str]], feature_filter: str) -> list[str]:
    if feature_filter == "positive":
        return feature_groups.get("positive", [])
    if feature_filter == "negative":
        return feature_groups.get("negative", [])
    return feature_groups.get("all", [])


def default_thresholds(df: pd.DataFrame, target_col: str | None) -> tuple[float, float]:
    if not target_col or target_col not in df.columns:
        return 75.0, 25.0
    scores = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if scores.empty:
        return 75.0, 25.0
    return float(scores.quantile(0.75)), float(scores.quantile(0.25))


def score_bounds(df: pd.DataFrame, target_col: str | None) -> tuple[float, float]:
    if not target_col or target_col not in df.columns:
        return 0.0, 100.0
    scores = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if scores.empty:
        return 0.0, 100.0
    return float(math.floor(scores.min())), float(math.ceil(scores.max()))


def create_grade_categories(
    df: pd.DataFrame,
    target_col: str | None,
    high_threshold: float,
    low_threshold: float,
    display_mode: str = "all",
) -> pd.DataFrame:
    out = df.copy()
    if not target_col or target_col not in out.columns or high_threshold <= low_threshold:
        out["grade_category"] = pd.Series(["Unavailable"] * len(out), index=out.index, dtype="string")
        return out

    scores = pd.to_numeric(out[target_col], errors="coerce")
    out["grade_category"] = np.select(
        [scores >= high_threshold, scores <= low_threshold],
        ["High grade", "Low grade"],
        default="Medium/Average grade",
    )
    out.loc[scores.isna(), "grade_category"] = "Unavailable"
    if display_mode == "high_low":
        out = out.loc[out["grade_category"].isin(["High grade", "Low grade"])].copy()
    return out


def normalize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    normalized = pd.DataFrame(index=df.index)
    for feature in features:
        values = _feature_numeric_values(df[feature], feature)
        span = values.max() - values.min()
        if pd.isna(span) or span == 0:
            normalized[feature] = 0.5
        else:
            normalized[feature] = (values - values.min()) / span
    return normalized


def prepare_radar_data(
    df: pd.DataFrame,
    features: list[str],
    target_col: str | None,
    high_threshold: float,
    low_threshold: float,
    display_mode: str,
) -> pd.DataFrame:
    categorized = create_grade_categories(df, target_col, high_threshold, low_threshold, display_mode)
    valid_features = [feature for feature in features if feature in categorized.columns]
    if not valid_features or "grade_category" not in categorized.columns:
        return pd.DataFrame()

    normalized = normalize_features(categorized, valid_features)
    normalized["grade_category"] = categorized["grade_category"]

    rows = []
    for category in _category_order(display_mode):
        mask = normalized["grade_category"].eq(category)
        if not mask.any():
            continue
        for feature in valid_features:
            rows.append(
                {
                    "grade_category": category,
                    "feature": feature,
                    "feature_label": pretty_name(feature),
                    "normalized_average": normalized.loc[mask, feature].mean(),
                    "actual_average": _feature_average_value(categorized.loc[mask, feature], feature),
                    "actual_label": _feature_summary(categorized.loc[mask, feature]),
                    "student_count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


NUMERIC_FEATURE_LEVELS = ["Low feature value", "Medium feature value", "High feature value"]


def _is_numeric_like(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().mean() >= 0.9


def _feature_numeric_values(series: pd.Series, feature: str | None = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        return numeric
    labels = series.astype("string").fillna("Missing")
    if feature in ORDINAL_CATEGORY_MAPS:
        mapping = ORDINAL_CATEGORY_MAPS[feature]
        return labels.str.lower().map(mapping).astype(float)
    categories = sorted(labels.dropna().unique().tolist())
    return labels.map({category: index for index, category in enumerate(categories)}).astype(float)


def _feature_average_value(series: pd.Series, feature: str | None = None) -> float:
    return float(_feature_numeric_values(series, feature).mean())


def _feature_summary(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        return f"Average: {numeric.mean():.2f}"
    counts = series.astype("string").fillna("Missing").value_counts(dropna=False)
    if counts.empty:
        return "Most common: unavailable"
    top = counts.index[0]
    share = counts.iloc[0] / counts.sum() * 100
    return f"Most common: {top} ({share:.0f}%)"


def _dot_positions(count: int, columns: int = 10) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.arange(count)
    return indexes % columns, -(indexes // columns)


def _feature_levels(values: pd.Series, feature: str | None = None) -> tuple[pd.Series, dict[str, int]]:
    if not _is_numeric_like(values):
        labels = values.astype("string").fillna("Missing")
        if feature in ORDINAL_CATEGORY_MAPS:
            order = ORDINAL_CATEGORY_MAPS[feature]
            categories = sorted(labels.dropna().unique().tolist(), key=lambda value: order.get(str(value).lower(), len(order)))
        else:
            categories = sorted(labels.dropna().unique().tolist())
        rank = {category: index for index, category in enumerate(categories)}
        return labels, rank

    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(["Feature value unavailable"] * len(values), index=values.index, dtype="string")
    valid = numeric.dropna()
    if valid.empty:
        return result, {"Feature value unavailable": 0}
    try:
        binned = pd.qcut(valid.rank(method="first"), q=3, labels=NUMERIC_FEATURE_LEVELS).astype("string")
        result.loc[binned.index] = binned
        return result, {level: index for index, level in enumerate(NUMERIC_FEATURE_LEVELS)}
    except ValueError:
        return result, {"Feature value unavailable": 0}


def prepare_dot_matrix_data(
    df: pd.DataFrame,
    features: list[str],
    target_col: str | None,
    student_id_col: str | None,
    high_threshold: float,
    low_threshold: float,
    display_mode: str,
    max_students: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    categorized = create_grade_categories(df, target_col, high_threshold, low_threshold, display_mode)
    metadata = {"sampled": False, "total_students": len(categorized), "shown_students": len(categorized), "counts": {}}
    if not target_col or target_col not in categorized.columns or high_threshold <= low_threshold:
        return pd.DataFrame(), metadata

    if max_students and len(categorized) > max_students:
        categorized = categorized.sample(n=max_students, random_state=42).sort_index()
        metadata["sampled"] = True
        metadata["shown_students"] = len(categorized)

    categories = _category_order(display_mode)
    category_rank = {category: index for index, category in enumerate(categories)}
    categorized["grade_category_rank"] = categorized["grade_category"].map(category_rank).fillna(len(categories))
    counts = categorized["grade_category"].value_counts().reindex(categories, fill_value=0).to_dict()
    metadata["counts"] = {category: int(count) for category, count in counts.items()}

    panels = ["student_count"] + [feature for feature in features if feature in categorized.columns]
    rows = []
    for panel in panels:
        if panel == "student_count":
            ordered = categorized.sort_values(["grade_category_rank", target_col], ascending=[True, False]).copy()
            ordered["feature_value"] = np.nan
            ordered["feature_value_display"] = ""
            ordered["feature_level"] = "Student count"
            ordered["feature_level_rank"] = ordered["grade_category_rank"]
            panel_label = "Student Count"
            groups = [(key, group) for key, group in ordered.groupby("grade_category", sort=False)]
        else:
            ordered = categorized.copy()
            ordered["feature_value"] = _feature_numeric_values(ordered[panel], panel)
            ordered["feature_value_display"] = ordered[panel].map(lambda value: "" if pd.isna(value) else str(value))
            ordered["feature_level"], level_rank = _feature_levels(ordered[panel], panel)
            ordered["feature_level_rank"] = ordered["feature_level"].map(level_rank).fillna(len(level_rank))
            ordered = ordered.sort_values(
                ["grade_category_rank", "feature_level_rank", "feature_value", target_col],
                ascending=[True, True, True, False],
                na_position="last",
            )
            panel_label = pretty_name(panel)
            groups = [(key, group) for key, group in ordered.groupby(["grade_category", "feature_level"], sort=False)]

        for _, group in groups:
            if group.empty:
                continue
            local_x, local_y = _dot_positions(len(group), columns=8)
            for dot_index, (_, row) in enumerate(group.iterrows()):
                grade_rank = int(row["grade_category_rank"])
                if panel == "student_count":
                    base_x = grade_rank * 11
                    base_y = 0
                else:
                    feature_rank = int(row["feature_level_rank"])
                    base_x = feature_rank * 11
                    base_y = -(grade_rank * 9)
                rows.append(
                    {
                        "panel": panel,
                        "panel_label": panel_label,
                        "student_id": row.get(student_id_col, f"Student {dot_index + 1}") if student_id_col else f"Student {dot_index + 1}",
                        "x": float(base_x + local_x[dot_index]),
                        "y": float(base_y + local_y[dot_index]),
                        "feature": panel if panel != "student_count" else "Student Count",
                        "feature_label": panel_label,
                        "feature_value": row["feature_value"],
                        "feature_value_display": row["feature_value_display"],
                        "feature_level": row["feature_level"],
                        "feature_level_rank": int(row["feature_level_rank"]),
                        "score": row[target_col],
                        "grade_category": row["grade_category"],
                    }
                )
    return pd.DataFrame(rows), metadata


def _category_order(display_mode: str) -> list[str]:
    if display_mode == "high_low":
        return ["Low grade", "High grade"]
    return ["Low grade", "Medium/Average grade", "High grade"]
