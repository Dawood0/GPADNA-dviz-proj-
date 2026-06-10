
"""Beeswarm visualization for student habits and academic performance."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocessing import (
    ORDINAL_CATEGORY_MAPS,
    create_grade_categories,
    detect_columns,
    pretty_name,
)


TITLE = "Student Habit Distribution by Grade Category"
EXPLANATION = (
    "Shows how individual student habit values spread across the selected factors. "
    "Colors use the same high (green), average (orange), and low grade (red) groups."
)
TEMPLATE = "plotly_white"
COLORS = {
    "High grade": "#0f9f6e",
    "Medium/Average grade": "#d97706",
    "Low grade": "#dc2626",
}
MARKER_SIZE = 9
MARKER_PADDING = 2
SWARM_STEP = 0.55
PLOT_WIDTH = 720
PLOT_MARGIN = {"l": 56, "r": 16, "t": 18, "b": 72}


@dataclass(frozen=True)
class FeatureConfig:
    key: str
    label: str
    kind: str
    categories: list[str] | None = None


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


def _category_order(display_mode: str) -> list[str]:
    if display_mode == "high_low":
        return ["Low grade", "High grade"]
    return ["Low grade", "Medium/Average grade", "High grade"]


def _is_numeric_like(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().mean() >= 0.9


def _is_discrete_numeric(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty or not _is_numeric_like(series):
        return False
    return valid.nunique() <= 10 and valid.round().eq(valid).all()


def _ordered_categories(series: pd.Series, feature: str) -> list[str]:
    labels = series.astype("string").dropna().unique().tolist()
    if feature in ORDINAL_CATEGORY_MAPS:
        order = ORDINAL_CATEGORY_MAPS[feature]
        return sorted(labels, key=lambda value: order.get(str(value).lower(), len(order)))
    return sorted(labels)


def _build_feature_configs(df: pd.DataFrame, features: list[str]) -> list[FeatureConfig]:
    configs: list[FeatureConfig] = []
    for feature in features:
        if feature not in df.columns:
            continue
        series = df[feature]
        if _is_discrete_numeric(series):
            categories = [str(int(value)) for value in sorted(pd.to_numeric(series, errors="coerce").dropna().unique())]
            configs.append(FeatureConfig(feature, pretty_name(feature), "discrete_numeric", categories))
        elif _is_numeric_like(series):
            configs.append(FeatureConfig(feature, pretty_name(feature), "continuous"))
        else:
            configs.append(FeatureConfig(feature, pretty_name(feature), "categorical", _ordered_categories(series, feature)))
    return configs


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _place_points(prepared_rows: list[dict[str, object]], diameter: float) -> list[dict[str, object]]:
    placed_points: list[dict[str, object]] = []
    buckets: dict[int, list[dict[str, object]]] = {}

    for point in prepared_rows:
        center_x = float(point["center_x"])
        bucket = int(center_x / diameter)
        nearby_points: list[dict[str, object]] = []
        for bucket_index in (bucket - 1, bucket, bucket + 1):
            nearby_points.extend(buckets.get(bucket_index, []))

        y = 0.0
        placed = False

        while not placed:
            candidate_ys = (0.0,) if y == 0.0 else (y, -y)
            for candidate_y in candidate_ys:
                collision = False
                for existing in nearby_points:
                    dx = center_x - float(existing["x"])
                    dy = candidate_y - float(existing["y"])
                    if math.hypot(dx, dy) < diameter:
                        collision = True
                        break
                if not collision:
                    placed_point = {"x": center_x, "y": candidate_y, "point": point}
                    placed_points.append(placed_point)
                    buckets.setdefault(bucket, []).append(placed_point)
                    placed = True
                    break
            if not placed:
                y += SWARM_STEP

    return placed_points


def _create_beeswarm_points(
    rows: pd.DataFrame,
    feature_config: FeatureConfig,
    target_col: str,
    student_id_col: str | None,
    display_mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    diameter = MARKER_SIZE + MARKER_PADDING
    inner_width = PLOT_WIDTH - PLOT_MARGIN["l"] - PLOT_MARGIN["r"]
    numeric_factor = feature_config.kind in {"continuous", "discrete_numeric"}
    categories = feature_config.categories if not numeric_factor else None
    category_section_width = inner_width / max(len(categories or []), 1) if categories else None
    grade_rank = {label: index for index, label in enumerate(_category_order(display_mode))}

    prepared_rows: list[dict[str, object]] = []
    x_min = None
    x_max = None

    if numeric_factor:
        numeric_values = pd.to_numeric(rows[feature_config.key], errors="coerce")
        valid_mask = numeric_values.notna()
        if not valid_mask.any():
            return pd.DataFrame(), {"numeric_factor": True, "categories": None, "x_min": 0.0, "x_max": 1.0}

        filtered_rows = rows.loc[valid_mask]
        numeric_values = numeric_values.loc[valid_mask]

        x_min = float(numeric_values.min())
        x_max = float(numeric_values.max())
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0

        scale = inner_width / (x_max - x_min)

        sortable = pd.DataFrame(
            {
                "grade_rank": filtered_rows["grade_category"].map(grade_rank).fillna(len(grade_rank)),
                "student_id": filtered_rows[student_id_col].astype(str) if student_id_col and student_id_col in filtered_rows.columns else "",
                "raw_x": numeric_values.astype(float),
                "category_label": filtered_rows[feature_config.key].map(_format_value),
                "grade_category": filtered_rows["grade_category"],
                "score": filtered_rows[target_col],
            },
            index=filtered_rows.index,
        )
        sortable["center_x"] = (sortable["raw_x"] - x_min) * scale
        sortable = sortable.sort_values(["center_x", "grade_rank", "student_id"])

        for row in sortable.itertuples():
            prepared_rows.append(
                {
                    "center_x": float(row.center_x),
                    "plot_x": float(row.raw_x),
                    "category_label": row.category_label,
                    "grade_category": row.grade_category,
                    "student_id": row.student_id if row.student_id != "" else "Unavailable",
                    "score": row.score,
                }
            )
    else:
        assert categories is not None
        label_series = rows[feature_config.key].astype("string")

        for category_index, category in enumerate(categories):
            mask = label_series.eq(category)
            if not mask.any():
                continue

            category_rows = rows.loc[mask]
            sorted_rows = category_rows.assign(
                _grade_rank=category_rows["grade_category"].map(grade_rank).fillna(len(grade_rank)),
                _student_id=category_rows[student_id_col].astype(str) if student_id_col and student_id_col in category_rows.columns else "",
            ).sort_values(["_grade_rank", target_col, "_student_id"], ascending=[True, False, True], na_position="last")

            section_center = category_section_width * (category_index + 0.5)
            max_spread = category_section_width * 0.72
            target_spread = max(diameter, min(max_spread, (len(sorted_rows) - 1) * diameter * 0.55))
            section_start = section_center - target_spread / 2
            section_step = 0 if len(sorted_rows) <= 1 else target_spread / (len(sorted_rows) - 1)

            for row_index, row in enumerate(sorted_rows.itertuples()):
                center_x = section_center if len(sorted_rows) == 1 else section_start + row_index * section_step
                section_offset = 0 if category_section_width == 0 else (center_x - section_center) / category_section_width
                prepared_rows.append(
                    {
                        "center_x": float(center_x),
                        "plot_x": category_index + section_offset,
                        "category_label": category,
                        "grade_category": row.grade_category,
                        "student_id": getattr(row, student_id_col) if student_id_col and hasattr(row, student_id_col) else "Unavailable",
                        "score": getattr(row, target_col),
                    }
                )

        prepared_rows.sort(
            key=lambda item: (
                float(item["center_x"]),
                grade_rank.get(str(item["grade_category"]), len(grade_rank)),
                str(item["student_id"]),
            )
        )

    placed_points = _place_points(prepared_rows, diameter)

    if not placed_points:
        return pd.DataFrame(), {"numeric_factor": numeric_factor, "categories": categories, "x_min": x_min, "x_max": x_max}

    points = pd.DataFrame(
        {
            "x": [item["point"]["plot_x"] for item in placed_points],
            "y": [item["y"] for item in placed_points],
            "grade_category": [item["point"]["grade_category"] for item in placed_points],
            "feature_value_display": [item["point"]["category_label"] for item in placed_points],
            "student_id": [item["point"]["student_id"] for item in placed_points],
            "score": [item["point"]["score"] for item in placed_points],
        }
    )
    metadata = {"numeric_factor": numeric_factor, "categories": categories, "x_min": x_min, "x_max": x_max}
    return points, metadata


def _render_beeswarm_figure(
    categorized: pd.DataFrame,
    feature_configs: list[FeatureConfig],
    target_col: str,
    student_id_col: str | None,
    display_mode: str,
) -> go.Figure:
    if categorized.empty or not feature_configs:
        return _empty_figure("Beeswarm chart needs a detected grade column and at least one selected feature.")

    cols = 2 if len(feature_configs) > 1 else 1
    rows = math.ceil(len(feature_configs) / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[config.label for config in feature_configs],
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )
    category_order = _category_order(display_mode)
    legend_seen: set[str] = set()

    for index, feature_config in enumerate(feature_configs):
        row_number = index // cols + 1
        col_number = index % cols + 1
        feature_rows = categorized.loc[categorized[feature_config.key].notna()]
        points, metadata = _create_beeswarm_points(feature_rows, feature_config, target_col, student_id_col, display_mode)
        if points.empty:
            continue

        grouped_points = {category: group for category, group in points.groupby("grade_category", sort=False)}
        y_limit = max(points["y"].abs().max(), 1.0) + 1.5

        for category in category_order:
            category_points = grouped_points.get(category)
            if category_points is None or category_points.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=category_points["x"],
                    y=category_points["y"],
                    mode="markers",
                    name=category,
                    legendgroup=category,
                    showlegend=category not in legend_seen,
                    marker={
                        "size": MARKER_SIZE,
                        "color": COLORS[category],
                        "line": {"color": "#ffffff", "width": 0.8},
                        "opacity": 0.84,
                    },
                    customdata=category_points[["feature_value_display", "student_id", "score"]].to_numpy(),
                    hovertemplate=(
                        f"<b>{feature_config.label}</b><br>"
                        "Value: %{customdata[0]}<br>"
                        "Student ID: %{customdata[1]}<br>"
                        "Score: %{customdata[2]:.2f}<br>"
                        "Group: %{fullData.name}<extra></extra>"
                    ),
                ),
                row=row_number,
                col=col_number,
            )
            legend_seen.add(category)

        if metadata["numeric_factor"]:
            fig.update_xaxes(title_text=feature_config.label, row=row_number, col=col_number)
        else:
            category_labels = metadata["categories"] or []
            fig.update_xaxes(
                title_text=feature_config.label,
                tickmode="array",
                tickvals=list(range(len(category_labels))),
                ticktext=category_labels,
                tickangle=35,
                row=row_number,
                col=col_number,
            )

        fig.update_yaxes(
            visible=False,
            range=[-y_limit, y_limit],
            zeroline=True,
            zerolinecolor="#cbd5e1",
            zerolinewidth=1.2,
            row=row_number,
            col=col_number,
        )

    fig.update_layout(
        template=TEMPLATE,
        height=max(420, rows * 340),
        margin={"l": 40, "r": 40, "t": 70, "b": 60},
        legend={"orientation": "h", "y": -0.08, "x": 0},
        font={"family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"},
    )
    return fig


def create_visual(df: pd.DataFrame, **kwargs) -> tuple[go.Figure, str, str]:
    """Return the beeswarm figure, title, and explanation for the supplied data."""
    detected = detect_columns(df)
    features = kwargs.get("features", detected["features"])
    target_col = kwargs.get("target_col", detected["target_col"])
    student_id_col = kwargs.get("student_id_col", detected["student_id_col"])
    high_threshold = float(kwargs.get("high_threshold", 80))
    low_threshold = float(kwargs.get("low_threshold", 50))
    display_mode = kwargs.get("display_mode", "all")

    if not target_col or target_col not in df.columns or high_threshold <= low_threshold:
        figure = _empty_figure("Beeswarm chart requires a detected grade column and valid thresholds.")
        return figure, TITLE, EXPLANATION

    categorized = create_grade_categories(df, target_col, high_threshold, low_threshold, display_mode)
    valid_features = [feature for feature in features if feature in categorized.columns]
    feature_configs = _build_feature_configs(categorized, valid_features)
    figure = _render_beeswarm_figure(categorized, feature_configs, target_col, student_id_col, display_mode)
    return figure, TITLE, EXPLANATION











# """Beeswarm visualization for student habits and academic performance."""

# from __future__ import annotations

# from dataclasses import dataclass
# import math

# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# from preprocessing import (
#     ORDINAL_CATEGORY_MAPS,
#     create_grade_categories,
#     detect_columns,
#     pretty_name,
# )


# TITLE = "Student Habit Distribution by Grade Category"
# EXPLANATION = (
#     "Shows how individual student habit values spread across the selected factors. "
#     "Colors use the same high (green), average (orange), and low grade (red) groups."
# )
# TEMPLATE = "plotly_white"
# COLORS = {
#     "High grade": "#0f9f6e",
#     "Medium/Average grade": "#d97706",
#     "Low grade": "#dc2626",
# }
# MARKER_SIZE = 9
# MARKER_PADDING = 2
# SWARM_STEP = 0.55
# PLOT_WIDTH = 720
# PLOT_MARGIN = {"l": 56, "r": 16, "t": 18, "b": 72}


# @dataclass(frozen=True)
# class FeatureConfig:
#     key: str
#     label: str
#     kind: str
#     categories: list[str] | None = None


# def _empty_figure(message: str) -> go.Figure:
#     fig = go.Figure()
#     fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16, "color": "#64748b"})
#     fig.update_layout(
#         template=TEMPLATE,
#         height=520,
#         margin={"l": 30, "r": 30, "t": 50, "b": 30},
#         xaxis={"visible": False},
#         yaxis={"visible": False},
#     )
#     return fig


# def _category_order(display_mode: str) -> list[str]:
#     if display_mode == "high_low":
#         return ["Low grade", "High grade"]
#     return ["Low grade", "Medium/Average grade", "High grade"]


# def _target_column(df: pd.DataFrame, target_col: str | None) -> str | None:
#     if target_col and target_col in df.columns:
#         return target_col
#     detected = detect_columns(df)
#     return detected["target_col"]


# def _student_id_column(df: pd.DataFrame, student_id_col: str | None) -> str | None:
#     if student_id_col and student_id_col in df.columns:
#         return student_id_col
#     detected = detect_columns(df)
#     return detected["student_id_col"]


# def _is_numeric_like(series: pd.Series) -> bool:
#     return pd.to_numeric(series, errors="coerce").notna().mean() >= 0.9


# def _is_discrete_numeric(series: pd.Series) -> bool:
#     numeric = pd.to_numeric(series, errors="coerce")
#     valid = numeric.dropna()
#     if valid.empty or not _is_numeric_like(series):
#         return False
#     return valid.nunique() <= 10 and (valid.round().eq(valid)).all()


# def _ordered_categories(series: pd.Series, feature: str) -> list[str]:
#     labels = series.astype("string").dropna().unique().tolist()
#     if feature in ORDINAL_CATEGORY_MAPS:
#         order = ORDINAL_CATEGORY_MAPS[feature]
#         return sorted(labels, key=lambda value: order.get(str(value).lower(), len(order)))
#     return sorted(labels)


# def _build_feature_configs(df: pd.DataFrame, features: list[str]) -> list[FeatureConfig]:
#     configs: list[FeatureConfig] = []
#     for feature in features:
#         if feature not in df.columns:
#             continue
#         series = df[feature]
#         if _is_discrete_numeric(series):
#             categories = [str(int(value)) for value in sorted(pd.to_numeric(series, errors="coerce").dropna().unique())]
#             configs.append(FeatureConfig(feature, pretty_name(feature), "discrete_numeric", categories))
#         elif _is_numeric_like(series):
#             configs.append(FeatureConfig(feature, pretty_name(feature), "continuous"))
#         else:
#             configs.append(FeatureConfig(feature, pretty_name(feature), "categorical", _ordered_categories(series, feature)))
#     return configs


# def _format_value(value: object) -> str:
#     if pd.isna(value):
#         return "Unavailable"
#     if isinstance(value, float):
#         return f"{value:.2f}"
#     return str(value)


# def _has_collision(candidate_x: float, candidate_y: float, placed_points: list[dict[str, object]], diameter: float) -> bool:
#     for point in placed_points:
#         dx = candidate_x - float(point["x"])
#         dy = candidate_y - float(point["y"])
#         if math.hypot(dx, dy) < diameter:
#             return True
#     return False


# def _create_beeswarm_points(
#     rows: pd.DataFrame,
#     feature_config: FeatureConfig,
#     target_col: str,
#     student_id_col: str | None,
# ) -> tuple[pd.DataFrame, dict[str, object]]:
#     diameter = MARKER_SIZE + MARKER_PADDING
#     inner_width = PLOT_WIDTH - PLOT_MARGIN["l"] - PLOT_MARGIN["r"]
#     numeric_factor = feature_config.kind in {"continuous", "discrete_numeric"}
#     categories = feature_config.categories if not numeric_factor else None
#     category_section_width = inner_width / max(len(categories or []), 1) if categories else None
#     grade_rank = {label: index for index, label in enumerate(_category_order("all"))}

#     prepared_rows: list[dict[str, object]] = []
#     x_min = None
#     x_max = None

#     if numeric_factor:
#         numeric_values = pd.to_numeric(rows[feature_config.key], errors="coerce").dropna()
#         if numeric_values.empty:
#             return pd.DataFrame(), {"numeric_factor": True, "categories": None, "x_min": 0.0, "x_max": 1.0}
#         x_min = float(numeric_values.min())
#         x_max = float(numeric_values.max())
#         if x_min == x_max:
#             x_min -= 1.0
#             x_max += 1.0

#         def x_value_to_pixel(value: float) -> float:
#             return ((value - x_min) / (x_max - x_min)) * inner_width

#         for _, row in rows.iterrows():
#             raw_x = pd.to_numeric(pd.Series([row[feature_config.key]]), errors="coerce").iloc[0]
#             if pd.isna(raw_x):
#                 continue
#             prepared_rows.append(
#                 {
#                     "row": row,
#                     "center_x": x_value_to_pixel(float(raw_x)),
#                     "plot_x": float(raw_x),
#                     "category_label": _format_value(row[feature_config.key]),
#                 }
#             )

#         prepared_rows.sort(
#             key=lambda item: (
#                 float(item["center_x"]),
#                 grade_rank.get(str(item["row"]["grade_category"]), len(grade_rank)),
#                 str(item["row"].get(student_id_col, "")) if student_id_col else "",
#             )
#         )
#     else:
#         assert categories is not None
#         for category_index, category in enumerate(categories):
#             category_rows = rows.loc[rows[feature_config.key].astype("string").eq(category)].copy()
#             if category_rows.empty:
#                 continue
#             category_rows = category_rows.sort_values(
#                 ["grade_category", target_col],
#                 key=lambda column: column.map(grade_rank) if column.name == "grade_category" else column,
#                 ascending=[True, False],
#                 na_position="last",
#             )
#             section_center = category_section_width * (category_index + 0.5)
#             max_spread = category_section_width * 0.72
#             target_spread = max(diameter, min(max_spread, (len(category_rows) - 1) * diameter * 0.55))
#             section_start = section_center - target_spread / 2
#             section_step = 0 if len(category_rows) <= 1 else target_spread / (len(category_rows) - 1)

#             for row_index, (_, row) in enumerate(category_rows.iterrows()):
#                 center_x = section_center if len(category_rows) == 1 else section_start + row_index * section_step
#                 section_offset = 0 if category_section_width == 0 else (center_x - section_center) / category_section_width
#                 prepared_rows.append(
#                     {
#                         "row": row,
#                         "center_x": center_x,
#                         "plot_x": category_index + section_offset,
#                         "category_label": category,
#                     }
#                 )

#         prepared_rows.sort(
#             key=lambda item: (
#                 float(item["center_x"]),
#                 grade_rank.get(str(item["row"]["grade_category"]), len(grade_rank)),
#                 str(item["row"].get(student_id_col, "")) if student_id_col else "",
#             )
#         )

#     placed_points: list[dict[str, object]] = []
#     for point in prepared_rows:
#         nearby_points = [existing for existing in placed_points if abs(float(point["center_x"]) - float(existing["x"])) < diameter]
#         if not _has_collision(float(point["center_x"]), 0.0, nearby_points, diameter):
#             placed_points.append({"x": float(point["center_x"]), "y": 0.0, "point": point})
#             continue

#         level = SWARM_STEP
#         placed = False
#         while not placed:
#             for candidate_y in (level, -level):
#                 if not _has_collision(float(point["center_x"]), candidate_y, nearby_points, diameter):
#                     placed_points.append({"x": float(point["center_x"]), "y": candidate_y, "point": point})
#                     placed = True
#                     break
#             level += SWARM_STEP

#     if not placed_points:
#         return pd.DataFrame(), {"numeric_factor": numeric_factor, "categories": categories, "x_min": x_min, "x_max": x_max}

#     points = pd.DataFrame(
#         [
#             {
#                 "x": item["point"]["plot_x"],
#                 "y": item["y"],
#                 "grade_category": item["point"]["row"]["grade_category"],
#                 "feature_value_display": item["point"]["category_label"],
#                 "student_id": item["point"]["row"].get(student_id_col, "Unavailable") if student_id_col else "Unavailable",
#                 "score": item["point"]["row"][target_col],
#             }
#             for item in placed_points
#         ]
#     )
#     metadata = {"numeric_factor": numeric_factor, "categories": categories, "x_min": x_min, "x_max": x_max}
#     return points, metadata


# def _render_beeswarm_figure(
#     categorized: pd.DataFrame,
#     feature_configs: list[FeatureConfig],
#     target_col: str,
#     student_id_col: str | None,
#     display_mode: str,
# ) -> go.Figure:
#     if categorized.empty or not feature_configs:
#         return _empty_figure("Beeswarm chart needs a detected grade column and at least one selected feature.")

#     cols = 2 if len(feature_configs) > 1 else 1
#     rows = math.ceil(len(feature_configs) / cols)
#     fig = make_subplots(
#         rows=rows,
#         cols=cols,
#         subplot_titles=[config.label for config in feature_configs],
#         horizontal_spacing=0.1,
#         vertical_spacing=0.16,
#     )
#     category_order = _category_order(display_mode)
#     legend_seen: set[str] = set()

#     for index, feature_config in enumerate(feature_configs):
#         row_number = index // cols + 1
#         col_number = index % cols + 1
#         feature_rows = categorized.loc[categorized[feature_config.key].notna()].copy()
#         points, metadata = _create_beeswarm_points(feature_rows, feature_config, target_col, student_id_col)
#         if points.empty:
#             continue

#         y_limit = max(points["y"].abs().max(), 1.0) + 1.5
#         for category in category_order:
#             category_points = points.loc[points["grade_category"].eq(category)]
#             if category_points.empty:
#                 continue
#             fig.add_trace(
#                 go.Scatter(
#                     x=category_points["x"],
#                     y=category_points["y"],
#                     mode="markers",
#                     name=category,
#                     legendgroup=category,
#                     showlegend=category not in legend_seen,
#                     marker={
#                         "size": MARKER_SIZE,
#                         "color": COLORS[category],
#                         "line": {"color": "#ffffff", "width": 0.8},
#                         "opacity": 0.84,
#                     },
#                     customdata=category_points[["feature_value_display", "student_id", "score"]],
#                     hovertemplate=(
#                         f"<b>{feature_config.label}</b><br>"
#                         "Value: %{customdata[0]}<br>"
#                         "Student ID: %{customdata[1]}<br>"
#                         "Score: %{customdata[2]:.2f}<br>"
#                         "Group: %{fullData.name}<extra></extra>"
#                     ),
#                 ),
#                 row=row_number,
#                 col=col_number,
#             )
#             legend_seen.add(category)

#         axis_suffix = "" if index == 0 else str(index + 1)
#         if metadata["numeric_factor"]:
#             fig.update_xaxes(title_text=feature_config.label, row=row_number, col=col_number)
#         else:
#             categories = metadata["categories"] or []
#             fig.update_xaxes(
#                 title_text=feature_config.label,
#                 tickmode="array",
#                 tickvals=list(range(len(categories))),
#                 ticktext=categories,
#                 tickangle=35,
#                 row=row_number,
#                 col=col_number,
#             )

#         fig.update_yaxes(
#             visible=False,
#             range=[-y_limit, y_limit],
#             zeroline=True,
#             zerolinecolor="#cbd5e1",
#             zerolinewidth=1.2,
#             row=row_number,
#             col=col_number,
#         )

#     fig.update_layout(
#         template=TEMPLATE,
#         height=max(420, rows * 340),
#         margin={"l": 40, "r": 40, "t": 70, "b": 60},
#         legend={"orientation": "h", "y": -0.08, "x": 0},
#         font={"family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"},
#     )
#     return fig


# def create_visual(df: pd.DataFrame, **kwargs) -> tuple[go.Figure, str, str]:
#     """Return the beeswarm figure, title, and explanation for the supplied data."""
#     detected = detect_columns(df)
#     features = kwargs.get("features", detected["features"])
#     target_col = _target_column(df, kwargs.get("target_col", detected["target_col"]))
#     student_id_col = _student_id_column(df, kwargs.get("student_id_col", detected["student_id_col"]))
#     high_threshold = float(kwargs.get("high_threshold", 80))
#     low_threshold = float(kwargs.get("low_threshold", 50))
#     display_mode = kwargs.get("display_mode", "all")

#     if not target_col or high_threshold <= low_threshold:
#         figure = _empty_figure("Beeswarm chart requires a detected grade column and valid thresholds.")
#         return figure, TITLE, EXPLANATION

#     categorized = create_grade_categories(df, target_col, high_threshold, low_threshold, display_mode)
#     valid_features = [feature for feature in features if feature in categorized.columns]
#     feature_configs = _build_feature_configs(categorized, valid_features)
#     figure = _render_beeswarm_figure(categorized, feature_configs, target_col, student_id_col, display_mode)
#     return figure, TITLE, EXPLANATION
