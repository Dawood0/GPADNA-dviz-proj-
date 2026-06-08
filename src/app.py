from __future__ import annotations

import json
import socket

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from preprocessing import (
    DATA_FILE,
    classify_features_by_grade_pattern,
    detect_columns,
    load_data,
    prepare_radar_data,
    pretty_name,
    select_features,
)
from all_in_one import create_visual


DF = load_data(DATA_FILE)
DETECTED = detect_columns(DF)
TARGET_COL = DETECTED["target_col"]
STUDENT_ID_COL = DETECTED["student_id_col"]
ALL_FEATURES = DETECTED["features"]
DEFAULT_HIGH, DEFAULT_LOW = 80, 50
HIGH_THRESHOLD_MIN, HIGH_THRESHOLD_MAX = 60, 100
LOW_THRESHOLD_MIN, LOW_THRESHOLD_MAX = 20, 60
MAX_FEATURES = max(1, len(ALL_FEATURES))
DEFAULT_RADAR_FEATURE_COUNT = min(7, MAX_FEATURES)
DEFAULT_FEATURE_GROUPS = classify_features_by_grade_pattern(DF, ALL_FEATURES, TARGET_COL, DEFAULT_HIGH, DEFAULT_LOW)

print("Detected columns:")
print(f"- Grade/exam score: {TARGET_COL or 'not found'}")
print(f"- Student ID: {STUDENT_ID_COL or 'not found'}")
print("- Features:", ", ".join(ALL_FEATURES) or "none")
print("- Numeric features:", ", ".join(DETECTED["numeric_features"]) or "none")
print("- Categorical features:", ", ".join(DETECTED["categorical_features"]) or "none")
print("- Positive features:", ", ".join(DEFAULT_FEATURE_GROUPS["positive"]) or "none")
print("- Negative features:", ", ".join(DEFAULT_FEATURE_GROUPS["negative"]) or "none")


def explanation_cards(chart: str) -> html.Div:
    cards = [
        ("What it shows", "Average normalized habit values for high, average, and low grade students."),
        ("Thresholds", "Changing the high and low thresholds rebuilds the student groups used in the averages."),
        ("Feature filter", "Positive means high feature values contain more high-grade than low-grade students; Negative means the reverse."),
        ("What to look for", "In All features, the strongest high-vs-low differences are placed first so related separations are easier to scan."),
    ]
    return html.Div([html.Div([html.Strong(title), html.P(text)]) for title, text in cards], className="explanation-grid")


def filtered_features(feature_filter: str, high_threshold: float, low_threshold: float) -> list[str]:
    if feature_filter == "all":
        return rank_features_by_grade_difference(list(ALL_FEATURES), high_threshold, low_threshold)
    groups = classify_features_by_grade_pattern(DF, ALL_FEATURES, TARGET_COL, high_threshold, low_threshold)
    return select_features(groups, feature_filter)


def parse_manual_features(raw_features: list[str] | str | None) -> list[str]:
    if not raw_features:
        return []
    if isinstance(raw_features, str):
        try:
            features = json.loads(raw_features)
        except (TypeError, ValueError):
            return []
    else:
        features = raw_features
    if not isinstance(features, list):
        return []
    return [feature for feature in features if feature in ALL_FEATURES]


def selected_features_for_mode(
    selection_mode: str,
    feature_filter: str,
    high_threshold: float,
    low_threshold: float,
    manual_features: list[str] | str | None,
) -> list[str]:
    if selection_mode == "manual":
        return parse_manual_features(manual_features)
    return filtered_features(feature_filter, high_threshold, low_threshold)


def feature_pill(feature: str, selected: bool) -> html.Div:
    return html.Div(
        [
            html.Span(pretty_name(feature), className="feature-pill-label"),
            html.Button(
                "x",
                id={"type": "feature-remove", "feature": feature},
                type="button",
                className="feature-remove",
                **{"data-feature": feature},
            )
            if selected
            else html.Button(
                "+",
                id={"type": "feature-add", "feature": feature},
                type="button",
                className="feature-add",
                **{"data-feature": feature},
            ),
        ],
        className="feature-pill",
        draggable="true",
        **{"data-feature": feature},
    )


def feature_selector_panel(raw_features: list[str] | str | None) -> html.Div:
    selected = parse_manual_features(raw_features)
    available = [feature for feature in ALL_FEATURES if feature not in selected]
    selected_children = [feature_pill(feature, True) for feature in selected] or [
        html.Div("Drop features here", className="feature-empty")
    ]
    available_children = [feature_pill(feature, False) for feature in available] or [
        html.Div("All features selected", className="feature-empty")
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Strong("Selected features"),
                    html.Div(selected_children, id="manual-selected-list", className="feature-dropzone selected-features"),
                ],
                className="feature-list-block",
            ),
            html.Div(
                [
                    html.Strong("Available features"),
                    html.Div(available_children, id="manual-available-list", className="feature-dropzone available-features"),
                ],
                className="feature-list-block",
            ),
        ],
        className="manual-feature-panel",
    )


def rank_features_by_grade_difference(features: list[str], high_threshold: float, low_threshold: float) -> list[str]:
    radar_data = prepare_radar_data(DF, features, TARGET_COL, high_threshold, low_threshold, "all")
    if radar_data.empty:
        return features
    pivot = radar_data.pivot_table(
        index="feature",
        columns="grade_category",
        values="normalized_average",
        aggfunc="mean",
    )
    if {"High grade", "Low grade"}.issubset(pivot.columns):
        scores = (pivot["High grade"] - pivot["Low grade"]).abs()
    else:
        scores = pivot.max(axis=1) - pivot.min(axis=1)
    order = scores.sort_values(ascending=False).index.tolist()
    return [feature for feature in order if feature in features] + [feature for feature in features if feature not in order]


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Student Habits Dashboard"

app.layout = html.Div(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.P("Student habits dashboard", className="eyebrow"),
                        html.H1("Student Habits and Academic Performance"),
                        html.P(
                            "This dashboard explores how student habits relate to academic performance by comparing high, average, and low grade students.",
                            className="intro",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Span("Dataset"),
                        html.Strong(f"{len(DF):,} students"),
                        html.P(f"Grade column: {pretty_name(TARGET_COL)}"),
                        html.P(f"Student ID: {pretty_name(STUDENT_ID_COL)}"),
                    ],
                    className="dataset-summary",
                ),
            ],
            className="hero",
        ),
        html.Section(
            [
                dcc.Store(
                    id="manual-features-value",
                    data=filtered_features("all", DEFAULT_HIGH, DEFAULT_LOW),
                ),
                html.Div(
                    [
                        html.Label("Feature filter", htmlFor="feature-filter"),
                        dcc.RadioItems(
                            id="feature-filter",
                            value="all",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Positive", "value": "positive"},
                                {"label": "Negative", "value": "negative"},
                            ],
                            className="button-group",
                            labelClassName="filter-button",
                            inputClassName="filter-button-input",
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("High grade threshold", htmlFor="high-threshold"),
                        dcc.Slider(
                            id="high-threshold",
                            min=HIGH_THRESHOLD_MIN,
                            max=HIGH_THRESHOLD_MAX,
                            step=1,
                            value=round(DEFAULT_HIGH),
                            marks={60: "60", 80: "80", 100: "100"},
                            tooltip={"always_visible": True, "placement": "bottom"},
                        ),
                    ],
                    className="control slider-control",
                ),
                html.Div(
                    [
                        html.Label("Low grade threshold", htmlFor="low-threshold"),
                        dcc.Slider(
                            id="low-threshold",
                            min=LOW_THRESHOLD_MIN,
                            max=LOW_THRESHOLD_MAX,
                            step=1,
                            value=round(DEFAULT_LOW),
                            marks={20: "20", 50: "50", 60: "60"},
                            tooltip={"always_visible": True, "placement": "bottom"},
                        ),
                    ],
                    className="control slider-control",
                ),
                html.Div(
                    [
                        html.Label("Category display", htmlFor="category-display"),
                        dcc.Dropdown(
                            id="category-display",
                            value="all",
                            clearable=False,
                            options=[
                                {"label": "Show High, Medium, Low", "value": "all"},
                                {"label": "Show High and Low only", "value": "high_low"},
                            ],
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Feature selection", htmlFor="feature-selection-mode"),
                        dcc.RadioItems(
                            id="feature-selection-mode",
                            value="auto",
                            options=[
                                {"label": "Auto", "value": "auto"},
                                {"label": "Manual list", "value": "manual"},
                            ],
                            className="button-group two-buttons",
                            labelClassName="filter-button",
                            inputClassName="filter-button-input",
                        ),
                    ],
                    className="control",
                ),
            ],
            className="controls",
        ),
        html.Div(
            [
                html.Section(
                    [
                        html.Label("Radar feature count", htmlFor="radar-feature-count"),
                        dcc.Slider(
                            id="radar-feature-count",
                            min=1,
                            max=MAX_FEATURES,
                            step=1,
                            value=DEFAULT_RADAR_FEATURE_COUNT,
                            marks={1: "1", MAX_FEATURES: str(MAX_FEATURES)},
                            tooltip={"always_visible": True, "placement": "bottom"},
                        ),
                    ],
                    className="radar-controls",
                ),
            ],
            className="analysis-layout",
        ),
        html.Div(id="status-message", className="status-message"),
        html.Section(
            [
                html.Div(
                    [
                        dcc.Graph(id="chart", config={"displayModeBar": True, "responsive": True}),
                        html.Aside(
                            [
                                html.Div(
                                    [
                                        html.Strong("Manual features"),
                                        html.Span("Drag, drop, add, or remove."),
                                    ],
                                    className="manual-feature-heading",
                                ),
                                html.Div(id="manual-feature-panel"),
                            ],
                            id="manual-feature-aside",
                            className="manual-feature-aside",
                        ),
                    ],
                    className="chart-layout",
                ),
                html.Div(id="chart-explanation"),
            ],
            className="chart-shell",
        ),
    ],
    className="app",
)


@app.callback(
    Output("manual-features-value", "data"),
    Input("feature-selection-mode", "value"),
    Input("feature-filter", "value"),
    Input("high-threshold", "value"),
    Input("low-threshold", "value"),
    Input({"type": "feature-add", "feature": ALL}, "n_clicks"),
    Input({"type": "feature-remove", "feature": ALL}, "n_clicks"),
    State("manual-features-value", "data"),
)
def sync_manual_features(
    selection_mode: str,
    feature_filter: str,
    high_threshold: float,
    low_threshold: float,
    add_clicks: list[int | None],
    remove_clicks: list[int | None],
    current_manual_features: list[str] | str | None,
):
    current_features = parse_manual_features(current_manual_features)
    triggered = ctx.triggered_id
    if isinstance(triggered, dict):
        feature = triggered.get("feature")
        if triggered.get("type") == "feature-add" and feature in ALL_FEATURES and feature not in current_features:
            return current_features + [feature]
        if triggered.get("type") == "feature-remove":
            return [item for item in current_features if item != feature]

    if selection_mode == "manual" and triggered != "feature-selection-mode":
        return no_update
    high_threshold = float(high_threshold or DEFAULT_HIGH)
    low_threshold = float(low_threshold or DEFAULT_LOW)
    if selection_mode == "manual" and current_features:
        return current_features
    return filtered_features(feature_filter, high_threshold, low_threshold)


@app.callback(
    Output("manual-feature-aside", "className"),
    Input("feature-selection-mode", "value"),
)
def toggle_manual_feature_panel(selection_mode: str):
    class_name = "manual-feature-aside"
    if selection_mode == "manual":
        return f"{class_name} is-open"
    return class_name


@app.callback(
    Output("manual-feature-panel", "children"),
    Input("manual-features-value", "data"),
)
def update_manual_feature_panel(manual_features: list[str] | str | None):
    return feature_selector_panel(manual_features)


@app.callback(
    Output("radar-feature-count", "max"),
    Output("radar-feature-count", "value"),
    Output("radar-feature-count", "marks"),
    Input("feature-selection-mode", "value"),
    Input("feature-filter", "value"),
    Input("high-threshold", "value"),
    Input("low-threshold", "value"),
    Input("manual-features-value", "data"),
    Input("radar-feature-count", "value"),
)
def update_radar_feature_count(
    selection_mode: str,
    feature_filter: str,
    high_threshold: float,
    low_threshold: float,
    manual_features: list[str] | str | None,
    current_value: int,
):
    high_threshold = float(high_threshold or DEFAULT_HIGH)
    low_threshold = float(low_threshold or DEFAULT_LOW)
    available = selected_features_for_mode(selection_mode, feature_filter, high_threshold, low_threshold, manual_features)
    maximum = max(1, len(available))
    if ctx.triggered_id in {"feature-filter", "high-threshold", "low-threshold", "feature-selection-mode", "manual-features-value"}:
        value = maximum
    else:
        value = min(max(1, int(current_value or maximum)), maximum)
    return maximum, value, {1: "1", maximum: str(maximum)}


@app.callback(
    Output("chart", "figure"),
    Output("chart-explanation", "children"),
    Output("status-message", "children"),
    Input("feature-selection-mode", "value"),
    Input("feature-filter", "value"),
    Input("high-threshold", "value"),
    Input("low-threshold", "value"),
    Input("manual-features-value", "data"),
    Input("category-display", "value"),
    Input("radar-feature-count", "value"),
)
def update_chart(
    selection_mode: str,
    feature_filter: str,
    high_threshold: float,
    low_threshold: float,
    manual_features: list[str] | str | None,
    display_mode: str,
    radar_feature_count: int,
):
    high_threshold = float(high_threshold or DEFAULT_HIGH)
    low_threshold = float(low_threshold or DEFAULT_LOW)
    selected_features = selected_features_for_mode(selection_mode, feature_filter, high_threshold, low_threshold, manual_features)
    status = ""
    if high_threshold <= low_threshold:
        status = "High threshold must be greater than the low threshold."
    elif not selected_features:
        status = "No features selected. Switch to Auto or add features from the manual panel."

    radar_features = selected_features[: max(1, int(radar_feature_count or len(selected_features)))]
    figure, _title, _explanation = create_visual(
        DF,
        features=radar_features,
        target_col=TARGET_COL,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        display_mode=display_mode,
    )
    return figure, explanation_cards("radar"), status


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --bg: #f6f8fb;
                --panel: #ffffff;
                --ink: #111827;
                --muted: #64748b;
                --line: #dbe3ee;
                --accent: #0f766e;
                --warn: #b45309;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                background: var(--bg);
                color: var(--ink);
                font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            .app { max-width: 1260px; margin: 0 auto; padding: 28px; }
            .hero {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 260px;
                gap: 24px;
                align-items: stretch;
                padding: 28px;
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
            }
            .eyebrow {
                margin: 0 0 8px;
                color: var(--accent);
                font-size: 13px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0;
            }
            h1 { margin: 0; font-size: 38px; line-height: 1.08; letter-spacing: 0; }
            p { margin: 8px 0 0; color: var(--muted); line-height: 1.5; }
            .intro { max-width: 760px; font-size: 16px; }
            .dataset-summary {
                border-left: 1px solid var(--line);
                padding-left: 22px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .dataset-summary span { color: var(--muted); font-size: 13px; }
            .dataset-summary strong { display: block; margin: 4px 0 8px; font-size: 25px; }
            .dataset-summary p { margin: 2px 0; font-size: 13px; }
            .controls {
                display: grid;
                grid-template-columns: 300px minmax(190px, 1fr) minmax(190px, 1fr) 220px 190px;
                gap: 18px;
                margin-top: 20px;
                padding: 20px;
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                align-items: start;
            }
            .radar-controls {
                padding: 18px 20px 22px;
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
            }
            .analysis-layout {
                margin-top: 14px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 800;
                font-size: 14px;
            }
            .button-group {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
                background: #f8fafc;
            }
            .button-group.two-buttons { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .filter-button {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 40px;
                margin: 0;
                padding: 0 10px;
                border-right: 1px solid var(--line);
                color: var(--muted);
                cursor: pointer;
                font-size: 13px;
                font-weight: 800;
                text-align: center;
                user-select: none;
            }
            .filter-button:last-child { border-right: 0; }
            .filter-button:has(.filter-button-input:checked) {
                background: var(--accent);
                color: #ffffff;
            }
            .filter-button-input {
                position: absolute;
                opacity: 0;
                pointer-events: none;
            }
            .slider-control { padding: 0 6px; }
            .manual-feature-aside {
                display: none;
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 11px;
            }
            .manual-feature-aside.is-open { display: block; }
            .manual-feature-heading {
                display: flex;
                justify-content: space-between;
                gap: 8px;
                align-items: baseline;
                margin-bottom: 8px;
            }
            .manual-feature-heading strong {
                font-size: 12px;
            }
            .manual-feature-heading span {
                color: var(--muted);
                font-size: 10px;
                font-weight: 700;
            }
            .manual-feature-panel {
                display: grid;
                gap: 10px;
            }
            .feature-list-block strong {
                display: block;
                margin-bottom: 6px;
                font-size: 11px;
            }
            .feature-dropzone {
                min-height: 92px;
                max-height: 210px;
                overflow: auto;
                padding: 6px;
                border: 1px dashed #b7c4d6;
                border-radius: 8px;
                background: #f8fafc;
            }
            .selected-features {
                border-color: #5eead4;
                background: #f0fdfa;
            }
            .feature-pill {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 22px;
                gap: 6px;
                align-items: center;
                min-height: 28px;
                margin-bottom: 5px;
                padding: 4px 4px 4px 8px;
                border: 1px solid var(--line);
                border-radius: 6px;
                background: #ffffff;
                cursor: grab;
            }
            .feature-pill:last-child { margin-bottom: 0; }
            .feature-pill:active { cursor: grabbing; }
            .feature-pill-label {
                min-width: 0;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                font-size: 11px;
                font-weight: 700;
            }
            .feature-add,
            .feature-remove {
                width: 22px;
                height: 22px;
                border: 1px solid var(--line);
                border-radius: 5px;
                background: #ffffff;
                color: var(--ink);
                cursor: pointer;
                font-weight: 900;
                font-size: 11px;
                line-height: 1;
            }
            .feature-add {
                border-color: #99f6e4;
                color: #0f766e;
            }
            .feature-remove {
                border-color: #fecaca;
                color: #b91c1c;
            }
            .feature-empty {
                padding: 10px;
                color: var(--muted);
                font-size: 11px;
                font-weight: 700;
                text-align: center;
            }
            .status-message {
                min-height: 22px;
                margin: 12px 2px 0;
                color: var(--warn);
                font-weight: 700;
            }
            .tabs {
                margin-top: 16px;
                border-bottom: 1px solid var(--line);
            }
            .tab {
                border: 0 !important;
                border-bottom: 3px solid transparent !important;
                background: transparent !important;
                color: var(--muted) !important;
                font-weight: 800;
                padding: 13px 18px !important;
            }
            .tab.selected {
                color: var(--ink) !important;
                border-bottom-color: var(--accent) !important;
            }
            .chart-shell {
                padding: 22px;
                background: var(--panel);
                border: 1px solid var(--line);
                border-top: 0;
                border-radius: 0 0 8px 8px;
            }
            .chart-layout {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 260px;
                gap: 14px;
                align-items: start;
            }
            .explanation-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                border-top: 1px solid var(--line);
                padding-top: 18px;
            }
            .explanation-grid div {
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 13px;
                background: #f8fafc;
            }
            .explanation-grid strong { display: block; margin-bottom: 6px; }
            .explanation-grid p { margin: 0; font-size: 14px; }
            @media (max-width: 980px) {
                .app { padding: 14px; }
                .hero { grid-template-columns: 1fr; }
                .dataset-summary { border-left: 0; border-top: 1px solid var(--line); padding: 18px 0 0; }
                .controls { grid-template-columns: 1fr; }
                .chart-layout { grid-template-columns: 1fr; }
                h1 { font-size: 30px; }
                .explanation-grid { grid-template-columns: 1fr; }
            }
        </style>
        <script>
            (function () {
                function writeFeatures(features) {
                    if (!window.dash_clientside || !window.dash_clientside.set_props) {
                        return;
                    }
                    const unique = features.filter((feature, index) => features.indexOf(feature) === index);
                    window.dash_clientside.set_props("manual-features-value", { data: unique });
                }

                function selectedFeaturesFromDom() {
                    return Array.from(document.querySelectorAll("#manual-selected-list .feature-pill"))
                        .map((pill) => pill.dataset.feature)
                        .filter(Boolean);
                }

                function moveFeature(feature, destination, beforeFeature) {
                    if (!feature) {
                        return;
                    }
                    let features = selectedFeaturesFromDom().filter((item) => item !== feature);
                    if (destination === "selected") {
                        const insertAt = beforeFeature ? features.indexOf(beforeFeature) : -1;
                        if (insertAt >= 0) {
                            features.splice(insertAt, 0, feature);
                        } else {
                            features.push(feature);
                        }
                    }
                    writeFeatures(features);
                }

                document.addEventListener("dragstart", function (event) {
                    const pill = event.target.closest(".feature-pill");
                    if (!pill) {
                        return;
                    }
                    event.dataTransfer.setData("text/plain", pill.dataset.feature);
                    event.dataTransfer.effectAllowed = "move";
                });

                document.addEventListener("dragover", function (event) {
                    if (event.target.closest(".feature-dropzone")) {
                        event.preventDefault();
                    }
                });

                document.addEventListener("drop", function (event) {
                    const zone = event.target.closest(".feature-dropzone");
                    if (!zone) {
                        return;
                    }
                    event.preventDefault();
                    const feature = event.dataTransfer.getData("text/plain");
                    const targetPill = event.target.closest(".feature-pill");
                    const beforeFeature = targetPill ? targetPill.dataset.feature : null;
                    const destination = zone.id === "manual-selected-list" ? "selected" : "available";
                    moveFeature(feature, destination, beforeFeature);
                });
            })();
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def find_open_port(start: int = 8050, attempts: int = 20) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return start


if __name__ == "__main__":
    app.run(debug=False, port=find_open_port())
