from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from preprocessing import DATA_FILE, detect_columns, load_data, pretty_name
from vizs_src.radar import create_visual as create_radar

# Add teammate visuals with direct imports, for example:
# from vizs_src.scatter_plot import create_visual as create_scatter_plot


DF = load_data(DATA_FILE)
DETECTED = detect_columns(DF)
TARGET_COL = DETECTED["target_col"]
FEATURES = DETECTED["features"]
DEFAULT_FEATURES = FEATURES[:7]
FEATURE_OPTIONS = [{"label": pretty_name(feature), "value": feature} for feature in FEATURES]
DEFAULT_LOW = 50
DEFAULT_HIGH = 80

app = Dash(__name__)
app.title = "GPADNA"


def make_control(label: str, component) -> html.Div:
    return html.Div([html.Label(label), component], className="control")


def create_layout() -> html.Main:
    return html.Main(
        [
            html.Header(
                [
                    html.P("DNA OF STUDENT PERFORMANCE", className="eyebrow"),
                    html.H1("GPADNA 🧬📊"),
                    html.P(
                        f"Explore how habits differ across grade groups for {len(DF):,} students.",
                        className="subtitle",
                    ),
                ],
                className="hero",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Filters"),
                            html.P("Choose the habits and grade groups to compare."),
                        ],
                        className="section-heading",
                    ),
                    html.Div(
                        [
                            make_control(
                                "Features",
                                dcc.Dropdown(
                                    id="features",
                                    options=FEATURE_OPTIONS,
                                    value=DEFAULT_FEATURES,
                                    multi=True,
                                ),
                            ),
                            make_control(
                                "Low grade threshold",
                                dcc.Input(id="low-threshold", type="number", value=DEFAULT_LOW, min=0, max=100),
                            ),
                            make_control(
                                "High grade threshold",
                                dcc.Input(id="high-threshold", type="number", value=DEFAULT_HIGH, min=0, max=100),
                            ),
                            make_control(
                                "Groups",
                                dcc.RadioItems(
                                    id="display-mode",
                                    options=[
                                        {"label": "All", "value": "all"},
                                        {"label": "High and low", "value": "high_low"},
                                    ],
                                    value="all",
                                    inline=True,
                                ),
                            ),
                        ],
                        className="control-grid",
                    ),
                ],
                className="filter-card",
            ),
            html.Section(
                [
                    html.P(id="status", className="status"),
                    dcc.Graph(
                        id="radar-chart",
                        config={"displayModeBar": False, "responsive": True},
                    ),
                    html.Div(
                        [
                            html.H2(id="visual-title"),
                            html.P(id="visual-explanation"),
                        ],
                        className="explanation",
                    ),
                ],
                className="chart-card",
            ),
        ],
        className="page",
    )


app.layout = create_layout()


@app.callback(
    Output("radar-chart", "figure"),
    Output("visual-title", "children"),
    Output("visual-explanation", "children"),
    Output("status", "children"),
    Input("features", "value"),
    Input("low-threshold", "value"),
    Input("high-threshold", "value"),
    Input("display-mode", "value"),
)
def update_radar(features, low_threshold, high_threshold, display_mode):
    features = features or []
    low_threshold = float(low_threshold if low_threshold is not None else DEFAULT_LOW)
    high_threshold = float(high_threshold if high_threshold is not None else DEFAULT_HIGH)
    status = ""
    if high_threshold <= low_threshold:
        status = "High threshold must be greater than the low threshold."
    elif not features:
        status = "Select at least one feature."

    figure, title, explanation = create_radar(
        DF,
        features=features,
        target_col=TARGET_COL,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        display_mode=display_mode,
    )
    # Call another imported visual in its own callback using the same pattern.
    return figure, title, explanation, status


if __name__ == "__main__":
    app.run(debug=False)
