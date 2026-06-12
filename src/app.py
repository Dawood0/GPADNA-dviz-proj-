from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from preprocessing import DATA_FILE, detect_columns, load_data, pretty_name
from vizs_src.radar import create_visual as create_radar
from vizs_src.heatmap import create_visual as create_heatmap
from vizs_src.bar_chart import make_bar_chart, TITLE as CHART_TITLE, EXPLANATION as CHART_EXPLANATION
from vizs_src.beeswarm import create_visual as create_beeswarm

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

HEATMAP_FIGURE_HEALTH, HEATMAP_TITLE, HEATMAP_EXPLANATION = create_heatmap(DF, target_col=TARGET_COL, mode="health")
HEATMAP_FIGURE_PAIRWISE, _, _ = create_heatmap(DF, target_col=TARGET_COL, mode="pairwise")
BAR_CHART_FIGURE = make_bar_chart(DF)


BEESWARM_FIGURE, BEESWARM_TITLE, BEESWARM_EXPLANATION = create_beeswarm(
    DF,
    features=DEFAULT_FEATURES,
    target_col=TARGET_COL,
    high_threshold=DEFAULT_HIGH,
    low_threshold=DEFAULT_LOW,
    display_mode="all",
)

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
                    html.H1("GPA DNA 🧬📊"),
                    html.P(
                        ["""
                        Ever wondered what the "genetic code" of academic success might look like?
                        GPA DNA invites you to explore how everyday habits vary across different groups of students.
                        """,
                        html.Br(),
                        html.Br(),
                        f"""
                        Using data from {len(DF):,} students, this interactive experience helps you uncover patterns,
                        challenge assumptions, and discover which habits are commonly associated with different levels
                        of academic performance. Adjust the grade thresholds, select the factors that interest you most,
                        and investigate the relationships that matter to you.
                        """,
                        html.Br(),
                        html.Br(),
                        """
                        From study routines and sleep schedules to social media use and extracurricular activities,
                        start exploring and see what combinations of habits shape the diverse "DNA" of student performance!
                        """],
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
                    html.Div(
                        [
                            html.H2(CHART_TITLE),
                            html.P(CHART_EXPLANATION),
                        ],
                        className="explanation",
                    ),
                    dcc.Graph(
                        id="bar-chart",
                        figure=BAR_CHART_FIGURE,
                        config={"displayModeBar": False, "responsive": True},
                    ),
                ],
                className="chart-card",
            ),

            html.Section(
                [
                    html.Div(
                        [
                            html.H2(id="visual-title"),
                            html.P(id="visual-explanation"),
                        ],
                        className="explanation",
                    ),
                    html.P(id="status", className="status"),
                    dcc.Graph(
                        id="radar-chart",
                        config={"displayModeBar": False, "responsive": True},
                    ),
                ],
                className="chart-card",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2(id="beeswarm-title"),
                            html.P(id="beeswarm-explanation"),
                        ],
                        className="explanation",
                    ),
                    dcc.Graph(
                        id="beeswarm-chart",
                        figure=BEESWARM_FIGURE,
                        config={"displayModeBar": False, "responsive": True},
                    ),
                ],
                className="chart-card",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2(HEATMAP_TITLE),
                            html.P(HEATMAP_EXPLANATION),
                        ],
                        className="explanation",
                    ),
                    dcc.Tabs(
                        id="heatmap-tabs",
                        value="health",
                        children=[
                            dcc.Tab(
                                label="Health Factors",
                                value="health",
                                children=[
                                    html.Div(
                                        dcc.Graph(
                                            id="heatmap-health-graph",
                                            figure=HEATMAP_FIGURE_HEALTH,
                                            config={"displayModeBar": False, "scrollZoom": False, "responsive": True},
                                            style={"width": "100%", "height": "900px"},
                                        ),
                                        style={"padding": "0.5rem"},
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Other Habits",
                                value="pairwise",
                                children=[
                                    html.Div(
                                        dcc.Graph(
                                            id="heatmap-pairwise-graph",
                                            figure=HEATMAP_FIGURE_PAIRWISE,
                                            config={"displayModeBar": False, "scrollZoom": False, "responsive": True},
                                            style={"width": "100%", "height": "900px"},
                                        ),
                                        style={"padding": "0.5rem"},
                                    )
                                ],
                            ),
                        ],
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
    Output("beeswarm-chart", "figure"),
    Output("beeswarm-title", "children"),
    Output("beeswarm-explanation", "children"),
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
    
    beeswarm_figure, beeswarm_title, beeswarm_explanation = create_beeswarm(
        DF,
        features=features,
        target_col=TARGET_COL,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        display_mode=display_mode,
    )

    # Call another imported visual in its own callback using the same pattern.
    return figure, title, explanation, beeswarm_figure, beeswarm_title,beeswarm_explanation, status, 


if __name__ == "__main__":
    app.run(debug=False)
