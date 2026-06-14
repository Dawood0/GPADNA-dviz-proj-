from pathlib import Path

from dash import Dash, Input, Output, dcc, html

from src.preprocessing import DATA_FILE, detect_columns, load_data, pretty_name
from src.vizs_src.radar import create_visual as create_radar
from src.vizs_src.heatmap import create_visual as create_heatmap
from src.vizs_src.bar_chart import create_visual as create_bar_chart, TITLE as CHART_TITLE, EXPLANATION as CHART_EXPLANATION
from src.vizs_src.beeswarm import create_visual as create_beeswarm, prepare_beeswarm_data

# Add teammate visuals with direct imports, for example:
# from vizs_src.scatter_plot import create_visual as create_scatter_plot


DF = load_data(DATA_FILE)
DETECTED = detect_columns(DF)
TARGET_COL = DETECTED["target_col"]
FEATURES = DETECTED["features"]
DEFAULT_FEATURES = FEATURES[:3]
FEATURE_OPTIONS = [{"label": pretty_name(feature), "value": feature} for feature in FEATURES]
DEFAULT_LOW = 50
DEFAULT_HIGH = 80
BEESWARM_CACHE_DIR = Path(__file__).resolve().parents[1] / "preloaded" / "beeswarm"
BEESWARM_DATA = prepare_beeswarm_data(
    DF,
    features=FEATURES,
    target_col=TARGET_COL,
    student_id_col=DETECTED["student_id_col"],
)

HEATMAP_FIGURE_HEALTH, HEATMAP_TITLE, HEATMAP_EXPLANATION = create_heatmap(
    DF,
    target_col=TARGET_COL,
    mode="health",
)
HEATMAP_FIGURE_PAIRWISE, _, _ = create_heatmap(
    DF,
    target_col=TARGET_COL,
    mode="pairwise",
)

BEESWARM_FIGURE, BEESWARM_TITLE, BEESWARM_EXPLANATION = create_beeswarm(
    BEESWARM_DATA,
    features=DEFAULT_FEATURES,
    target_col=TARGET_COL,
    high_threshold=DEFAULT_HIGH,
    low_threshold=DEFAULT_LOW,
    display_mode="all",
    cache_dir=BEESWARM_CACHE_DIR,
)

app = Dash(__name__)
server = app.server
app.title = "GPADNA"


def make_control(label: str, component) -> html.Div:
    return html.Div([html.Label(label), component], className="control")


def make_feature_filter(store_id: str) -> html.Div:
    selected = set(DEFAULT_FEATURES)

    def make_chip(feature: str, is_selected: bool) -> html.Div:
        return html.Div(
            pretty_name(feature),
            className=f"feature-chip{' selected' if is_selected else ''}",
            draggable="true",
            **{"data-feature": feature},
        )

    return html.Div(
        [
            dcc.Store(id=store_id, data=DEFAULT_FEATURES),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Available", className="feature-zone-label"),
                            html.Div(
                                [make_chip(feature, False) for feature in FEATURES if feature not in selected],
                                className="feature-drop-zone",
                                **{"data-zone": "available"},
                            ),
                        ],
                        className="feature-zone",
                    ),
                    html.Div("→", className="feature-drag-arrow"),
                    html.Div(
                        [
                            html.Span("Selected", className="feature-zone-label"),
                            html.Div(
                                [make_chip(feature, True) for feature in DEFAULT_FEATURES],
                                className="feature-drop-zone",
                                **{"data-zone": "selected"},
                            ),
                        ],
                        className="feature-zone",
                    ),
                ],
                className="feature-drag-board",
                **{"data-store-id": store_id},
            ),
        ],
        className="feature-filter",
    )


def create_layout() -> html.Main:
    return html.Main(
        [
            html.Header(
                [
                    html.P("DNA OF STUDENT PERFORMANCE", className="eyebrow"),
                    html.H1("GPA DNA 🧬📊"),
                    html.P(
                        [
                            """
                        Ever wondered what the "genetic code" of academic success might look like?
                        GPA DNA invites you to explore how everyday habits vary across different groups of students.
                        """,
                            html.Br(),
                            html.Br(),
                            f"""
                        Using data from {len(DF):,} students, this interactive application helps you 
                        discover which habits are commonly associated with different levels
                        of academic performance. Adjust the grade thresholds, select the factors that interest you most,
                        and investigate the relationships that matter to you.
                        """,
                            html.Br(),
                            html.Br(),
                            """
                        From study routines and sleep schedules to social media use and extracurricular activities,
                        start exploring and see what combinations of habits shape the diverse "DNA" of student performance!
                        """,
                        ],
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
                                "Main features",
                                make_feature_filter("features"),
                            ),
                            make_control(
                                "Grade thresholds",
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span("Low grade"),
                                                html.Span("High grade"),
                                            ],
                                            className="threshold-labels",
                                        ),
                                        dcc.RangeSlider(
                                            id="grade-thresholds",
                                            value=[DEFAULT_LOW, DEFAULT_HIGH],
                                            min=0,
                                            max=100,
                                            step=1,
                                            marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            updatemode="mouseup",
                                            allowCross=False,
                                        ),
                                    ],
                                    className="threshold-slider",
                                    style={
                                        "marginRight": "50px"
                                    }
                                ),
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
                            html.H2(BEESWARM_TITLE),
                            html.P(BEESWARM_EXPLANATION),
                        ],
                        className="explanation",
                    ),
                    html.Div(
                        [
                            make_control(
                                "Beeswarm features",
                                make_feature_filter("beeswarm-features"),
                            ),
                        ],
                        className="control-grid",
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
                                            config={
                                                "displayModeBar": False,
                                                "scrollZoom": False,
                                                "responsive": True,
                                            },
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
                                            config={
                                                "displayModeBar": False,
                                                "scrollZoom": False,
                                                "responsive": True,
                                            },
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
    Output("status", "children"),
    Input("features", "data"),
    Input("grade-thresholds", "value"),
    Input("display-mode", "value"),
)
def update_radar(features, grade_thresholds, display_mode):
    features = features or []
    low_threshold, high_threshold = grade_thresholds or [DEFAULT_LOW, DEFAULT_HIGH]
    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)

    status = ""
    if high_threshold <= low_threshold:
        status = "High threshold must be greater than the low threshold."
    elif not features:
        status = "Select at least one radar feature."

    figure, title, explanation = create_radar(
        DF,
        features=features,
        target_col=TARGET_COL,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        display_mode=display_mode,
    )

    return figure, title, explanation, status


@app.callback(
    Output("bar-chart", "figure"),
    Input("features", "data"),
    Input("grade-thresholds", "value"),
    Input("display-mode", "value"),
)
def update_bar_chart(features, grade_thresholds, display_mode):
    features = features or []
    low_threshold, high_threshold = grade_thresholds or [DEFAULT_LOW, DEFAULT_HIGH]
    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)

    figure, _, _ = create_bar_chart(
        DF,
        features=features,
        target_col=TARGET_COL,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        display_mode=display_mode,
    )

    return figure


@app.callback(
    Output("beeswarm-chart", "figure"),
    Input("beeswarm-features", "data"),
    Input("grade-thresholds", "value"),
    Input("display-mode", "value"),
)
def update_beeswarm(beeswarm_features, grade_thresholds, display_mode):
    beeswarm_features = beeswarm_features or []
    low_threshold, high_threshold = grade_thresholds or [DEFAULT_LOW, DEFAULT_HIGH]
    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)

    figure, _, _ = create_beeswarm(
        BEESWARM_DATA,
        features=beeswarm_features,
        target_col=TARGET_COL,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        display_mode=display_mode,
        cache_dir=BEESWARM_CACHE_DIR,
    )

    return figure

