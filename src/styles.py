"""Design tokens and custom CSS for the Customer Sentiment Intelligence Dashboard.

Visual language
---------------
Warm off-white canvas, dark navy structural elements, one bright turquoise
accent, light grey borders, and exactly three semantic colours (teal / amber /
coral-red) so a reader can decode any chart without a legend lookup. Sized for a
1920x1080 presentation screenshot: no chart label below 11px.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
PAGE_BG = "#FAF8F5"          # warm off-white canvas
SURFACE = "#FFFFFF"          # analytical surfaces (cards, charts, tables)
SURFACE_ALT = "#F4F1EC"      # subtle alternate surface
NAVY = "#0F2544"             # dark navy titles and chart structure
NAVY_SOFT = "#22405F"
NAVY_TINT = "#5A7391"
BORDER = "#E6E3DD"           # light grey borders
BORDER_STRONG = "#D2CEC6"

TURQUOISE = "#00B3A4"        # bright turquoise accent
TURQUOISE_DARK = "#00897B"
TURQUOISE_SOFT = "#E1F5F2"

POSITIVE = "#00A896"         # teal - strong performance
NEUTRAL = "#E0A23C"          # amber - average performance
NEGATIVE = "#D9544A"         # coral-red - high negative rate

POSITIVE_SOFT = "#DEF3F0"
NEUTRAL_SOFT = "#FBEEDA"
NEGATIVE_SOFT = "#F9E2DF"

TEXT = "#1A2634"
TEXT_MUTED = "#66717D"
ACCENT = TURQUOISE

SENTIMENT_COLORS = {
    "Positive": POSITIVE,
    "Neutral": NEUTRAL,
    "Negative": NEGATIVE,
}

RECOMMENDATION_COLORS = {
    "Recommended": POSITIVE,
    "Not Recommended": NEGATIVE,
}

RATING_COLORS = {
    1: NEGATIVE,
    2: "#E07C72",
    3: NEUTRAL,
    4: "#4CBFAF",
    5: POSITIVE,
}

PERFORMANCE_COLORS = {
    "Strong": POSITIVE,
    "Average": NEUTRAL,
    "Needs Attention": NEGATIVE,
}

#: Neutral navy-to-turquoise ramp for volume heatmaps.
SEQUENTIAL_NAVY = [
    [0.0, "#F1F4F7"],
    [0.5, "#6E93AF"],
    [1.0, NAVY],
]

FONT_STACK = '"Segoe UI", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif'


def custom_css(presentation_mode: bool = False) -> str:
    """Return the dashboard stylesheet.

    ``presentation_mode`` tightens vertical rhythm and hides chrome that is not
    useful in a 1920x1080 screenshot destined for a slide deck.
    """
    presentation_extra = """
        header[data-testid="stHeader"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        div[data-testid="stAppViewContainer"] .block-container {
            padding-top: 1.0rem;
        }
        .csid-kpi { padding: 0.8rem 0.9rem 0.7rem 0.9rem; }
        .csid-kpi-note { display: none; }
        .csid-section { margin-top: 1.0rem; }
        .csid-hero { padding: 1.0rem 1.3rem; margin-bottom: 0.9rem; }
        .csid-hero h1 { font-size: 1.75rem; }
    """ if presentation_mode else ""

    return f"""
    <style>
    /* ------------------------------------------------------------------ */
    /* Canvas                                                              */
    /* ------------------------------------------------------------------ */
    html, body, [data-testid="stAppViewContainer"] {{
        background: {PAGE_BG};
        color: {TEXT};
        font-family: {FONT_STACK};
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.2rem;
        max-width: 1820px;
    }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    h1, h2, h3, h4, h5 {{
        color: {NAVY};
        font-family: {FONT_STACK};
        letter-spacing: -0.01em;
    }}

    /* ------------------------------------------------------------------ */
    /* Hero header                                                         */
    /* ------------------------------------------------------------------ */
    .csid-hero {{
        background: linear-gradient(120deg, {NAVY} 0%, {NAVY_SOFT} 100%);
        border-radius: 12px;
        border-bottom: 3px solid {TURQUOISE};
        padding: 1.4rem 1.7rem;
        color: #FFFFFF;
        box-shadow: 0 8px 22px rgba(15, 37, 68, 0.14);
        margin-bottom: 1.1rem;
    }}
    .csid-hero h1 {{
        color: #FFFFFF;
        font-size: 2.0rem;
        font-weight: 700;
        margin: 0 0 0.2rem 0;
        line-height: 1.15;
    }}
    .csid-hero p.csid-sub {{
        color: {TURQUOISE_SOFT};
        font-size: 1.0rem;
        margin: 0;
        font-weight: 400;
    }}
    .csid-hero .csid-hero-meta {{
        margin-top: 0.65rem;
        font-size: 0.78rem;
        color: #A6BCCF;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}

    /* ------------------------------------------------------------------ */
    /* Section headers                                                     */
    /* ------------------------------------------------------------------ */
    .csid-section {{
        margin-top: 1.5rem;
        margin-bottom: 0.65rem;
        border-left: 4px solid {TURQUOISE};
        padding-left: 0.75rem;
    }}
    .csid-section h3 {{
        margin: 0;
        font-size: 1.14rem;
        font-weight: 650;
        color: {NAVY};
    }}
    .csid-section p {{
        margin: 0.15rem 0 0 0;
        font-size: 0.85rem;
        color: {TEXT_MUTED};
    }}

    /* ------------------------------------------------------------------ */
    /* KPI cards                                                           */
    /* ------------------------------------------------------------------ */
    .csid-kpi {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.95rem 1.0rem 0.85rem 1.0rem;
        box-shadow: 0 1px 6px rgba(15, 37, 68, 0.05);
        height: 100%;
        transition: box-shadow 130ms ease-in-out;
    }}
    .csid-kpi:hover {{ box-shadow: 0 5px 14px rgba(15, 37, 68, 0.09); }}
    .csid-kpi-accent-positive  {{ border-top: 3px solid {POSITIVE}; }}
    .csid-kpi-accent-neutral   {{ border-top: 3px solid {NEUTRAL}; }}
    .csid-kpi-accent-negative  {{ border-top: 3px solid {NEGATIVE}; }}
    .csid-kpi-accent-navy      {{ border-top: 3px solid {NAVY}; }}
    .csid-kpi-accent-turquoise {{ border-top: 3px solid {TURQUOISE}; }}

    .csid-kpi-label {{
        font-size: 0.73rem;
        font-weight: 650;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {TEXT_MUTED};
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }}
    .csid-kpi-value {{
        font-size: 1.95rem;
        font-weight: 700;
        color: {NAVY};
        line-height: 1.12;
        margin-top: 0.28rem;
    }}
    .csid-kpi-note {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-top: 0.3rem;
        line-height: 1.35;
    }}
    .csid-help {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 1px solid {BORDER_STRONG};
        color: {TEXT_MUTED};
        font-size: 0.62rem;
        font-weight: 700;
        cursor: help;
        flex: none;
    }}

    /* ------------------------------------------------------------------ */
    /* Actionable Insight band                                             */
    /* ------------------------------------------------------------------ */
    .csid-actionable {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 5px solid {TURQUOISE};
        border-radius: 10px;
        padding: 1.0rem 1.25rem;
        margin: 0.9rem 0 1.1rem 0;
        box-shadow: 0 2px 10px rgba(0, 179, 164, 0.07);
    }}
    .csid-actionable-head {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.7rem;
    }}
    .csid-actionable-head .csid-tag {{
        background: {TURQUOISE};
        color: #FFFFFF;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
    }}
    .csid-actionable-head .csid-scope {{
        font-size: 0.79rem;
        color: {TEXT_MUTED};
    }}
    .csid-actionable-item {{
        padding: 0.55rem 0 0.55rem 0;
        border-top: 1px solid {BORDER};
    }}
    .csid-actionable-item:first-of-type {{ border-top: none; padding-top: 0; }}
    .csid-actionable-item .csid-ai-title {{
        font-size: 0.88rem;
        font-weight: 650;
        color: {NAVY};
        margin-bottom: 0.15rem;
    }}
    .csid-actionable-item .csid-ai-row {{
        font-size: 0.87rem;
        color: {TEXT};
        line-height: 1.5;
        margin: 0.1rem 0;
    }}
    .csid-actionable-item .csid-ai-row b {{ color: {NAVY}; }}
    .csid-actionable-item .csid-ai-label {{
        display: inline-block;
        min-width: 148px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {TURQUOISE_DARK};
        vertical-align: top;
    }}
    .csid-actionable-item .csid-ai-text {{
        display: inline-block;
        max-width: calc(100% - 160px);
    }}

    /* ------------------------------------------------------------------ */
    /* Panels, notes, badges                                               */
    /* ------------------------------------------------------------------ */
    .csid-panel {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        box-shadow: 0 1px 6px rgba(15, 37, 68, 0.05);
        margin-bottom: 0.75rem;
    }}
    .csid-panel h4 {{ margin: 0 0 0.4rem 0; font-size: 0.98rem; }}
    .csid-panel p, .csid-panel li {{
        font-size: 0.88rem;
        color: {TEXT};
        line-height: 1.5;
    }}
    .csid-note {{
        font-size: 0.81rem;
        color: {TEXT_MUTED};
        font-style: italic;
        margin: 0.2rem 0 0.6rem 0;
    }}
    .csid-badge {{
        display: inline-block;
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        font-size: 0.71rem;
        font-weight: 650;
        letter-spacing: 0.03em;
    }}
    .csid-badge-strong    {{ background: {POSITIVE_SOFT}; color: #046156; }}
    .csid-badge-average   {{ background: {NEUTRAL_SOFT};  color: #8A5C10; }}
    .csid-badge-attention {{ background: {NEGATIVE_SOFT}; color: #8E2F27; }}

    /* ------------------------------------------------------------------ */
    /* Sidebar                                                             */
    /* ------------------------------------------------------------------ */
    section[data-testid="stSidebar"] {{
        background: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.0rem; }}
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
        gap: 0.55rem;
    }}
    .csid-sidebar-title {{
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: {NAVY};
        border-bottom: 2px solid {TURQUOISE};
        display: inline-block;
        padding-bottom: 0.12rem;
        margin-bottom: 0.3rem;
    }}
    .csid-scope-box {{
        background: {TURQUOISE_SOFT};
        border: 1px solid #BFE6E1;
        border-radius: 8px;
        padding: 0.55rem 0.7rem;
        font-size: 0.78rem;
        color: {TEXT};
        line-height: 1.5;
    }}
    .csid-scope-box .csid-scope-count {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {NAVY};
        display: block;
    }}

    /* ------------------------------------------------------------------ */
    /* Tabs                                                                */
    /* ------------------------------------------------------------------ */
    /* The four dashboard views are a segmented control styled as a tab strip. */
    div[data-testid="stButtonGroup"] {{
        gap: 0.1rem !important;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 0.6rem;
    }}
    button[data-variant="segmented_control"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        border-bottom: 3px solid transparent !important;
        padding: 0.5rem 1.3rem !important;
        box-shadow: none !important;
        white-space: nowrap;
        transition: none !important;
    }}
    button[data-variant="segmented_control"] span,
    button[data-variant="segmented_control"] p,
    button[data-variant="segmented_control"] div {{
        font-size: 0.99rem !important;
        font-weight: 600 !important;
        color: {TEXT_MUTED} !important;
    }}
    button[data-variant="segmented_control"]:hover span {{
        color: {TURQUOISE_DARK} !important;
    }}
    button[data-variant="segmented_control"][aria-checked="true"] {{
        border-bottom: 3px solid {TURQUOISE} !important;
        background: transparent !important;
        background-color: transparent !important;
    }}
    button[data-variant="segmented_control"][aria-checked="true"] span,
    button[data-variant="segmented_control"][aria-checked="true"] p,
    button[data-variant="segmented_control"][aria-checked="true"] div {{
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Tables & inputs                                                     */
    /* ------------------------------------------------------------------ */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        background: {SURFACE};
        overflow: hidden;
    }}
    div[data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        background: {SURFACE};
    }}
    .stButton > button {{
        border-radius: 7px;
        border: 1px solid {BORDER_STRONG};
        font-weight: 600;
    }}
    .stButton > button:hover {{
        border-color: {TURQUOISE};
        color: {TURQUOISE_DARK};
    }}
    .stDownloadButton > button {{
        border-radius: 7px;
        border: 1px solid {TURQUOISE};
        color: {TURQUOISE_DARK};
        font-weight: 600;
    }}

    /* ------------------------------------------------------------------ */
    /* Review cards                                                        */
    /* ------------------------------------------------------------------ */
    .csid-review {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 0.75rem 1.0rem;
        margin-bottom: 0.5rem;
    }}
    .csid-review-positive {{ border-left: 4px solid {POSITIVE}; }}
    .csid-review-neutral  {{ border-left: 4px solid {NEUTRAL}; }}
    .csid-review-negative {{ border-left: 4px solid {NEGATIVE}; }}
    .csid-review-meta {{
        font-size: 0.75rem;
        color: {TEXT_MUTED};
        margin-bottom: 0.22rem;
    }}
    .csid-review-title {{
        font-weight: 650;
        color: {NAVY};
        font-size: 0.95rem;
    }}
    .csid-review-text {{
        font-size: 0.89rem;
        color: {TEXT};
        line-height: 1.5;
    }}

    /* ------------------------------------------------------------------ */
    /* Footer                                                              */
    /* ------------------------------------------------------------------ */
    .csid-footer {{
        margin-top: 2.0rem;
        padding-top: 0.85rem;
        border-top: 2px solid {TURQUOISE};
        font-size: 0.79rem;
        color: {TEXT_MUTED};
        line-height: 1.65;
    }}
    .csid-footer b {{ color: {NAVY}; }}
    .csid-footer a {{ color: {TURQUOISE_DARK}; text-decoration: none; }}

    /* ------------------------------------------------------------------ */
    /* Simulation & Decision Lab                                           */
    /* ------------------------------------------------------------------ */
    .csid-sim-banner {{
        background: {TURQUOISE_SOFT};
        border: 1px solid {TURQUOISE};
        border-left: 4px solid {TURQUOISE_DARK};
        border-radius: 10px;
        padding: 0.85rem 1.05rem;
        color: {TEXT};
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 0.85rem;
    }}
    .csid-sim-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.55rem 0 0.2rem 0;
    }}
    .csid-sim-chip {{
        background: {SURFACE};
        border: 1px solid {BORDER_STRONG};
        border-radius: 8px;
        padding: 0.45rem 0.75rem;
        font-size: 0.84rem;
        color: {TEXT};
    }}
    .csid-sim-chip b {{ color: {NAVY}; }}
    .csid-sim-kpi {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.85rem 0.95rem 0.75rem 0.95rem;
        box-shadow: 0 1px 2px rgba(15, 37, 68, 0.04);
        min-height: 118px;
    }}
    .csid-sim-kpi-accent-positive  {{ border-top: 3px solid {POSITIVE}; }}
    .csid-sim-kpi-accent-neutral   {{ border-top: 3px solid {NEUTRAL}; }}
    .csid-sim-kpi-accent-negative  {{ border-top: 3px solid {NEGATIVE}; }}
    .csid-sim-kpi-accent-navy      {{ border-top: 3px solid {NAVY}; }}
    .csid-sim-kpi-accent-turquoise {{ border-top: 3px solid {TURQUOISE}; }}
    .csid-sim-kpi-label {{
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: {TEXT_MUTED};
        font-weight: 650;
        margin-bottom: 0.25rem;
    }}
    .csid-sim-kpi-value {{
        font-size: 1.55rem;
        font-weight: 700;
        color: {NAVY};
        line-height: 1.15;
    }}
    .csid-sim-kpi-delta {{
        margin-top: 0.35rem;
        font-size: 0.88rem;
        font-weight: 600;
    }}
    .csid-sim-delta-positive {{ color: #046156; }}
    .csid-sim-delta-negative {{ color: #8E2F27; }}
    .csid-sim-delta-neutral  {{ color: #8A5C10; }}
    .csid-sim-kpi-tag {{
        display: inline-block;
        margin-top: 0.4rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: {TURQUOISE_DARK};
        background: {TURQUOISE_SOFT};
        border-radius: 4px;
        padding: 0.15rem 0.45rem;
    }}
    .csid-sim-kpi-note {{
        margin-top: 0.25rem;
        font-size: 0.8rem;
        color: {TEXT_MUTED};
    }}
    .csid-workflow {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        align-items: stretch;
        margin: 0.4rem 0 0.8rem 0;
    }}
    .csid-workflow-step {{
        background: {SURFACE};
        border: 1px solid {BORDER_STRONG};
        border-radius: 8px;
        padding: 0.55rem 0.7rem;
        min-width: 118px;
        flex: 1 1 118px;
        text-align: center;
        font-size: 0.82rem;
        font-weight: 650;
        color: {NAVY};
        box-shadow: 0 1px 2px rgba(15, 37, 68, 0.04);
    }}
    .csid-workflow-arrow {{
        align-self: center;
        color: {TURQUOISE_DARK};
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0 0.1rem;
    }}
    .csid-owner-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.55rem;
        margin-top: 0.35rem;
    }}
    .csid-owner-card {{
        background: {SURFACE_ALT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
        font-size: 0.86rem;
        line-height: 1.4;
    }}
    .csid-owner-card b {{ color: {NAVY}; display: block; margin-bottom: 0.2rem; }}
    .csid-disclaimer {{
        background: {NEUTRAL_SOFT};
        border: 1px solid {NEUTRAL};
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        color: {TEXT};
        font-size: 0.9rem;
        margin-top: 0.6rem;
    }}
    .csid-group-label {{
        font-size: 0.84rem;
        font-weight: 700;
        color: {NAVY};
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin: 0.35rem 0 0.45rem 0;
    }}

    /* ------------------------------------------------------------------ */
    /* Guided Tour (UI onboarding only)                                    */
    /* ------------------------------------------------------------------ */
    .csid-tour-anchor {{
        height: 0;
        width: 0;
        overflow: hidden;
        margin: 0;
        padding: 0;
    }}
    .csid-tour-anchor-active {{
        outline: none;
    }}
    .csid-tour-panel {{
        background: {SURFACE};
        border: 1px solid {BORDER_STRONG};
        border-left: 4px solid {TURQUOISE};
        border-radius: 12px;
        box-shadow: 0 8px 28px rgba(15, 37, 68, 0.12);
        padding: 0.95rem 1.1rem 0.85rem 1.1rem;
        margin: 0.35rem 0 0.85rem 0;
    }}
    .csid-tour-panel-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.45rem;
    }}
    .csid-tour-badge {{
        display: inline-block;
        background: {TURQUOISE_SOFT};
        color: {TURQUOISE_DARK};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-radius: 4px;
        padding: 0.18rem 0.5rem;
    }}
    .csid-tour-progress {{
        font-size: 0.84rem;
        font-weight: 650;
        color: {TEXT_MUTED};
        white-space: nowrap;
    }}
    .csid-tour-title {{
        margin: 0 0 0.45rem 0;
        color: {NAVY};
        font-size: 1.12rem;
        line-height: 1.3;
        font-weight: 700;
    }}
    .csid-tour-body {{
        color: {TEXT};
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 0.35rem;
    }}
    .csid-tour-tip {{
        margin: 0.45rem 0 0 0;
        padding: 0.45rem 0.65rem;
        background: {SURFACE_ALT};
        border-radius: 8px;
        color: {TEXT_MUTED};
        font-size: 0.86rem;
        line-height: 1.45;
    }}
    /* Soft highlight around major regions while their tour step is active */
    .csid-tour-region {{
        border-radius: 10px;
        transition: box-shadow 0.2s ease, background-color 0.2s ease;
    }}
    .csid-tour-region-active {{
        box-shadow: 0 0 0 3px {TURQUOISE}, 0 0 0 6px rgba(0, 179, 164, 0.18);
        background: rgba(225, 245, 242, 0.35);
        padding: 0.35rem 0.45rem;
        margin: 0.15rem 0 0.55rem 0;
    }}
    /* Header launcher: keyed widget (stable st-key-guided_tour_start) */
    div[class*="st-key-guided_tour_start"] {{
        margin-top: 0.15rem;
        padding: 0.55rem 0.65rem;
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_SOFT} 100%);
        border: 1px solid rgba(0, 179, 164, 0.55);
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 37, 68, 0.12);
    }}
    div[class*="st-key-guided_tour_start"] button {{
        background: {TURQUOISE} !important;
        color: #ffffff !important;
        border: 1px solid {TURQUOISE_DARK} !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        white-space: normal !important;
        min-height: 2.6rem;
        line-height: 1.25;
    }}
    div[class*="st-key-guided_tour_start"] button:hover {{
        background: {TURQUOISE_DARK} !important;
        border-color: {TURQUOISE_DARK} !important;
    }}
    /* Keep the Escape-key helper iframe invisible */
    iframe[src^="data:text/html"] {{
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        pointer-events: none !important;
        border: 0 !important;
    }}

    {presentation_extra}
    </style>
    """


def plotly_layout(height: int = 380, **overrides) -> dict:
    """Consistent Plotly layout defaults for every chart in the dashboard."""
    layout = dict(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, size=13, color=TEXT),
        title=dict(font=dict(size=15, color=NAVY), x=0.0, xanchor="left"),
        margin=dict(l=60, r=28, t=52, b=52),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER_STRONG,
            font=dict(size=12, family=FONT_STACK, color=TEXT),
        ),
        xaxis=dict(
            gridcolor=BORDER,
            linecolor=BORDER_STRONG,
            zerolinecolor=BORDER,
            tickfont=dict(size=12),
            title=dict(font=dict(size=12, color=TEXT_MUTED)),
        ),
        yaxis=dict(
            gridcolor=BORDER,
            linecolor=BORDER_STRONG,
            zerolinecolor=BORDER,
            tickfont=dict(size=12),
            title=dict(font=dict(size=12, color=TEXT_MUTED)),
        ),
        transition=dict(duration=0),
    )
    layout.update(overrides)
    return layout
