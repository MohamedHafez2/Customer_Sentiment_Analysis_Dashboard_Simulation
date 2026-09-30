"""Customer Sentiment Intelligence Dashboard
Turning Online Reviews into Actionable Business Insights

A descriptive and diagnostic analytics dashboard for the MBA case study
"Customer Sentiment Analysis from Online Reviews".

Every figure is calculated directly from the supplied dataset by aggregation.
There is no machine-learning, NLP or predictive component in this application:
sentiment is a rating-based business classification, and review text is used
only for search and display.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from html import escape as html_escape
from typing import Any, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from src import analytics, charts, data_cleaning, data_loader, guided_tour, simulation, styles

# --------------------------------------------------------------------------- #
# Page configuration - must be the first Streamlit call
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Customer Sentiment Intelligence Dashboard",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "about": "Customer Sentiment Intelligence Dashboard - MBA Case Studies "
        "in AI for Business. All analytics are calculated from the supplied "
        "Women's E-Commerce Clothing Reviews dataset."
    },
)

DASHBOARD_TITLE = "Customer Sentiment Intelligence Dashboard"
DASHBOARD_SUBTITLE = "Turning Online Reviews into Actionable Business Insights"

TABS = (
    "Executive Overview",
    "Product Performance",
    "Customer & Review Insights",
    "Data Quality",
    "Simulation & Decision Lab",
)

FILTER_DEFAULTS: dict[str, Any] = {
    "f_divisions": [],
    "f_departments": [],
    "f_classes": [],
    "f_ratings": [],
    "f_sentiments": [],
    "f_recommendations": [],
    "f_age_bands": [],
    "f_has_text": "All records",
    "f_min_sample": 50,
}

TEXT_AVAILABILITY_OPTIONS = (
    "All records",
    "With review text only",
    "Without review text only",
)


# --------------------------------------------------------------------------- #
# Cached data plumbing - the dataset is read and cleaned once per session
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading the supplied dataset...")
def load_bundled_dataset() -> tuple[pd.DataFrame, str]:
    return data_loader.load_default_dataset()


@st.cache_data(show_spinner="Reading the uploaded file...")
def load_uploaded_dataset(payload: bytes, filename: str) -> pd.DataFrame:
    frame = data_loader.read_any(io.BytesIO(payload), filename)
    return data_loader.coerce_types(frame)


@st.cache_data(show_spinner="Cleaning the dataset...")
def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the validated cleaning pipeline. Cached, so it runs once per file."""
    return data_cleaning.clean_dataset(raw)


@st.cache_data(show_spinner=False)
def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------------------- #
# Small HTML renderers
# --------------------------------------------------------------------------- #
def html(markup: str) -> str:
    """Collapse indented HTML into a single line before handing it to Streamlit.

    ``st.markdown`` runs the string through a Markdown parser first, and any
    line left with four or more leading spaces is turned into a code block
    instead of rendered as HTML.
    """
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def render_hero(
    source_name: str, rows: int, total_rows: int, *, presentation: bool = False
) -> None:
    scope = (
        f"{rows:,} of {total_rows:,} reviews in scope"
        if rows != total_rows
        else f"{rows:,} reviews"
    )
    header_left, header_right = st.columns([4.6, 1.55], vertical_alignment="top")
    with header_left:
        st.markdown(
            html(
                f"""
                <div class="csid-hero">
                  <h1>{DASHBOARD_TITLE}</h1>
                  <p class="csid-sub">{DASHBOARD_SUBTITLE}</p>
                  <div class="csid-hero-meta">
                    Data source: {source_name} &nbsp;&middot;&nbsp; {scope}
                    &nbsp;&middot;&nbsp; MBA Case Studies in AI for Business
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    with header_right:
        guided_tour.render_start_button(presentation=presentation)


def section(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="csid-section"><h3>{title}</h3>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str, value: str, note: str, formula: str, accent: str = "navy"
) -> None:
    st.markdown(
        html(
            f"""
            <div class="csid-kpi csid-kpi-accent-{accent}">
              <div class="csid-kpi-label">{label}
                <span class="csid-help" title="{formula}">?</span>
              </div>
              <div class="csid-kpi-value">{value}</div>
              <div class="csid-kpi-note">{note}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def actionable_insight(
    insights: Sequence[analytics.ActionableInsight],
    scope_note: str = "",
) -> None:
    """Render the turquoise-bordered Actionable Insight band."""
    if not insights:
        return
    items = []
    for insight in insights:
        items.append(
            html(
                f"""
                <div class="csid-actionable-item">
                  <div class="csid-ai-title">{insight.title}</div>
                  <div class="csid-ai-row">
                    <span class="csid-ai-label">What the data shows</span>
                    <span class="csid-ai-text">{insight.what}</span>
                  </div>
                  <div class="csid-ai-row">
                    <span class="csid-ai-label">Why it matters</span>
                    <span class="csid-ai-text">{insight.why}</span>
                  </div>
                  <div class="csid-ai-row">
                    <span class="csid-ai-label">Area to focus on</span>
                    <span class="csid-ai-text">{insight.focus}</span>
                  </div>
                </div>
                """
            )
        )
    scope_html = f'<span class="csid-scope">{scope_note}</span>' if scope_note else ""
    st.markdown(
        html(
            f"""
            <div class="csid-actionable">
              <div class="csid-actionable-head">
                <span class="csid-tag">Actionable Insight</span>{scope_html}
              </div>
              {''.join(items)}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def panel(title: str, body_html: str) -> None:
    st.markdown(
        html(f'<div class="csid-panel"><h4>{title}</h4>{body_html}</div>'),
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<p class="csid-note">{text}</p>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        html(
            """
            <div class="csid-footer">
              <b>MBA Case Studies in AI for Business | Customer Sentiment
              Analysis</b><br>
              Data Source: Women&rsquo;s E-Commerce Clothing Reviews &mdash;
              Kaggle &nbsp;&middot;&nbsp;
              <a href="https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews"
              target="_blank">kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews</a><br>
              All results are calculated from the supplied historical dataset. No
              unavailable financial, time-series, geographical, or predictive
              variables are inferred.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Table helpers
# --------------------------------------------------------------------------- #
PERCENT_COLUMNS = (
    "Share %", "Recommendation Rate", "Positive Rate", "Neutral Rate",
    "Negative Rate", "Share of Helpful Votes %", "Completeness %", "Percentage",
    "Missing %", "Text Coverage %",
    "Baseline Positive Rate", "Simulated Positive Rate",
    "Baseline Negative Rate", "Simulated Negative Rate",
    "Baseline Recommendation Rate", "Simulated Recommendation Rate",
)
DECIMAL_COLUMNS = (
    "Avg Rating", "Avg Helpful Votes", "Median Helpful Votes",
    "Avg Review Length",
    "Baseline Average Rating", "Simulated Average Rating",
)
INTEGER_COLUMNS = (
    "Reviews", "Review Count", "Negative Reviews", "Positive Reviews",
    "Neutral Reviews", "Recommended Reviews", "Not Recommended Reviews",
    "Helpful Votes", "Total Helpful Votes", "Reviews With Text",
    "Reviews With >=1 Vote", "Affected Rows", "Missing", "Present", "Rows",
    "Count", "Positive Feedback Count", "Rating", "Age", "Clothing ID",
    "Rows Affected", "#", "Non-Null", "Unique",
    "Successful Negative Cases", "Remaining Negative Reviews", "Improved Cases",
)

PERFORMANCE_ICON = {
    "Strong": "Strong",
    "Average": "Average",
    "Needs Attention": "Needs Attention",
}


def show_table(
    frame: pd.DataFrame,
    height: int | None = None,
    column_config: dict[str, Any] | None = None,
    hide_index: bool = True,
) -> None:
    """Consistent numeric formatting for every table in the dashboard."""
    if frame.empty:
        st.info("No rows match the current filter selection.")
        return

    config: dict[str, Any] = dict(column_config or {})
    for column in frame.columns:
        if column in config:
            continue
        if column in PERCENT_COLUMNS:
            config[column] = st.column_config.NumberColumn(column, format="%.1f%%")
        elif column in DECIMAL_COLUMNS:
            config[column] = st.column_config.NumberColumn(column, format="%.2f")
        elif column in INTEGER_COLUMNS:
            config[column] = st.column_config.NumberColumn(column, format="%d")
    # `height` must be omitted rather than passed as None: recent Streamlit
    # versions reject None explicitly.
    extra = {"height": int(height)} if height else {}
    st.dataframe(
        frame,
        hide_index=hide_index,
        width="stretch",
        column_config=config,
        **extra,
    )


STRONG_CELL = f"background-color: {styles.POSITIVE_SOFT}; color: #046156;"
AVERAGE_CELL = f"background-color: {styles.NEUTRAL_SOFT}; color: #8A5C10;"
ATTENTION_CELL = f"background-color: {styles.NEGATIVE_SOFT}; color: #8E2F27;"


def style_performance_matrix(
    frame: pd.DataFrame,
    baseline_rating: float,
    baseline_recommendation: float,
    baseline_negative: float,
):
    """Conditional formatting: teal strong, amber average, coral-red attention.

    Thresholds are relative to the filtered baselines, so the colours mean the
    same thing under every filter combination.
    """

    def band_cell(value: Any) -> str:
        if value == "Strong":
            return STRONG_CELL + " font-weight: 650;"
        if value == "Needs Attention":
            return ATTENTION_CELL + " font-weight: 650;"
        return AVERAGE_CELL + " font-weight: 650;"

    def rating_cell(value: Any) -> str:
        gap = float(value) - baseline_rating
        if gap >= 0.10:
            return STRONG_CELL
        if gap <= -0.20:
            return ATTENTION_CELL
        return AVERAGE_CELL

    def recommendation_cell(value: Any) -> str:
        gap = float(value) - baseline_recommendation
        if gap >= 2.0:
            return STRONG_CELL
        if gap <= -5.0:
            return ATTENTION_CELL
        return AVERAGE_CELL

    def negative_cell(value: Any) -> str:
        gap = float(value) - baseline_negative
        if gap >= analytics.ATTENTION_GAP_PP:
            return ATTENTION_CELL + " font-weight: 650;"
        if gap >= analytics.AVERAGE_GAP_PP:
            return AVERAGE_CELL
        return STRONG_CELL

    return (
        frame.style.map(band_cell, subset=["Performance"])
        .map(rating_cell, subset=["Avg Rating"])
        .map(recommendation_cell, subset=["Recommendation Rate"])
        .map(negative_cell, subset=["Negative Rate"])
        .format(
            {
                "Review Count": "{:,.0f}",
                "Avg Rating": "{:.2f}",
                "Recommended Reviews": "{:,.0f}",
                "Recommendation Rate": "{:.1f}%",
                "Negative Reviews": "{:,.0f}",
                "Negative Rate": "{:.1f}%",
                "Total Helpful Votes": "{:,.0f}",
            }
        )
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def reset_filters() -> None:
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


def sanitize_multiselect(key: str, options: Sequence[Any]) -> None:
    """Drop stored selections that upstream filters removed from scope."""
    current = st.session_state.get(key, [])
    if current:
        st.session_state[key] = [v for v in current if v in options]


def sorted_unique(frame: pd.DataFrame, column: str) -> list[str]:
    return sorted(frame[column].astype("object").astype(str).dropna().unique().tolist())


def build_sidebar(clean: pd.DataFrame, source_name: str) -> dict[str, Any]:
    """Render the compact sidebar and return the active filter selections."""
    with st.sidebar:
        active_anchor = guided_tour.active_anchor_id()

        st.markdown(
            '<div class="csid-sidebar-title">Data Source</div>', unsafe_allow_html=True
        )
        guided_tour.tour_anchor("tour-data-source", active_anchor)
        if active_anchor == "tour-data-source":
            st.markdown(
                '<div class="csid-tour-region csid-tour-region-active">'
                '<span class="csid-tour-badge">Tour focus</span></div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Active file: **{source_name}**")
        st.file_uploader(
            "Replace the dataset (CSV, XLSX, XLS)",
            type=["csv", "xlsx", "xls"],
            help="The supplied dataset loads automatically. Upload a file with "
            "the same schema to analyse a different review export.",
            key="uploader",
        )

        guided_tour.tour_anchor("tour-presentation", active_anchor)
        if active_anchor == "tour-presentation":
            st.markdown(
                '<div class="csid-tour-region csid-tour-region-active">'
                '<span class="csid-tour-badge">Tour focus</span></div>',
                unsafe_allow_html=True,
            )
        st.toggle(
            "Presentation Mode",
            key="presentation_mode",
            help="Compact, screenshot-ready layout for 1920x1080 PowerPoint "
            "capture: hides the sidebar and secondary controls and keeps the "
            "KPIs, charts and insights visible.",
        )

        st.divider()
        header = st.columns([1.6, 1])
        with header[0]:
            st.markdown(
                '<div class="csid-sidebar-title">Global Filters</div>',
                unsafe_allow_html=True,
            )
        with header[1]:
            st.button(
                "Clear All",
                on_click=reset_filters,
                width="stretch",
                help="Reset every filter to its default.",
            )

        guided_tour.tour_anchor("tour-filters", active_anchor)
        if active_anchor == "tour-filters":
            st.markdown(
                '<div class="csid-tour-region csid-tour-region-active">'
                '<span class="csid-tour-badge">Tour focus</span></div>',
                unsafe_allow_html=True,
            )

        # Cascading product taxonomy - Department options follow Division, and
        # Class options follow Department, so the three stay synchronized.
        division_options = sorted_unique(clean, "Division Name")
        sanitize_multiselect("f_divisions", division_options)
        divisions = st.multiselect("Division", division_options, key="f_divisions")

        scope = clean
        if divisions:
            scope = scope[scope["Division Name"].astype(str).isin(divisions)]
        department_options = sorted_unique(scope, "Department Name")
        sanitize_multiselect("f_departments", department_options)
        departments = st.multiselect("Department", department_options, key="f_departments")

        if departments:
            scope = scope[scope["Department Name"].astype(str).isin(departments)]
        class_options = sorted_unique(scope, "Class Name")
        sanitize_multiselect("f_classes", class_options)
        classes = st.multiselect("Class", class_options, key="f_classes")

        rating_options = [
            int(r)
            for r in sorted(
                pd.to_numeric(clean["Rating"], errors="coerce").dropna().unique()
            )
        ]
        sanitize_multiselect("f_ratings", rating_options)
        ratings = st.multiselect(
            "Rating", rating_options, key="f_ratings", format_func=lambda r: f"{r} star"
        )

        present_sentiments = set(clean["Sentiment"].astype("object").astype(str))
        sentiment_options = [
            s for s in data_cleaning.SENTIMENT_ORDER if s in present_sentiments
        ]
        sanitize_multiselect("f_sentiments", sentiment_options)
        sentiments = st.multiselect("Sentiment", sentiment_options, key="f_sentiments")

        present_reco = set(clean["Recommendation Label"].astype("object").astype(str))
        recommendation_options = [
            r for r in data_cleaning.RECOMMENDATION_ORDER if r in present_reco
        ]
        sanitize_multiselect("f_recommendations", recommendation_options)
        recommendations = st.multiselect(
            "Recommendation", recommendation_options, key="f_recommendations"
        )

        present_bands = set(clean["Age Band"].astype("object").astype(str))
        age_band_options = [
            b
            for b in list(data_cleaning.AGE_BAND_ORDER)
            + [data_cleaning.UNKNOWN_CATEGORY]
            if b in present_bands
        ]
        sanitize_multiselect("f_age_bands", age_band_options)
        age_bands = st.multiselect("Age Band", age_band_options, key="f_age_bands")

        has_text = st.radio(
            "Has Review Text",
            TEXT_AVAILABILITY_OPTIONS,
            key="f_has_text",
            help="Rows without review text are excluded from review display and "
            "keyword search, but remain valid for rating, recommendation and "
            "category analytics.",
        )

        min_sample = st.slider(
            "Minimum Category Sample Size",
            min_value=0,
            max_value=300,
            step=10,
            key="f_min_sample",
            help="Categories with fewer reviews than this are suppressed from "
            "comparisons and insights, so a handful of reviews cannot look like "
            "a systemic problem.",
        )

        return {
            "divisions": divisions,
            "departments": departments,
            "classes": classes,
            "ratings": ratings,
            "sentiments": sentiments,
            "recommendations": recommendations,
            "age_bands": age_bands,
            "has_review_text": has_text,
            "min_sample": min_sample,
        }


def render_sidebar_scope(
    filters: dict[str, Any], filtered: pd.DataFrame, clean: pd.DataFrame
) -> None:
    with st.sidebar:
        st.divider()
        active_anchor = guided_tour.active_anchor_id()
        guided_tour.tour_anchor("tour-scope", active_anchor)
        if active_anchor == "tour-scope":
            st.markdown(
                '<div class="csid-tour-region csid-tour-region-active">'
                '<span class="csid-tour-badge">Tour focus</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="csid-sidebar-title">Current Scope</div>',
            unsafe_allow_html=True,
        )
        share = len(filtered) / len(clean) * 100 if len(clean) else 0.0
        lines = analytics.filter_summary(filters, len(filtered), len(clean))
        st.markdown(
            html(
                f"""
                <div class="csid-scope-box">
                  <span class="csid-scope-count">{len(filtered):,} records</span>
                  {share:.1f}% of the {len(clean):,} in the dataset
                  <div style="margin-top:0.4rem">{'<br>'.join(lines)}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download Filtered Data (CSV)",
            data=to_csv_bytes(filtered),
            file_name=f"filtered_reviews_{len(filtered)}_rows.csv",
            mime="text/csv",
            width="stretch",
            disabled=filtered.empty,
        )
        st.divider()
        st.caption(
            "Sentiment here is a rating-based business classification "
            "(1-2 Negative, 3 Neutral, 4-5 Positive), not a predicted score. "
            "The dataset has no date, revenue, geography, delivery or gender "
            "field, so no such analysis appears in this dashboard."
        )


# --------------------------------------------------------------------------- #
# Tab 1 - Executive Overview
# --------------------------------------------------------------------------- #
def tab_executive_overview(
    filtered: pd.DataFrame, min_sample: int, presentation: bool
) -> None:
    kpis = analytics.compute_kpis(filtered)

    section(
        "Executive KPIs",
        "Every figure is recalculated from the filtered dataset. Hover the "
        "question mark on any card for its formula.",
    )

    row = st.columns(4, gap="small")
    with row[0]:
        kpi_card(
            "Total Reviews",
            f"{int(kpis['total_records']):,}",
            f"Across {int(kpis['unique_clothing_ids']):,} distinct products in "
            f"scope.",
            "Count of rows in the filtered dataset.",
            "navy",
        )
    with row[1]:
        kpi_card(
            "Reviews With Text",
            f"{int(kpis['records_with_text']):,}",
            f"{kpis['text_coverage_rate']:.1f}% of records carry review text; "
            f"{int(kpis['missing_review_text']):,} are rating-only.",
            "Count of rows where Review Text was present in the source file, "
            "captured before the placeholder was written.",
            "navy",
        )
    with row[2]:
        kpi_card(
            "Average Rating",
            f"{kpis['average_rating']:.2f} / 5",
            f"Median {kpis['median_rating']:.0f} stars. The only ordinal "
            f"satisfaction measure in the dataset.",
            "Mean of the Rating column across the filtered rows.",
            "positive" if kpis["average_rating"] >= 4 else "neutral",
        )
    with row[3]:
        kpi_card(
            "Recommendation Rate",
            f"{kpis['recommendation_rate']:.2f}%",
            f"{int(kpis['recommended_count']):,} recommended against "
            f"{int(kpis['not_recommended_count']):,} not recommended.",
            "Rows with Recommended IND = 1 divided by all rows with a valid "
            "0/1 flag, times 100.",
            "positive" if kpis["recommendation_rate"] >= 80 else "neutral",
        )

    row = st.columns(4, gap="small")
    with row[0]:
        kpi_card(
            "Positive Sentiment Rate",
            f"{kpis['positive_rate']:.2f}%",
            f"{int(kpis['positive_reviews']):,} reviews rated 4 or 5 stars.",
            "Reviews rated 4-5 divided by all classified reviews, times 100.",
            "positive",
        )
    with row[1]:
        kpi_card(
            "Neutral Sentiment Rate",
            f"{kpis['neutral_rate']:.2f}%",
            f"{int(kpis['neutral_reviews']):,} reviews rated exactly 3 stars - "
            f"the cheapest group to move upward.",
            "Reviews rated 3 divided by all classified reviews, times 100.",
            "neutral",
        )
    with row[2]:
        kpi_card(
            "Negative Sentiment Rate",
            f"{kpis['negative_rate']:.2f}%",
            f"{int(kpis['negative_reviews']):,} reviews rated 1 or 2 stars. This "
            f"is the baseline every segment is compared against.",
            "Reviews rated 1-2 divided by all classified reviews, times 100.",
            "negative",
        )
    with row[3]:
        kpi_card(
            "Total Helpful Votes",
            f"{int(kpis['total_positive_feedback']):,}",
            f"Averaging {kpis['average_positive_feedback']:.2f} per review. This "
            f"counts how useful other shoppers found a review, not how positive "
            f"it was.",
            "Sum of the Positive Feedback Count column, which records how many "
            "readers marked a review as helpful.",
            "turquoise",
        )

    # ------------------------------------------------------------------ #
    section(
        "Sentiment, Rating and Recommendation",
        "Sentiment is derived from the star rating (1-2 Negative, 3 Neutral, "
        "4-5 Positive). Recommendation is the customer's own stated intent, "
        "recorded separately in the source data.",
    )
    row = st.columns([1, 1.15], gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.sentiment_donut(analytics.sentiment_distribution(filtered)),
            width="stretch",
        )
    with row[1]:
        st.plotly_chart(
            charts.rating_bar(analytics.rating_distribution(filtered)),
            width="stretch",
        )

    row = st.columns(2, gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.recommendation_bar(analytics.recommendation_distribution(filtered)),
            width="stretch",
        )
    with row[1]:
        st.plotly_chart(
            charts.recommendation_by_sentiment_stack(
                analytics.recommendation_by_sentiment(filtered)
            ),
            width="stretch",
        )

    # ------------------------------------------------------------------ #
    section(
        "Department Performance",
        f"Review volume, average rating, recommendation rate and "
        f"negative-review rate side by side. Departments with fewer than "
        f"{min_sample} reviews are suppressed.",
    )
    departments = analytics.department_performance(filtered, min_reviews=min_sample)
    st.plotly_chart(
        charts.segment_overview(
            departments,
            "Department Name",
            "Department Performance: volume, rating, advocacy and criticism",
        ),
        width="stretch",
    )

    actionable_insight(
        analytics.executive_insights(filtered, min_reviews=min_sample),
        scope_note=f"Calculated from the {len(filtered):,} filtered records "
        f"&middot; minimum category sample {min_sample}",
    )

    if not presentation:
        with st.expander("Department performance table"):
            show_table(departments)


# --------------------------------------------------------------------------- #
# Tab 2 - Product Performance
# --------------------------------------------------------------------------- #
def tab_product_performance(
    filtered: pd.DataFrame, min_sample: int, presentation: bool
) -> None:
    kpis = analytics.compute_kpis(filtered)
    baseline_negative = kpis["negative_rate"]
    baseline_recommendation = kpis["recommendation_rate"]
    baseline_rating = kpis["average_rating"]

    section(
        "Division Performance",
        "The highest level of the product hierarchy: review count, average "
        "rating, recommendation rate and negative rate.",
    )
    divisions = analytics.division_performance(filtered, min_reviews=min_sample)
    row = st.columns([1.35, 1], gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.segment_overview(
                divisions,
                "Division Name",
                "Division Performance: volume, rating, advocacy and criticism",
                height=400,
            ),
            width="stretch",
        )
    with row[1]:
        show_table(
            divisions[
                [
                    "Division Name", "Reviews", "Avg Rating", "Recommendation Rate",
                    "Negative Reviews", "Negative Rate",
                ]
            ]
            if not divisions.empty
            else divisions
        )

    # ------------------------------------------------------------------ #
    section(
        "Department Comparison",
        f"Volume, satisfaction and advocacy per department, each compared "
        f"against the filtered baseline.",
    )
    departments = analytics.department_performance(filtered, min_reviews=min_sample)
    row = st.columns(3, gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.horizontal_bar(
                departments,
                "Department Name",
                "Reviews",
                "Reviews by Department",
                height=390,
                color=charts.VOLUME_COLOR,
                x_title="Reviews",
            ),
            width="stretch",
        )
    with row[1]:
        st.plotly_chart(
            charts.rating_by_segment(
                departments,
                "Department Name",
                baseline_rating,
                f"Average Rating by Department (baseline {baseline_rating:.2f})",
                height=390,
            ),
            width="stretch",
        )
    with row[2]:
        st.plotly_chart(
            charts.rate_by_segment(
                departments,
                "Department Name",
                "Recommendation Rate",
                f"Recommendation Rate by Department "
                f"(baseline {baseline_recommendation:.1f}%)",
                baseline_recommendation,
                height=390,
                high_is_good=True,
            ),
            width="stretch",
        )

    # ------------------------------------------------------------------ #
    section(
        "Top Ten Product Classes by Review Volume",
        "Where customer attention concentrates, and how those classes perform "
        "on advocacy against criticism.",
    )
    top_classes = analytics.top_classes_by_volume(filtered, top_n=10)
    row = st.columns([1.1, 1], gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.horizontal_bar(
                top_classes,
                "Class Name",
                "Reviews",
                "Top Ten Product Classes by Review Volume",
                height=430,
                color=styles.NAVY,
                x_title="Reviews",
            ),
            width="stretch",
        )
    with row[1]:
        st.plotly_chart(
            charts.grouped_rate_chart(
                top_classes.head(8),
                "Class Name",
                "Advocacy against Criticism in the Highest-Volume Classes",
                height=430,
            ),
            width="stretch",
        )

    # ------------------------------------------------------------------ #
    section(
        "Product Class Performance Matrix",
        f"Every product class with at least {min_sample} reviews. Colour marks "
        f"performance against the filtered baselines: teal is strong, amber is "
        f"average, coral-red needs attention.",
    )
    matrix = analytics.class_performance_matrix(filtered, min_reviews=min_sample)
    if matrix.empty:
        st.info(
            f"No product class reaches the {min_sample}-review minimum in the "
            f"current scope. Lower the Minimum Category Sample Size in the "
            f"sidebar to include smaller classes."
        )
    else:
        st.dataframe(
            style_performance_matrix(
                matrix, baseline_rating, baseline_recommendation, baseline_negative
            ),
            hide_index=True,
            width="stretch",
            height=min(560, 60 + 35 * len(matrix)),
        )
        note(
            f"Performance band: <b>Needs Attention</b> is a negative rate at "
            f"least {analytics.ATTENTION_GAP_PP:.0f} percentage points above the "
            f"{baseline_negative:.1f}% filtered baseline; <b>Strong</b> is at "
            f"least {analytics.AVERAGE_GAP_PP:.0f} points below it with a "
            f"recommendation rate at or above the {baseline_recommendation:.1f}% "
            f"baseline; everything else is <b>Average</b>."
        )
        st.download_button(
            "Download the performance matrix (CSV)",
            data=to_csv_bytes(matrix),
            file_name="product_class_performance_matrix.csv",
            mime="text/csv",
        )

        st.plotly_chart(
            charts.rate_by_segment(
                analytics.segment_performance(
                    filtered, "Class Name", min_reviews=min_sample
                ),
                "Class Name",
                "Negative Rate",
                f"Negative-Review Rate by Product Class, highest 15 "
                f"(baseline {baseline_negative:.1f}%)",
                baseline_negative,
                height=470,
                high_is_good=False,
                top_n=15,
            ),
            width="stretch",
        )

    actionable_insight(
        analytics.product_insights(filtered, min_reviews=min_sample),
        scope_note=f"Product classes with at least {min_sample} reviews "
        f"&middot; baseline negative rate {baseline_negative:.1f}%",
    )


# --------------------------------------------------------------------------- #
# Tab 3 - Customer & Review Insights
# --------------------------------------------------------------------------- #
def tab_customer_insights(
    filtered: pd.DataFrame, min_sample: int, presentation: bool
) -> None:
    kpis = analytics.compute_kpis(filtered)
    baseline_recommendation = kpis["recommendation_rate"]
    baseline_rating = kpis["average_rating"]

    section(
        "Customer Age Profile",
        f"Age is the only customer attribute in the dataset - there is no "
        f"gender, location or tenure field. Average age "
        f"{kpis['average_age']:.2f}, median {kpis['median_age']:.0f}.",
    )
    ages = analytics.age_band_performance(filtered, min_reviews=0)
    row = st.columns([1.25, 1], gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.age_histogram(analytics.age_distribution(filtered)),
            width="stretch",
        )
    with row[1]:
        st.plotly_chart(
            charts.horizontal_bar(
                ages,
                "Age Band",
                "Reviews",
                "Review Volume by Age Band",
                height=380,
                color=charts.VOLUME_COLOR,
                x_title="Reviews",
                preserve_order=True,
            ),
            width="stretch",
        )

    row = st.columns(2, gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.rating_by_segment(
                ages,
                "Age Band",
                baseline_rating,
                f"Average Rating by Age Band (baseline {baseline_rating:.2f})",
                height=370,
                preserve_order=True,
            ),
            width="stretch",
        )
    with row[1]:
        st.plotly_chart(
            charts.rate_by_segment(
                ages,
                "Age Band",
                "Recommendation Rate",
                f"Recommendation Rate by Age Band "
                f"(baseline {baseline_recommendation:.1f}%)",
                baseline_recommendation,
                height=370,
                high_is_good=True,
                preserve_order=True,
            ),
            width="stretch",
        )

    if not presentation:
        with st.expander("Age-band performance table"):
            show_table(ages)

    # ------------------------------------------------------------------ #
    section(
        "Helpful Votes",
        "Positive Feedback Count records how many readers marked a review as "
        "helpful. It measures usefulness, not positive sentiment, so a critical "
        "review can carry the most votes.",
    )
    helpful = analytics.helpful_votes_by_sentiment(filtered)
    row = st.columns(2, gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.helpful_votes_by_sentiment_chart(helpful),
            width="stretch",
        )
    with row[1]:
        st.plotly_chart(
            charts.horizontal_bar(
                analytics.helpful_votes_by_class(filtered, top_n=10),
                "Product Class",
                "Helpful Votes",
                "Helpful Votes by Product Class (top ten)",
                height=360,
                color=styles.TURQUOISE_DARK,
                x_title="Helpful votes",
            ),
            width="stretch",
        )
    if not presentation:
        with st.expander("Helpful votes by sentiment - table"):
            show_table(helpful)

    # ------------------------------------------------------------------ #
    section(
        "Most Helpful Reviews",
        "The original customer comments, sorted and searchable. Review text is "
        "used here only for search and display - no text analysis is applied to "
        "it anywhere in this dashboard.",
    )

    controls = st.columns([1.7, 1, 1, 1.1], gap="medium")
    with controls[0]:
        search = st.text_input(
            "Review keyword search",
            key="review_search",
            placeholder="e.g. runs small, zipper, fabric, too sheer",
            help="Case-insensitive substring match across Review Text and Title.",
        )
    with controls[1]:
        negative_only = st.toggle("Negative reviews only", key="review_negative")
    with controls[2]:
        max_votes = int(
            pd.to_numeric(filtered["Positive Feedback Count"], errors="coerce")
            .fillna(0)
            .max()
            or 0
        )
        helpful_floor = st.number_input(
            "Minimum helpful votes",
            min_value=0,
            max_value=max(max_votes, 1),
            value=0,
            step=1,
            key="review_helpful",
        )
    with controls[3]:
        sort_choice = st.selectbox(
            "Sort by",
            [
                "Helpful votes (high to low)",
                "Rating (low to high)",
                "Rating (high to low)",
                "Review length (long to short)",
                "Age (young to old)",
                "Clothing ID",
            ],
            key="review_sort",
        )

    working = filtered[filtered["Has Review Text"]].copy()
    working = analytics.search_reviews(working, search)
    if negative_only:
        working = working[
            working["Sentiment"].astype("object").astype(str) == "Negative"
        ]
    if helpful_floor > 0:
        working = working[
            pd.to_numeric(working["Positive Feedback Count"], errors="coerce").fillna(0)
            >= helpful_floor
        ]

    sort_map = {
        "Helpful votes (high to low)": ("Positive Feedback Count", False),
        "Rating (low to high)": ("Rating", True),
        "Rating (high to low)": ("Rating", False),
        "Review length (long to short)": ("Review Length Words", False),
        "Age (young to old)": ("Age", True),
        "Clothing ID": ("Clothing ID", True),
    }
    sort_column, ascending = sort_map[sort_choice]
    working = working.sort_values(sort_column, ascending=ascending, kind="stable")

    if working.empty:
        st.info(
            "No reviews with text match this combination of global filters and "
            "review controls."
        )
    else:
        page_controls = st.columns([1, 1, 2.4], gap="medium")
        with page_controls[0]:
            page_size = st.selectbox(
                "Rows to display", [10, 25, 50, 100], index=1, key="review_page_size"
            )
        total_pages = max(1, int(np.ceil(len(working) / page_size)))
        with page_controls[1]:
            page_number = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="review_page",
            )
        start = (int(page_number) - 1) * int(page_size)
        page_frame = working.iloc[start : start + int(page_size)]
        with page_controls[2]:
            st.markdown(
                f'<div class="csid-scope-box" style="margin-top:1.75rem">'
                f"<b>{len(working):,}</b> reviews match &middot; showing rows "
                f"{start + 1:,}-{start + len(page_frame):,} &middot; page "
                f"{int(page_number)} of {total_pages:,}</div>",
                unsafe_allow_html=True,
            )

        display = page_frame[
            [c for c in analytics.REVIEW_DISPLAY_COLUMNS if c in page_frame.columns]
        ].copy()
        for column in ("Sentiment", "Recommendation Label", "Age Band"):
            if column in display.columns:
                display[column] = display[column].astype("object").astype(str)
        display = display.rename(
            columns={
                "Recommendation Label": "Recommendation",
                "Division Name": "Division",
                "Department Name": "Department",
                "Class Name": "Class",
            }
        )
        show_table(
            display,
            height=min(600, 90 + 35 * len(display)),
            column_config={
                "Review Text": st.column_config.TextColumn(
                    "Review Text", width="large"
                ),
                "Title": st.column_config.TextColumn("Title", width="medium"),
                "Rating": st.column_config.NumberColumn("Rating", format="%d"),
                "Positive Feedback Count": st.column_config.NumberColumn(
                    "Helpful Votes", format="%d"
                ),
            },
        )

        download_row = st.columns([1, 1, 2.2], gap="medium")
        with download_row[0]:
            st.download_button(
                "Download displayed reviews (CSV)",
                data=to_csv_bytes(display),
                file_name=f"reviews_page_{int(page_number)}.csv",
                mime="text/csv",
                width="stretch",
            )
        with download_row[1]:
            st.download_button(
                "Download all matching reviews (CSV)",
                data=to_csv_bytes(
                    working[
                        [
                            c
                            for c in analytics.REVIEW_DISPLAY_COLUMNS
                            if c in working.columns
                        ]
                    ]
                ),
                file_name=f"reviews_matching_{len(working)}_rows.csv",
                mime="text/csv",
                width="stretch",
            )

        with st.expander(
            f"Read the {len(page_frame)} displayed reviews in full", expanded=False
        ):
            for _, record in page_frame.iterrows():
                sentiment_value = str(record["Sentiment"])
                tone = {
                    "Positive": "positive",
                    "Neutral": "neutral",
                    "Negative": "negative",
                }.get(sentiment_value, "neutral")
                votes = int(
                    pd.to_numeric(record["Positive Feedback Count"], errors="coerce")
                    or 0
                )
                st.markdown(
                    html(
                        f"""
                        <div class="csid-review csid-review-{tone}">
                          <div class="csid-review-meta">
                            Clothing ID {int(record['Clothing ID'])} &middot;
                            {record['Division Name']} /
                            {record['Department Name']} /
                            {record['Class Name']} &middot;
                            age {int(record['Age'])} ({record['Age Band']})
                            &middot; {int(record['Rating'])} stars &middot;
                            {sentiment_value} &middot;
                            {record['Recommendation Label']} &middot;
                            {votes:,} helpful votes
                          </div>
                          <div class="csid-review-title">{record['Title']}</div>
                          <div class="csid-review-text">{record['Review Text']}</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

    actionable_insight(
        analytics.customer_insights(filtered, min_reviews=min_sample),
        scope_note=f"Age bands with at least {min_sample} reviews "
        f"&middot; {int(kpis['records_with_text']):,} records carry review text",
    )


# --------------------------------------------------------------------------- #
# Tab 4 - Data Quality
# --------------------------------------------------------------------------- #
def tab_data_quality(
    clean: pd.DataFrame, raw: pd.DataFrame, audit: pd.DataFrame,
    source_name: str, schema_report: data_loader.SchemaReport, presentation: bool
) -> None:
    summary = data_cleaning.data_quality_summary(raw, clean)
    kpis = analytics.compute_kpis(clean)

    section(
        "Data Quality Summary",
        "Calculated on the complete dataset, before the global filters are "
        "applied, so the figures describe the source data itself.",
    )

    row = st.columns(4, gap="small")
    with row[0]:
        kpi_card(
            "Total Rows",
            f"{int(summary['total_rows']):,}",
            f"{int(summary['source_columns'])} source columns, "
            f"{int(summary['total_columns'])} after cleaning added the derived "
            f"analytical fields.",
            "Row count of the cleaned dataset. No row is ever deleted by the "
            "cleaning pipeline.",
            "navy",
        )
    with row[1]:
        kpi_card(
            "Overall Completeness",
            f"{summary['completeness_rate']:.2f}%",
            f"{int(summary['missing_cells']):,} missing cells out of "
            f"{int(summary['total_cells']):,} analytical cells.",
            "Non-missing cells divided by all cells in the analytical columns; "
            "the export index column is excluded.",
            "positive" if summary["completeness_rate"] >= 95 else "neutral",
        )
    with row[2]:
        kpi_card(
            "Fully Complete Rows",
            f"{int(summary['complete_rows']):,}",
            f"{summary['complete_row_rate']:.1f}% of rows have no missing value "
            f"in any analytical column.",
            "Rows where every analytical column is populated in the source file.",
            "navy",
        )
    with row[3]:
        clean_checks = (
            summary["invalid_ratings"] == 0 and summary["invalid_recommendations"] == 0
        )
        kpi_card(
            "Domain Validity",
            "Pass" if clean_checks else "Fail",
            f"{int(summary['invalid_ratings']):,} invalid ratings and "
            f"{int(summary['invalid_recommendations']):,} invalid recommendation "
            f"values found.",
            "Rating must be an integer 1-5 and Recommended IND must be 0 or 1.",
            "positive" if clean_checks else "negative",
        )

    row = st.columns(4, gap="small")
    with row[0]:
        kpi_card(
            "Missing Titles",
            f"{int(summary['missing_titles']):,}",
            f"{summary['missing_titles'] / summary['total_rows'] * 100:.1f}% of "
            f"rows. Replaced with '{data_cleaning.MISSING_TITLE}'.",
            "Rows where Title was absent in the source file.",
            "neutral",
        )
    with row[1]:
        kpi_card(
            "Missing Review Texts",
            f"{int(summary['missing_review_texts']):,}",
            f"{summary['missing_review_texts'] / summary['total_rows'] * 100:.1f}% "
            f"of rows. Retained for rating and recommendation analysis.",
            "Rows where Review Text was absent in the source file.",
            "neutral",
        )
    with row[2]:
        kpi_card(
            "Missing Taxonomy Values",
            f"{int(summary['missing_division']):,} / "
            f"{int(summary['missing_department']):,} / "
            f"{int(summary['missing_class']):,}",
            f"Division / Department / Class. All labelled "
            f"'{data_cleaning.UNKNOWN_CATEGORY}'.",
            "Rows where the product taxonomy field was absent in the source "
            "file. The same rows are affected in all three columns.",
            "neutral",
        )
    with row[3]:
        kpi_card(
            "Duplicated Review Texts",
            f"{int(summary['duplicate_review_rows']):,}",
            f"{int(summary['duplicate_review_texts']):,} distinct texts appear "
            f"more than once. Flagged, never deleted.",
            "Rows whose normalised Review Text appears more than once. The "
            "'No Review Text' placeholder is excluded from this check.",
            "neutral" if summary["duplicate_review_rows"] else "positive",
        )

    # ------------------------------------------------------------------ #
    section("Missing Values by Field", "")
    row = st.columns([1.3, 1], gap="medium")
    with row[0]:
        st.plotly_chart(
            charts.missing_values_chart(data_cleaning.missing_values_by_field(raw)),
            width="stretch",
        )
    with row[1]:
        st.plotly_chart(
            charts.completeness_chart(data_cleaning.column_completeness(raw)),
            width="stretch",
        )

    # ------------------------------------------------------------------ #
    section(
        "Data Quality Issues, Impact and Treatment",
        "What is wrong with the data, how much of it, what that does to the "
        "analysis, and exactly how it was handled.",
    )
    quality_report = data_cleaning.data_quality_report(raw, clean)
    show_table(
        quality_report,
        height=60 + 36 * len(quality_report),
        column_config={
            "Analytical Impact": st.column_config.TextColumn(
                "Analytical Impact", width="large"
            ),
            "Treatment Used": st.column_config.TextColumn(
                "Treatment Used", width="large"
            ),
            "Data Quality Issue": st.column_config.TextColumn(
                "Data Quality Issue", width="medium"
            ),
        },
    )

    # ------------------------------------------------------------------ #
    section(
        "Dataset Validation",
        "The dashboard calculates every figure from the file. The published "
        "reference values below are used only to verify those calculations - "
        "they are never displayed as a dashboard result.",
    )
    validation = analytics.validation_summary(clean)
    if validation["all_passed"]:
        st.success(
            f"All {validation['checks']} validation checks match the published "
            f"reference figures for the Women's E-Commerce Clothing Reviews "
            f"dataset, so the loaded file is the expected dataset and the "
            f"cleaning pipeline reproduces its documented totals.",
            icon=":material/verified:",
        )
    else:
        st.warning(
            f"{validation['failed']} of {validation['checks']} checks deviate "
            f"from the reference figures: "
            f"{', '.join(validation['failed_metrics'])}. This is expected when a "
            f"different review export has been uploaded; every metric shown is "
            f"still calculated from the supplied file.",
            icon=":material/warning:",
        )
    row = st.columns([1.4, 1], gap="medium")
    with row[0]:
        show_table(validation["table"], height=560)
    with row[1]:
        show_table(
            pd.DataFrame(
                [
                    {"Property": "Source file", "Value": source_name},
                    {"Property": "Rows", "Value": f"{schema_report.row_count:,}"},
                    {
                        "Property": "Source columns",
                        "Value": str(len(schema_report.columns)),
                    },
                    {
                        "Property": "Matches expected schema",
                        "Value": "Yes" if schema_report.is_expected_schema else "No",
                    },
                    {
                        "Property": "Missing required columns",
                        "Value": ", ".join(schema_report.missing_required) or "None",
                    },
                    {
                        "Property": "Unrecognised columns",
                        "Value": ", ".join(schema_report.unexpected) or "None",
                    },
                ]
            )
        )
        show_table(analytics.numeric_summary(clean))

    if not presentation:
        # ------------------------------------------------------------- #
        section(
            "Cleaning Audit Trail",
            "Every transformation applied to the source file, in order, with the "
            "number of rows it touched and why the rule exists.",
        )
        show_table(
            audit,
            height=480,
            column_config={
                "Rationale": st.column_config.TextColumn("Rationale", width="large"),
                "Action": st.column_config.TextColumn("Action", width="medium"),
            },
        )
        with st.expander("Source-file column profile"):
            show_table(data_loader.profile_dataset(raw))
        with st.expander("Review length by sentiment"):
            st.plotly_chart(charts.review_length_chart(clean), width="stretch")

    # ------------------------------------------------------------------ #
    section("Data Limitations", "")
    row = st.columns(2, gap="medium")
    with row[0]:
        panel(
            "What this dataset cannot support",
            """
            <p>The 11 source columns contain no date, revenue, price, quantity,
            geography, delivery or gender field. This dashboard therefore
            contains <b>no monthly or annual trends, no revenue or cost
            analysis, no ROI calculation, no sales charts, no geographic maps,
            no delivery-time analysis, no gender analysis and no predictive
            forecast</b>. Producing any of them would require inventing data.</p>
            """,
        )
    with row[1]:
        panel(
            "How to read the sentiment figures",
            f"""
            <p>Sentiment is a <b>rating-based business classification</b>
            (1-2 Negative, 3 Neutral, 4-5 Positive), not a predicted or inferred
            score. Review text is used only for search and display.</p>
            <p>The data is also <b>self-selecting</b>: it contains only the
            {int(kpis['total_records']):,} customers who chose to write a review,
            which skews toward stronger opinions. These rates describe the review
            population, not the retailer's entire customer base, and the
            relationships shown are associations rather than proven causes.</p>
            """,
        )


# --------------------------------------------------------------------------- #
# Tab 5 - Simulation & Decision Lab
# --------------------------------------------------------------------------- #
def reset_simulation_controls() -> None:
    for key, value in simulation.SIM_DEFAULTS.items():
        st.session_state[key] = value if key != "sim_categories" else []


def sim_kpi_card(
    label: str,
    value: str,
    note: str = "",
    accent: str = "navy",
    delta: dict[str, Any] | None = None,
) -> None:
    """Baseline or simulated KPI card; simulated cards include delta + estimate tag."""
    delta_html = ""
    tag_html = ""
    if delta is not None:
        delta_html = (
            f'<div class="csid-sim-kpi-delta csid-sim-delta-{delta["accent"]}">'
            f'{delta["arrow"]} {delta["change_text"]} vs baseline</div>'
        )
        tag_html = f'<div class="csid-sim-kpi-tag">{delta["label"]}</div>'
    note_html = f'<div class="csid-sim-kpi-note">{note}</div>' if note else ""
    st.markdown(
        html(
            f"""
            <div class="csid-sim-kpi csid-sim-kpi-accent-{accent}">
              <div class="csid-sim-kpi-label">{label}</div>
              <div class="csid-sim-kpi-value">{value}</div>
              {delta_html}
              {tag_html}
              {note_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_simulation_insights(
    insights: Sequence[simulation.SimulationInsight],
) -> None:
    if not insights:
        return
    items = []
    for insight in insights:
        items.append(
            html(
                f"""
                <div class="csid-actionable-item">
                  <div class="csid-ai-title">{insight.title}</div>
                  <div class="csid-ai-row">
                    <span class="csid-ai-label">What the scenario shows</span>
                    <span class="csid-ai-text">{insight.what}</span>
                  </div>
                  <div class="csid-ai-row">
                    <span class="csid-ai-label">Why it matters</span>
                    <span class="csid-ai-text">{insight.why}</span>
                  </div>
                  <div class="csid-ai-row">
                    <span class="csid-ai-label">Recommended management action</span>
                    <span class="csid-ai-text">{insight.action}</span>
                  </div>
                </div>
                """
            )
        )
    st.markdown(
        html(
            f"""
            <div class="csid-actionable">
              <div class="csid-actionable-head">
                <span class="csid-tag">Scenario Insights</span>
                <span class="csid-scope">Rule-based estimates from selected assumptions</span>
              </div>
              {''.join(items)}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_management_workflow() -> None:
    steps = [
        "Customer Review",
        "Dashboard Detection",
        "Priority Classification",
        "Responsible Department",
        "Corrective Action",
        "Customer Follow-up",
        "KPI Monitoring",
    ]
    parts: list[str] = []
    for index, step in enumerate(steps):
        parts.append(f'<div class="csid-workflow-step">{step}</div>')
        if index < len(steps) - 1:
            parts.append('<div class="csid-workflow-arrow">&rarr;</div>')
    owners = [
        ("Product quality issue", "Product / Quality Management"),
        ("Delivery or service issue", "Customer Service / Operations"),
        ("Sizing or description issue", "E-commerce Content Team"),
        ("High negative-rate category", "Department Manager"),
        ("Recommendation recovery", "CRM / Customer Retention"),
    ]
    owner_cards = "".join(
        f'<div class="csid-owner-card"><b>{issue}</b>{owner}</div>'
        for issue, owner in owners
    )
    st.markdown(
        html(
            f"""
            <div class="csid-workflow">{''.join(parts)}</div>
            <div class="csid-owner-grid">{owner_cards}</div>
            """
        ),
        unsafe_allow_html=True,
    )


def tab_simulation_decision_lab(
    filtered: pd.DataFrame, presentation: bool
) -> None:
    """Business what-if simulation using rating-based sentiment and structured KPIs."""
    for key, value in simulation.SIM_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    # -- Section 1: Purpose ------------------------------------------------ #
    section(
        "Simulation Purpose",
        "Scenario planning for customer-experience interventions — not a forecast.",
    )
    st.markdown(
        html(
            """
            <div class="csid-sim-banner">
              This decision-support simulation estimates how selected
              customer-experience interventions could change sentiment, rating
              and recommendation KPIs. Results are scenario-based estimates
              generated from user-defined assumptions; they are not predictions
              or realized outcomes.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # -- Section 2: Controls ----------------------------------------------- #
    section(
        "Scenario Controls",
        "Adjust intervention assumptions. All results recalculate from the "
        "filtered dataset and the selected scope.",
    )
    control_cols = st.columns([1.35, 1.35, 1.0])
    with control_cols[0]:
        scope = st.radio(
            "Simulation Scope",
            list(simulation.SCOPE_OPTIONS),
            key="sim_scope",
            help="Choose whether the scenario applies to all currently filtered "
            "reviews or only to selected departments, divisions, or product classes.",
        )
    with control_cols[1]:
        category_options = simulation.available_categories(filtered, scope)
        if scope != "All filtered data":
            sanitize_multiselect("sim_categories", category_options)
            categories = st.multiselect(
                "Categories in scope",
                category_options,
                key="sim_categories",
                help="Only categories present in the currently filtered dataset "
                "are listed. Leave empty to include every available category.",
            )
        else:
            categories = []
            st.caption("All filtered reviews are included in the scenario.")
    with control_cols[2]:
        st.write("")
        st.button(
            "Reset Scenario",
            on_click=reset_simulation_controls,
            width="stretch",
            help="Restore every scenario control to its default assumption.",
        )

    slider_cols = st.columns(5)
    with slider_cols[0]:
        neg_reached = st.slider(
            "Negative Reviews Reached (%)",
            0,
            100,
            key="sim_neg_reached",
            help="Share of existing negative reviews that management can contact, "
            "investigate, or address.",
        )
    with slider_cols[1]:
        success_rate = st.slider(
            "Intervention Success Rate (%)",
            0,
            100,
            key="sim_success_rate",
            help="Share of reached negative reviews that are successfully improved.",
        )
    with slider_cols[2]:
        neg_to_pos = st.slider(
            "Successful Negative → Positive (%)",
            0,
            100,
            key="sim_neg_to_pos",
            help="Of successfully improved negative reviews, the share that move "
            "to Positive. The remainder move to Neutral.",
        )
    with slider_cols[3]:
        neu_to_pos = st.slider(
            "Neutral → Positive Conversion (%)",
            0,
            100,
            key="sim_neu_to_pos",
            help="Share of current neutral reviews improved to positive after "
            "actions such as service recovery, better product information, "
            "sizing guidance, or quality correction.",
        )
    with slider_cols[4]:
        advocacy = st.slider(
            "Advocacy Conversion Rate (%)",
            0,
            100,
            key="sim_advocacy",
            help="Share of current Not Recommended cases that become Recommended "
            "after a successful intervention.",
        )

    assumptions = simulation.ScenarioAssumptions(
        negative_reached_pct=float(neg_reached),
        intervention_success_pct=float(success_rate),
        neg_to_pos_of_success_pct=float(neg_to_pos),
        neu_to_pos_pct=float(neu_to_pos),
        advocacy_conversion_pct=float(advocacy),
    )

    scoped = simulation.resolve_scope(filtered, scope, categories)
    scope_label = simulation.scope_description(scope, categories, len(scoped))
    result = simulation.run_simulation(
        scoped, assumptions=assumptions, scope_label=scope_label
    )

    st.markdown(
        html(
            f"""
            <div class="csid-sim-meta">
              <div class="csid-sim-chip"><b>Active filtered reviews:</b>
                {len(filtered):,}</div>
              <div class="csid-sim-chip"><b>Selected scope:</b>
                {html_escape(scope_label)}</div>
              <div class="csid-sim-chip"><b>Baseline period:</b>
                Dataset Baseline</div>
              <div class="csid-sim-chip"><b>Scenario:</b>
                Management Intervention Scenario</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    if scoped.empty:
        st.warning(
            "The selected simulation scope contains no reviews. Adjust the "
            "category selection or global filters to run a scenario.",
            icon=":material/filter_alt_off:",
        )
        return

    chart_height = 360 if presentation else 400
    impact_height = 380 if presentation else 420

    # -- Section 3: KPI cards ---------------------------------------------- #
    section(
        "Baseline vs Simulated KPI Cards",
        "Green marks favourable improvement, red deterioration, amber unchanged.",
    )
    st.markdown(
        '<div class="csid-group-label">Dataset Baseline</div>',
        unsafe_allow_html=True,
    )
    base_cols = st.columns(6)
    with base_cols[0]:
        sim_kpi_card("Total Reviews", f"{result.n:,}", accent="navy")
    with base_cols[1]:
        sim_kpi_card("Average Rating", f"{result.avg_rating:.2f}", accent="turquoise")
    with base_cols[2]:
        sim_kpi_card(
            "Positive Rate",
            f"{result.baseline_positive_rate:.2f}%",
            accent="positive",
        )
    with base_cols[3]:
        sim_kpi_card(
            "Neutral Rate",
            f"{result.baseline_neutral_rate:.2f}%",
            accent="neutral",
        )
    with base_cols[4]:
        sim_kpi_card(
            "Negative Rate",
            f"{result.baseline_negative_rate:.2f}%",
            accent="negative",
        )
    with base_cols[5]:
        sim_kpi_card(
            "Recommendation Rate",
            f"{result.baseline_recommendation_rate:.2f}%",
            accent="navy",
        )

    st.markdown(
        '<div class="csid-group-label">Simulated Scenario</div>',
        unsafe_allow_html=True,
    )
    sim_cols = st.columns(6)
    rating_delta = simulation.kpi_delta(
        result.sim_avg_rating, result.avg_rating, higher_is_better=True, is_percent=False
    )
    pos_delta = simulation.kpi_delta(
        result.sim_positive_rate,
        result.baseline_positive_rate,
        higher_is_better=True,
    )
    neu_delta = simulation.kpi_delta(
        result.sim_neutral_rate,
        result.baseline_neutral_rate,
        higher_is_better=False,
    )
    # Neutral rate movement is context-dependent; treat near-zero change as amber.
    if abs(neu_delta["change"]) < 1e-9:
        neu_delta["accent"] = "neutral"
        neu_delta["arrow"] = "●"
        neu_delta["direction"] = "unchanged"
    neg_delta = simulation.kpi_delta(
        result.sim_negative_rate,
        result.baseline_negative_rate,
        higher_is_better=False,
    )
    reco_delta = simulation.kpi_delta(
        result.sim_recommendation_rate,
        result.baseline_recommendation_rate,
        higher_is_better=True,
    )
    with sim_cols[0]:
        sim_kpi_card(
            "Simulated Average Rating",
            rating_delta["value_text"],
            accent=rating_delta["accent"],
            delta=rating_delta,
        )
    with sim_cols[1]:
        sim_kpi_card(
            "Simulated Positive Rate",
            pos_delta["value_text"],
            accent=pos_delta["accent"],
            delta=pos_delta,
        )
    with sim_cols[2]:
        sim_kpi_card(
            "Simulated Neutral Rate",
            neu_delta["value_text"],
            accent=neu_delta["accent"],
            delta=neu_delta,
        )
    with sim_cols[3]:
        sim_kpi_card(
            "Simulated Negative Rate",
            neg_delta["value_text"],
            accent=neg_delta["accent"],
            delta=neg_delta,
        )
    with sim_cols[4]:
        sim_kpi_card(
            "Simulated Recommendation Rate",
            reco_delta["value_text"],
            accent=reco_delta["accent"],
            delta=reco_delta,
        )
    with sim_cols[5]:
        improved_delta = {
            "arrow": "▲" if result.successfully_improved_reviews else "●",
            "change_text": f"{result.successfully_improved_reviews:,} reviews",
            "accent": "positive" if result.successfully_improved_reviews else "neutral",
            "label": "Scenario Estimate",
        }
        sim_kpi_card(
            "Successfully Improved Reviews",
            f"{result.successfully_improved_reviews:,}",
            note="Successful negatives + neutrals moved to positive",
            accent="turquoise",
            delta=improved_delta,
        )

    # -- Section 4: Visual comparison -------------------------------------- #
    section(
        "Visual Comparison",
        "Charts update instantly when global filters or scenario controls change.",
    )
    dept_table = simulation.department_simulation_table(
        scoped, assumptions=assumptions, min_reviews=0
    )
    impact = simulation.department_impact_chart_data(
        scoped, assumptions=assumptions, min_reviews=50
    )

    row1 = st.columns(2)
    with row1[0]:
        st.plotly_chart(
            charts.simulation_sentiment_comparison(
                simulation.sentiment_comparison_frame(result),
                height=chart_height,
            ),
            width="stretch",
        )
    with row1[1]:
        st.plotly_chart(
            charts.simulation_kpi_improvement(
                simulation.kpi_improvement_frame(result),
                height=chart_height + 20,
            ),
            width="stretch",
        )

    row2 = st.columns(2)
    with row2[0]:
        st.plotly_chart(
            charts.simulation_movement_waterfall(
                simulation.movement_waterfall_frame(result),
                height=chart_height,
            ),
            width="stretch",
        )
    with row2[1]:
        st.plotly_chart(
            charts.simulation_department_impact(impact, height=impact_height),
            width="stretch",
        )

    # -- Section 5: Insights ----------------------------------------------- #
    section(
        "Actionable Simulation Insights",
        "Deterministic, rule-based guidance from the calculated scenario — not an LLM.",
    )
    render_simulation_insights(
        simulation.simulation_insights(result, dept_table)
    )

    # -- Section 6: Workflow ----------------------------------------------- #
    section(
        "Management Action Workflow",
        "From review signal to monitoring — with suggested business owners.",
    )
    render_management_workflow()

    # -- Section 7: Detailed results table --------------------------------- #
    section(
        "Detailed Results Table",
        "One row per department in the simulation scope. Scroll horizontally to "
        "see every column.",
    )
    display_cols = [
        "Department",
        "Reviews",
        "Baseline Average Rating",
        "Simulated Average Rating",
        "Baseline Positive Rate",
        "Simulated Positive Rate",
        "Baseline Negative Rate",
        "Simulated Negative Rate",
        "Baseline Recommendation Rate",
        "Simulated Recommendation Rate",
        "Successful Negative Cases",
        "Remaining Negative Reviews",
        "Scenario Priority",
    ]
    table = (
        dept_table[display_cols]
        if not dept_table.empty
        else pd.DataFrame(columns=display_cols)
    )
    show_table(
        table,
        height=min(520, 56 + 36 * max(len(table), 1)),
        column_config={
            "Department": st.column_config.TextColumn(
                "Department",
                width="medium",
                help="Department remains the leading identity column.",
            ),
            "Scenario Priority": st.column_config.TextColumn(
                "Scenario Priority",
                help="High: remaining simulated negative rate above overall "
                "simulated negative rate AND at least 5% of review volume.",
            ),
        },
    )
    st.download_button(
        "Download Simulation Results (CSV)",
        data=to_csv_bytes(table),
        file_name=f"simulation_results_{result.n}_reviews.csv",
        mime="text/csv",
        width="stretch",
        disabled=table.empty,
    )

    # -- Section 8: Methodology -------------------------------------------- #
    section(
        "Methodology, Assumptions and Limitations",
        "Transparent documentation of how the scenario estimates are produced.",
    )
    with st.expander("Methodology & Formulas", expanded=not presentation):
        st.markdown(simulation.methodology_markdown())
    st.markdown(
        html(
            """
            <div class="csid-disclaimer">
              <b>Disclaimer.</b> Simulation outputs support discussion and
              planning. They must not be presented as actual achieved results
              or statistically validated forecasts.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    if presentation:
        note(
            "Presentation Mode: scenario summary and charts are compact for "
            "screenshot capture. Scenario estimates are not realised outcomes."
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.session_state.setdefault("presentation_mode", False)
    for key, value in FILTER_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    for key, value in simulation.SIM_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    guided_tour.init_tour_state()

    presentation = bool(st.session_state.get("presentation_mode"))
    st.markdown(styles.custom_css(presentation), unsafe_allow_html=True)

    # -- Load ---------------------------------------------------------- #
    try:
        raw, source_name = load_bundled_dataset()
    except data_loader.DatasetError as exc:
        st.error(
            f"The supplied dataset could not be loaded: {exc}\n\n"
            f"Place `{data_loader.DEFAULT_FILENAME}` in the `data/` folder, or "
            f"upload a compatible file from the sidebar."
        )
        st.stop()
        return

    uploaded = st.session_state.get("uploader")
    if uploaded is not None:
        try:
            candidate = load_uploaded_dataset(uploaded.getvalue(), uploaded.name)
            candidate_report = data_loader.describe_schema(candidate)
            if not candidate_report.is_usable:
                st.error(
                    "The uploaded file is missing required columns and cannot "
                    "replace the dataset: "
                    + ", ".join(f"`{c}`" for c in candidate_report.missing_required)
                    + ". The supplied dataset is still in use."
                )
            else:
                raw, source_name = candidate, uploaded.name
        except data_loader.DatasetError as exc:
            st.error(f"The uploaded file could not be read: {exc}")

    schema_report = data_loader.describe_schema(raw)
    clean, audit = prepare_dataset(raw)

    # -- Filters ------------------------------------------------------- #
    filters = build_sidebar(clean, source_name)
    presentation = bool(st.session_state.get("presentation_mode"))
    min_sample = int(filters["min_sample"])

    filtered = analytics.apply_filters(
        clean,
        divisions=filters["divisions"],
        departments=filters["departments"],
        classes=filters["classes"],
        ratings=filters["ratings"],
        sentiments=filters["sentiments"],
        recommendations=filters["recommendations"],
        age_bands=filters["age_bands"],
        has_review_text=filters["has_review_text"],
    )
    render_sidebar_scope(filters, filtered, clean)

    # Tour may open a specific dashboard view before the tab control renders.
    guided_tour.apply_step_tab_navigation()

    # -- Header -------------------------------------------------------- #
    if presentation:
        exit_row = st.columns([1.15, 6])
        with exit_row[0]:
            if st.button("Exit Presentation Mode", width="stretch"):
                st.session_state["presentation_mode"] = False
                st.rerun()

    active_anchor = guided_tour.active_anchor_id()
    guided_tour.tour_anchor("tour-welcome", active_anchor)
    render_hero(source_name, len(filtered), len(clean), presentation=presentation)
    guided_tour.render_tour_panel()

    for message in schema_report.messages():
        st.warning(message, icon=":material/rule:")

    validation = analytics.validation_summary(clean)
    if not validation["all_passed"]:
        st.warning(
            f"**Dataset validation notice.** {validation['failed']} of "
            f"{validation['checks']} checks differ from the published reference "
            f"totals for the Women's E-Commerce Clothing Reviews dataset "
            f"({', '.join(validation['failed_metrics'])}). Every figure shown is "
            f"still calculated from the loaded file - see the **Data Quality** "
            f"tab for the detail.",
            icon=":material/warning:",
        )

    if filtered.empty:
        st.error(
            "The current filter combination returns no reviews, so nothing can "
            "be calculated. Use **Clear All** in the sidebar to reset the "
            "filters.",
            icon=":material/filter_alt_off:",
        )
        render_footer()
        return

    # -- Tabs ---------------------------------------------------------- #
    # A segmented control rather than st.tabs: st.tabs renders the inactive
    # panels at zero width, which leaves their Plotly charts stuck at the
    # library's default 700px and clipped once the tab is opened. Rendering only
    # the selected view also keeps each rerun fast.
    guided_tour.tour_anchor("tour-tabs", active_anchor)
    if active_anchor == "tour-tabs":
        st.markdown(
            '<div class="csid-tour-region csid-tour-region-active">'
            '<span class="csid-tour-badge">Tour focus</span></div>',
            unsafe_allow_html=True,
        )
    selection = st.segmented_control(
        "Dashboard view",
        list(TABS),
        default=st.session_state.get("active_tab", TABS[0]),
        key="tab_selector",
        label_visibility="collapsed",
    )
    active = selection or st.session_state.get("active_tab", TABS[0])
    st.session_state["active_tab"] = active

    if active == TABS[0]:
        tab_executive_overview(filtered, min_sample, presentation)
    elif active == TABS[1]:
        tab_product_performance(filtered, min_sample, presentation)
    elif active == TABS[2]:
        tab_customer_insights(filtered, min_sample, presentation)
    elif active == TABS[3]:
        tab_data_quality(clean, raw, audit, source_name, schema_report, presentation)
    else:
        tab_simulation_decision_lab(filtered, presentation)

    render_footer()


if __name__ == "__main__":
    main()
