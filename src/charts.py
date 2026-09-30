"""Plotly figure builders.

One module owns every chart so colour, typography, axis treatment and label
sizing are identical across all four tabs. Charts are sized for a 1920x1080
presentation surface: no label below 11px, and no chart relies on colour alone
to carry its meaning.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data_cleaning import RATING_ORDER, RECOMMENDATION_ORDER, SENTIMENT_ORDER
from .styles import (
    BORDER,
    BORDER_STRONG,
    NAVY,
    NAVY_TINT,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    RATING_COLORS,
    RECOMMENDATION_COLORS,
    SENTIMENT_COLORS,
    SURFACE,
    TEXT,
    TEXT_MUTED,
    TURQUOISE,
    TURQUOISE_DARK,
    plotly_layout,
)

_EMPTY_MESSAGE = "No data available for the current filter selection"

VOLUME_COLOR = "#AFC0D0"  # muted navy tint used consistently for volume bars


def _empty_figure(message: str = _EMPTY_MESSAGE, height: int = 320) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=TEXT_MUTED),
    )
    figure.update_layout(
        **plotly_layout(height=height),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return figure


# --------------------------------------------------------------------------- #
# Tab 1 - Executive Overview
# --------------------------------------------------------------------------- #
def sentiment_donut(distribution: pd.DataFrame, height: int = 360) -> go.Figure:
    """Sentiment share as a donut, labelled with both count and percentage."""
    if distribution.empty or distribution["Reviews"].sum() == 0:
        return _empty_figure(height=height)

    total = int(distribution["Reviews"].sum())
    figure = go.Figure(
        go.Pie(
            labels=distribution["Sentiment"],
            values=distribution["Reviews"],
            hole=0.58,
            sort=False,
            direction="clockwise",
            marker=dict(
                colors=[SENTIMENT_COLORS[s] for s in distribution["Sentiment"]],
                line=dict(color=SURFACE, width=2),
            ),
            texttemplate="%{label}<br><b>%{value:,}</b><br>%{percent}",
            textposition="outside",
            textfont=dict(size=12.5, color=TEXT),
            hovertemplate="<b>%{label}</b><br>%{value:,} reviews<br>"
            "%{percent} of reviews in scope<extra></extra>",
        )
    )
    figure.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:11px'>reviews</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=22, color=NAVY),
    )
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Sentiment Distribution",
            showlegend=False,
            margin=dict(l=40, r=40, t=52, b=30),
        )
    )
    return figure


def rating_bar(distribution: pd.DataFrame, height: int = 360) -> go.Figure:
    """Ratings 1 to 5 in ascending order - never re-sorted by volume."""
    if distribution.empty or distribution["Reviews"].sum() == 0:
        return _empty_figure(height=height)

    figure = go.Figure(
        go.Bar(
            x=distribution["Rating Label"],
            y=distribution["Reviews"],
            marker=dict(
                color=[RATING_COLORS[int(r)] for r in distribution["Rating"]],
                line=dict(color=BORDER_STRONG, width=0.6),
            ),
            text=[
                f"{int(count):,}<br>{share:.1f}%"
                for count, share in zip(distribution["Reviews"], distribution["Share %"])
            ],
            textposition="outside",
            textfont=dict(size=12, color=TEXT),
            hovertemplate="<b>%{x}</b><br>%{y:,} reviews<extra></extra>",
            width=0.62,
        )
    )
    ceiling = float(distribution["Reviews"].max()) * 1.20
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Rating Distribution (1 to 5 stars)",
            showlegend=False,
            yaxis=dict(title="Reviews", gridcolor=BORDER, range=[0, ceiling],
                       tickformat=","),
            xaxis=dict(
                title="",
                gridcolor="rgba(0,0,0,0)",
                categoryorder="array",
                categoryarray=list(distribution["Rating Label"]),
            ),
        )
    )
    return figure


def recommendation_bar(distribution: pd.DataFrame, height: int = 360) -> go.Figure:
    if distribution.empty or distribution["Reviews"].sum() == 0:
        return _empty_figure(height=height)

    figure = go.Figure(
        go.Bar(
            x=distribution["Reviews"],
            y=distribution["Recommendation"],
            orientation="h",
            marker=dict(
                color=[RECOMMENDATION_COLORS[r] for r in distribution["Recommendation"]],
                line=dict(color=BORDER_STRONG, width=0.6),
            ),
            text=[
                f"{int(count):,}  ({share:.1f}%)"
                for count, share in zip(distribution["Reviews"], distribution["Share %"])
            ],
            textposition="outside",
            textfont=dict(size=12.5, color=TEXT),
            hovertemplate="<b>%{y}</b><br>%{x:,} reviews<extra></extra>",
            width=0.5,
        )
    )
    ceiling = float(distribution["Reviews"].max()) * 1.30
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Recommendation Status",
            showlegend=False,
            xaxis=dict(title="Reviews", gridcolor=BORDER, range=[0, ceiling],
                       tickformat=","),
            yaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=135, r=95, t=52, b=45),
        )
    )
    return figure


def recommendation_by_sentiment_stack(
    long_frame: pd.DataFrame, height: int = 360
) -> go.Figure:
    """Recommendation behaviour inside each sentiment group (100% stacked)."""
    if long_frame.empty or long_frame["Reviews"].sum() == 0:
        return _empty_figure(height=height)

    figure = go.Figure()
    for recommendation in RECOMMENDATION_ORDER:
        subset = (
            long_frame[long_frame["Recommendation"] == recommendation]
            .set_index("Sentiment")
            .reindex(SENTIMENT_ORDER)
        )
        figure.add_trace(
            go.Bar(
                name=recommendation,
                x=list(SENTIMENT_ORDER),
                y=subset["Share %"].fillna(0),
                marker=dict(color=RECOMMENDATION_COLORS[recommendation]),
                customdata=np.stack([subset["Reviews"].fillna(0)], axis=-1),
                text=[
                    f"{share:.1f}%" if share >= 6 else ""
                    for share in subset["Share %"].fillna(0)
                ],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=12, color="#FFFFFF"),
                hovertemplate="<b>%{x}</b><br>"
                + recommendation
                + ": %{y:.1f}% (%{customdata[0]:,} reviews)<extra></extra>",
                width=0.55,
            )
        )
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Recommendation Behaviour within each Sentiment Group "
                  "(100% stacked)",
            barmode="stack",
            xaxis=dict(title="Sentiment group", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Share of the group (%)", range=[0, 100],
                       ticksuffix="%", gridcolor=BORDER),
            margin=dict(l=65, r=30, t=72, b=52),
        )
    )
    return figure


def segment_overview(
    table: pd.DataFrame,
    dimension: str,
    title: str,
    height: int = 420,
) -> go.Figure:
    """Volume bars with rating and rate lines on a secondary axis."""
    if table.empty:
        return _empty_figure(height=height)

    labels = table[dimension].astype(str).tolist()
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Review volume",
            x=labels,
            y=table["Reviews"],
            marker=dict(color=VOLUME_COLOR, line=dict(color=BORDER_STRONG, width=0.6)),
            text=[f"{int(v):,}" for v in table["Reviews"]],
            textposition="outside",
            textfont=dict(size=11.5, color=TEXT_MUTED),
            hovertemplate="<b>%{x}</b><br>%{y:,} reviews<extra></extra>",
            width=0.55,
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Recommendation rate",
            x=labels,
            y=table["Recommendation Rate"],
            mode="lines+markers+text",
            line=dict(color=POSITIVE, width=2.6),
            marker=dict(size=9, color=POSITIVE, line=dict(color=SURFACE, width=1.5)),
            text=[f"{v:.1f}%" for v in table["Recommendation Rate"]],
            textposition="top center",
            textfont=dict(size=11.5, color=POSITIVE),
            hovertemplate="<b>%{x}</b><br>Recommendation rate %{y:.1f}%<extra></extra>",
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Negative-review rate",
            x=labels,
            y=table["Negative Rate"],
            mode="lines+markers+text",
            line=dict(color=NEGATIVE, width=2.6, dash="dot"),
            marker=dict(size=9, color=NEGATIVE, symbol="diamond",
                        line=dict(color=SURFACE, width=1.5)),
            text=[f"{v:.1f}%" for v in table["Negative Rate"]],
            textposition="bottom center",
            textfont=dict(size=11.5, color=NEGATIVE),
            hovertemplate="<b>%{x}</b><br>Negative rate %{y:.1f}%<extra></extra>",
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Average rating (of 5)",
            x=labels,
            y=table["Avg Rating"] * 20,
            mode="markers+text",
            marker=dict(size=11, color=NEUTRAL, symbol="square",
                        line=dict(color=SURFACE, width=1.5)),
            text=[f"{v:.2f}" for v in table["Avg Rating"]],
            textposition="middle right",
            textfont=dict(size=11.5, color="#9A6A12"),
            hovertemplate="<b>%{x}</b><br>Average rating %{text} of 5<extra></extra>",
            yaxis="y2",
        )
    )
    figure.update_layout(
        **plotly_layout(
            height=height,
            title=title,
            yaxis=dict(title="Review volume", gridcolor=BORDER, tickformat=","),
            yaxis2=dict(
                title="Rate (%) &middot; rating on a 0-100 scale",
                overlaying="y",
                side="right",
                range=[0, 105],
                ticksuffix="%",
                gridcolor="rgba(0,0,0,0)",
                showline=True,
                linecolor=BORDER_STRONG,
                title_font=dict(size=11.5, color=TEXT_MUTED),
            ),
            xaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
            # Legend sits below the plot so it can never collide with the title
            # in a narrow column.
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.14,
                xanchor="left",
                x=0.0,
                font=dict(size=11.5),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=70, r=95, t=58, b=95),
        )
    )
    return figure


# --------------------------------------------------------------------------- #
# Tab 2 - Product Performance
# --------------------------------------------------------------------------- #
def _baseline_colors(
    values: Sequence[float], baseline: float, high_is_good: bool
) -> list[str]:
    """Teal / amber / coral against a baseline, using symmetric thresholds."""
    out = []
    for value in values:
        gap = float(value) - baseline
        if not high_is_good:
            gap = -gap
        if gap >= 0.02 * max(abs(baseline), 1.0):
            out.append(POSITIVE)
        elif gap <= -0.05 * max(abs(baseline), 1.0):
            out.append(NEGATIVE)
        else:
            out.append(NEUTRAL)
    return out


def horizontal_bar(
    table: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    height: int = 400,
    color: str | Sequence[str] = NAVY,
    value_suffix: str = "",
    value_format: str = ",.0f",
    x_title: str = "",
    baseline: float | None = None,
    baseline_label: str | None = None,
    x_range: Sequence[float] | None = None,
    preserve_order: bool = False,
) -> go.Figure:
    """Sorted horizontal bar chart - the workhorse comparison chart.

    When ``baseline`` is given, a dashed reference line is drawn. The baseline
    value belongs in the chart title rather than in an in-plot annotation: at
    three-charts-per-row widths an annotation always lands on top of a bar.

    ``preserve_order`` keeps the caller's row order instead of sorting by value,
    which matters for an ordinal dimension such as the age bands.
    """
    if table.empty:
        return _empty_figure(height=height)

    if preserve_order:
        # Plotly draws the first category at the bottom, so reverse the frame to
        # read top-to-bottom in the caller's order.
        data = table.iloc[::-1]
    else:
        data = table.sort_values(value_column, ascending=True)
    values = data[value_column].astype(float)
    figure = go.Figure(
        go.Bar(
            x=values,
            y=data[label_column].astype(str),
            orientation="h",
            marker=dict(
                color=list(color) if isinstance(color, (list, tuple)) else color,
                line=dict(color=BORDER_STRONG, width=0.5),
            ),
            text=[format(v, value_format) + value_suffix for v in values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=11.5, color=TEXT),
            hovertemplate="<b>%{y}</b><br>%{x:,.2f}" + value_suffix + "<extra></extra>",
        )
    )
    if x_range is not None:
        axis_range = list(x_range)
    else:
        top = float(values.max()) if len(values) and values.max() > 0 else 1.0
        axis_range = [0, top * 1.25]
    if baseline is not None:
        axis_range[1] = max(axis_range[1], baseline * 1.15)
        # Light dashed line: value labels sit near the baseline by definition, so
        # a heavy line would cut straight through the digits.
        figure.add_vline(
            x=baseline,
            line=dict(color=NAVY_TINT, width=1.4, dash="dash"),
            annotation_text=baseline_label or "Baseline",
            annotation_position="top right",
            annotation_font=dict(size=11.5, color=NAVY),
        )
    figure.update_layout(
        **plotly_layout(
            height=height,
            title=title,
            showlegend=False,
            xaxis=dict(title=x_title, gridcolor=BORDER, range=axis_range),
            yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
            margin=dict(l=20, r=70, t=58, b=48),
        )
    )
    return figure


def rating_by_segment(
    table: pd.DataFrame,
    label_column: str,
    baseline: float,
    title: str,
    height: int = 390,
    preserve_order: bool = False,
) -> go.Figure:
    """Average rating per segment, coloured against the filtered baseline."""
    if table.empty:
        return _empty_figure(height=height)
    data = table if preserve_order else table.sort_values("Avg Rating", ascending=True)
    colors = _baseline_colors(data["Avg Rating"], baseline, high_is_good=True)
    if preserve_order:
        colors = colors[::-1]  # horizontal_bar reverses the frame
    return horizontal_bar(
        data,
        label_column,
        "Avg Rating",
        title,
        height=height,
        color=colors,
        value_format=".2f",
        x_title="Average rating (of 5)",
        baseline=baseline,
        x_range=[0, 5.75],
        preserve_order=preserve_order,
    )


def rate_by_segment(
    table: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    baseline: float,
    height: int = 400,
    high_is_good: bool = True,
    top_n: int | None = None,
    preserve_order: bool = False,
) -> go.Figure:
    """Rate per segment, coloured against the filtered baseline.

    The dashed line marks the baseline; its value belongs in the chart title,
    because an in-plot annotation collides with the bars at this width.
    """
    if table.empty:
        return _empty_figure(height=height)

    if preserve_order:
        data = table.iloc[::-1]
    else:
        data = table.sort_values(value_column, ascending=False)
        if top_n:
            data = data.head(top_n)
        data = data.sort_values(value_column, ascending=True)

    def color_for(value: float) -> str:
        gap = value - baseline
        if not high_is_good:
            gap = -gap
        if gap >= 2:
            return POSITIVE
        if gap <= -5:
            return NEGATIVE
        return NEUTRAL

    colors = [color_for(float(v)) for v in data[value_column]]
    figure = go.Figure(
        go.Bar(
            x=data[value_column],
            y=data[label_column].astype(str),
            orientation="h",
            marker=dict(color=colors, line=dict(color=BORDER_STRONG, width=0.5)),
            customdata=np.stack(
                [data["Reviews"], data["Negative Reviews"], data["Avg Rating"]], axis=-1
            ),
            text=[f"{v:.1f}%" for v in data[value_column]],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=11.5, color=TEXT),
            hovertemplate="<b>%{y}</b><br>"
            + value_column
            + " %{x:.1f}%<br>%{customdata[0]:,} reviews &middot; "
            "%{customdata[1]:,} negative<br>Average rating "
            "%{customdata[2]:.2f}<extra></extra>",
        )
    )
    ceiling = max(float(data[value_column].max()) * 1.22, baseline * 1.20, 5.0)
    figure.add_vline(x=baseline, line=dict(color=NAVY_TINT, width=1.4, dash="dash"))
    figure.update_layout(
        **plotly_layout(
            height=height,
            title=title,
            showlegend=False,
            xaxis=dict(title=f"{value_column} (%)", gridcolor=BORDER,
                       ticksuffix="%", range=[0, ceiling]),
            yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
            margin=dict(l=20, r=70, t=58, b=52),
        )
    )
    return figure


def grouped_rate_chart(
    table: pd.DataFrame,
    label_column: str,
    title: str,
    height: int = 400,
    metrics: Sequence[tuple[str, str]] = (
        ("Recommendation Rate", POSITIVE),
        ("Negative Rate", NEGATIVE),
    ),
) -> go.Figure:
    if table.empty:
        return _empty_figure(height=height)

    figure = go.Figure()
    for metric, color in metrics:
        if metric not in table.columns:
            continue
        figure.add_trace(
            go.Bar(
                name=metric,
                x=table[label_column].astype(str),
                y=table[metric],
                marker=dict(color=color, line=dict(color=BORDER_STRONG, width=0.5)),
                text=[f"{v:.1f}%" for v in table[metric]],
                textposition="outside",
                textfont=dict(size=11.5, color=TEXT),
                hovertemplate="<b>%{x}</b><br>" + metric + " %{y:.1f}%<extra></extra>",
            )
        )
    figure.update_layout(
        **plotly_layout(
            height=height,
            title=title,
            barmode="group",
            bargap=0.30,
            bargroupgap=0.08,
            xaxis=dict(title="", gridcolor="rgba(0,0,0,0)", tickangle=-30,
                       automargin=True),
            yaxis=dict(title="Rate (%)", gridcolor=BORDER, ticksuffix="%",
                       range=[0, 112]),
            margin=dict(l=65, r=30, t=72, b=95),
        )
    )
    return figure


# --------------------------------------------------------------------------- #
# Tab 3 - Customer & Review Insights
# --------------------------------------------------------------------------- #
def age_histogram(ages: pd.Series, height: int = 380) -> go.Figure:
    """Customer age distribution with the mean and median marked."""
    if ages.empty:
        return _empty_figure(height=height)

    figure = go.Figure(
        go.Histogram(
            x=ages,
            xbins=dict(start=float(ages.min()), end=float(ages.max()) + 5, size=5),
            marker=dict(color=NAVY_TINT, line=dict(color=SURFACE, width=1)),
            hovertemplate="Age %{x}<br>%{y:,} reviewers<extra></extra>",
        )
    )
    mean_age = float(ages.mean())
    median_age = float(ages.median())
    figure.add_vline(
        x=mean_age,
        line=dict(color=TURQUOISE_DARK, width=2),
        annotation_text=f"Mean {mean_age:.1f}",
        annotation_position="top right",
        annotation_font=dict(size=11.5, color=TURQUOISE_DARK),
    )
    figure.add_vline(
        x=median_age,
        line=dict(color=NAVY, width=2, dash="dash"),
        annotation_text=f"Median {median_age:.0f}",
        annotation_position="top left",
        annotation_font=dict(size=11.5, color=NAVY),
    )
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Customer Age Distribution (5-year bins)",
            showlegend=False,
            bargap=0.06,
            xaxis=dict(title="Customer age", gridcolor=BORDER),
            yaxis=dict(title="Reviews", gridcolor=BORDER, tickformat=","),
        )
    )
    return figure


def helpful_votes_by_sentiment_chart(
    table: pd.DataFrame, height: int = 360
) -> go.Figure:
    """Total helpful votes per sentiment class, annotated with the per-review rate.

    A single axis on purpose: the sentiment class with the most votes in total is
    also the one with the fewest votes per review, so a second axis puts the two
    series on a collision course and hides one of the labels behind a bar.
    """
    if table.empty:
        return _empty_figure(height=height)

    figure = go.Figure(
        go.Bar(
            x=table["Sentiment"],
            y=table["Total Helpful Votes"],
            marker=dict(
                color=[SENTIMENT_COLORS.get(s, NAVY) for s in table["Sentiment"]],
                line=dict(color=BORDER_STRONG, width=0.6),
            ),
            customdata=np.stack([table["Reviews"], table["Avg Helpful Votes"]], axis=-1),
            text=[
                f"<b>{int(total):,}</b> votes<br>"
                f"<span style='font-size:11.5px'>{average:.2f} per review</span>"
                for total, average in zip(
                    table["Total Helpful Votes"], table["Avg Helpful Votes"]
                )
            ],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=12.5, color=TEXT),
            hovertemplate="<b>%{x}</b><br>%{y:,} helpful votes in total<br>"
            "%{customdata[0]:,} reviews &middot; %{customdata[1]:.2f} votes per "
            "review<extra></extra>",
            width=0.52,
        )
    )
    ceiling = float(table["Total Helpful Votes"].max()) * 1.32 or 1
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Helpful Votes by Sentiment (total, and votes per review)",
            showlegend=False,
            xaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Total helpful votes", gridcolor=BORDER,
                       range=[0, ceiling], tickformat=","),
            margin=dict(l=70, r=30, t=58, b=48),
        )
    )
    return figure


# --------------------------------------------------------------------------- #
# Tab 4 - Data Quality
# --------------------------------------------------------------------------- #
def missing_values_chart(table: pd.DataFrame, height: int = 400) -> go.Figure:
    """Missing values by field, in absolute counts with the share on hover."""
    if table.empty:
        return _empty_figure(height=height)

    data = table.sort_values("Missing", ascending=True)
    colors = [
        POSITIVE if v == 0 else NEUTRAL if v / max(1, data["Rows"].max()) < 0.10 else NEGATIVE
        for v in data["Missing"]
    ]
    figure = go.Figure(
        go.Bar(
            x=data["Missing"],
            y=data["Field"].astype(str),
            orientation="h",
            marker=dict(color=colors, line=dict(color=BORDER_STRONG, width=0.5)),
            customdata=np.stack([data["Missing %"]], axis=-1),
            text=[
                f"{int(v):,}" + (f"  ({p:.1f}%)" if v else "")
                for v, p in zip(data["Missing"], data["Missing %"])
            ],
            textposition="outside",
            textfont=dict(size=11.5, color=TEXT),
            hovertemplate="<b>%{y}</b><br>%{x:,} missing values<br>"
            "%{customdata[0]:.2f}% of rows<extra></extra>",
        )
    )
    ceiling = float(data["Missing"].max()) * 1.30 or 1
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Missing Values by Field (source file)",
            showlegend=False,
            xaxis=dict(title="Missing values", gridcolor=BORDER, range=[0, ceiling],
                       tickformat=","),
            yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
            margin=dict(l=20, r=95, t=52, b=48),
        )
    )
    return figure


def completeness_chart(table: pd.DataFrame, height: int = 400) -> go.Figure:
    if table.empty:
        return _empty_figure(height=height)
    data = table.sort_values("Completeness %", ascending=True)
    colors = [
        POSITIVE if v >= 99 else NEUTRAL if v >= 90 else NEGATIVE
        for v in data["Completeness %"]
    ]
    figure = go.Figure(
        go.Bar(
            x=data["Completeness %"],
            y=data["Column"].astype(str),
            orientation="h",
            marker=dict(color=colors, line=dict(color=BORDER_STRONG, width=0.5)),
            customdata=np.stack([data["Missing"]], axis=-1),
            text=[f"{v:.1f}%" for v in data["Completeness %"]],
            textposition="outside",
            textfont=dict(size=11.5, color=TEXT),
            hovertemplate="<b>%{y}</b><br>%{x:.2f}% complete<br>"
            "%{customdata[0]:,} missing values<extra></extra>",
        )
    )
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Column Completeness in the Source Dataset",
            showlegend=False,
            xaxis=dict(title="Completeness (%)", gridcolor=BORDER, range=[0, 112],
                       ticksuffix="%"),
            yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
            margin=dict(l=20, r=65, t=52, b=48),
        )
    )
    return figure


def review_length_chart(df: pd.DataFrame, height: int = 380) -> go.Figure:
    """Review-length distribution per sentiment class."""
    if df.empty:
        return _empty_figure(height=height)
    figure = go.Figure()
    sentiment = df["Sentiment"].astype("object").astype(str)
    for label in SENTIMENT_ORDER:
        values = df.loc[
            (sentiment == label) & df["Has Review Text"], "Review Length Words"
        ]
        if values.empty:
            continue
        figure.add_trace(
            go.Box(
                name=label,
                y=values,
                marker=dict(color=SENTIMENT_COLORS[label]),
                line=dict(color=SENTIMENT_COLORS[label], width=1.8),
                fillcolor="rgba(0,0,0,0)",
                boxmean=True,
                boxpoints=False,
                hovertemplate="<b>" + label + "</b><br>Median %{median:.0f} words"
                "<extra></extra>",
                width=0.45,
            )
        )
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Review Length by Sentiment (words; rating-only rows excluded)",
            showlegend=False,
            xaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Words per review", gridcolor=BORDER),
        )
    )
    return figure


# --------------------------------------------------------------------------- #
# Tab 5 - Simulation & Decision Lab
# --------------------------------------------------------------------------- #
SERIES_COLORS = {
    "Dataset Baseline": NAVY_TINT,
    "Simulated Scenario": TURQUOISE,
}


def simulation_sentiment_comparison(
    comparison: pd.DataFrame, height: int = 400
) -> go.Figure:
    """Grouped bars: baseline vs simulated sentiment counts with % labels."""
    if comparison.empty or comparison["Reviews"].sum() == 0:
        return _empty_figure(height=height)

    figure = go.Figure()
    for series in ("Dataset Baseline", "Simulated Scenario"):
        subset = comparison[comparison["Series"] == series]
        if subset.empty:
            continue
        ordered = subset.set_index("Sentiment").reindex(list(SENTIMENT_ORDER)).dropna(
            how="all"
        )
        figure.add_trace(
            go.Bar(
                name=series,
                x=list(ordered.index),
                y=ordered["Reviews"],
                marker=dict(
                    color=SERIES_COLORS.get(series, NAVY),
                    line=dict(color=BORDER_STRONG, width=0.5),
                ),
                text=[
                    f"{int(c):,}<br>{s:.1f}%"
                    for c, s in zip(ordered["Reviews"], ordered["Share %"])
                ],
                textposition="outside",
                textfont=dict(size=11.5, color=TEXT),
                cliponaxis=False,
                hovertemplate=(
                    f"<b>{series}</b><br>%{{x}}<br>%{{y:,}} reviews<extra></extra>"
                ),
            )
        )

    ceiling = float(comparison["Reviews"].max()) * 1.28 if len(comparison) else 1.0
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Baseline vs Simulated Sentiment",
            barmode="group",
            bargap=0.28,
            bargroupgap=0.08,
            showlegend=True,
            yaxis=dict(
                title="Reviews",
                gridcolor=BORDER,
                range=[0, ceiling],
                tickformat=",",
            ),
            xaxis=dict(
                title="",
                gridcolor="rgba(0,0,0,0)",
                categoryorder="array",
                categoryarray=list(SENTIMENT_ORDER),
            ),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)),
            margin=dict(l=50, r=24, t=70, b=40),
        )
    )
    return figure


def simulation_kpi_improvement(
    metrics: pd.DataFrame, height: int = 420
) -> go.Figure:
    """Side-by-side subplots so rating (1-5) is not mixed with percentage axes."""
    from plotly.subplots import make_subplots

    if metrics.empty:
        return _empty_figure(height=height)

    rating = metrics[metrics["Kind"] == "rating"]
    percents = metrics[metrics["Kind"] == "percent"]

    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.16,
        subplot_titles=(
            "Average Rating (1–5 scale)",
            "Rate metrics (%)",
        ),
    )

    def _add_group(frame: pd.DataFrame, col: int, value_suffix: str) -> None:
        if frame.empty:
            return
        labels = frame["Metric"].tolist()
        figure.add_trace(
            go.Bar(
                name="Dataset Baseline",
                y=labels,
                x=frame["Baseline"],
                orientation="h",
                marker=dict(color=NAVY_TINT),
                text=[f"{v:.2f}{value_suffix}" for v in frame["Baseline"]],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=11, color=TEXT),
                hovertemplate="<b>%{y}</b><br>Baseline: %{x:.2f}"
                + value_suffix
                + "<extra></extra>",
                showlegend=(col == 1),
                legendgroup="baseline",
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Bar(
                name="Simulated Scenario",
                y=labels,
                x=frame["Simulated"],
                orientation="h",
                marker=dict(color=TURQUOISE),
                text=[f"{v:.2f}{value_suffix}" for v in frame["Simulated"]],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=11, color=TEXT),
                hovertemplate="<b>%{y}</b><br>Simulated: %{x:.2f}"
                + value_suffix
                + "<extra></extra>",
                showlegend=(col == 1),
                legendgroup="simulated",
            ),
            row=1,
            col=col,
        )

    _add_group(rating, 1, "")
    _add_group(percents, 2, "%")

    figure.update_layout(
        **plotly_layout(
            height=height,
            title="KPI Improvement Comparison",
            barmode="group",
            showlegend=True,
            legend=dict(orientation="h", y=1.14, x=0, font=dict(size=12)),
            margin=dict(l=20, r=70, t=90, b=40),
        )
    )
    figure.update_xaxes(
        title_text="Rating",
        gridcolor=BORDER,
        range=[0, 5.5],
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title_text="Percent",
        gridcolor=BORDER,
        ticksuffix="%",
        range=[0, 110],
        row=1,
        col=2,
    )
    figure.update_yaxes(automargin=True, gridcolor="rgba(0,0,0,0)")
    figure.update_annotations(font=dict(size=12.5, color=NAVY))
    return figure


def simulation_movement_waterfall(
    movements: pd.DataFrame, height: int = 380
) -> go.Figure:
    """Horizontal bars showing review-movement volumes under the scenario."""
    if movements.empty or int(movements["Count"].sum()) == 0:
        # Still show the structure with zeros rather than a blank empty state when
        # the frame exists but all movements are zero (zero-intervention case).
        if movements.empty:
            return _empty_figure(height=height)

    colors = [TURQUOISE, NEUTRAL, POSITIVE, TURQUOISE_DARK]
    data = movements.iloc[::-1]
    figure = go.Figure(
        go.Bar(
            y=data["Step"],
            x=data["Count"],
            orientation="h",
            marker=dict(
                color=colors[: len(data)][::-1]
                if len(data) <= len(colors)
                else TURQUOISE,
                line=dict(color=BORDER_STRONG, width=0.5),
            ),
            text=[f"{int(v):,}" for v in data["Count"]],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=13, color=TEXT),
            hovertemplate="<b>%{y}</b><br>%{x:,} reviews<extra></extra>",
        )
    )
    top = float(data["Count"].max()) if len(data) and data["Count"].max() > 0 else 1.0
    figure.update_layout(
        **plotly_layout(
            height=height,
            title="Review Movement Waterfall",
            showlegend=False,
            xaxis=dict(
                title="Reviews moved (scenario estimate)",
                gridcolor=BORDER,
                range=[0, top * 1.25],
                tickformat=",",
            ),
            yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
            margin=dict(l=20, r=70, t=58, b=48),
        )
    )
    return figure


def simulation_department_impact(
    impact: pd.DataFrame, height: int = 420
) -> go.Figure:
    """Ranked departments by expected reduction in negative-review count."""
    if impact.empty:
        return _empty_figure(
            "No department meets the minimum sample for this chart",
            height=height,
        )

    data = impact.sort_values("Negative Reduction", ascending=True)
    labels = [
        (
            f"{row['Department']}  |  "
            f"neg {row['Baseline Neg Rate']:.1f}% → {row['Simulated Neg Rate']:.1f}%  |  "
            f"improved {int(row['Negative Reduction']):,}"
        )
        for _, row in data.iterrows()
    ]
    figure = go.Figure(
        go.Bar(
            y=labels,
            x=data["Negative Reduction"],
            orientation="h",
            marker=dict(color=TURQUOISE, line=dict(color=BORDER_STRONG, width=0.5)),
            text=[f"{int(v):,}" for v in data["Negative Reduction"]],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=11.5, color=TEXT),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Negative reduction: %{x:,}<br>"
                "Baseline neg rate: %{customdata[1]:.1f}%<br>"
                "Simulated neg rate: %{customdata[2]:.1f}%<br>"
                "Improved cases: %{customdata[3]:,}<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    data["Department"].astype(str),
                    data["Baseline Neg Rate"],
                    data["Simulated Neg Rate"],
                    data["Improved Cases"]
                    if "Improved Cases" in data.columns
                    else data["Negative Reduction"],
                ]
            ),
        )
    )
    top = (
        float(data["Negative Reduction"].max())
        if len(data) and data["Negative Reduction"].max() > 0
        else 1.0
    )
    figure.update_layout(
        **plotly_layout(
            height=max(height, 48 * len(data) + 100),
            title="Department Scenario Impact (negative-count reduction)",
            showlegend=False,
            xaxis=dict(
                title="Successfully improved negative reviews",
                gridcolor=BORDER,
                range=[0, top * 1.25],
                tickformat=",",
            ),
            yaxis=dict(title="", gridcolor="rgba(0,0,0,0)", automargin=True),
            margin=dict(l=20, r=70, t=58, b=48),
        )
    )
    return figure
