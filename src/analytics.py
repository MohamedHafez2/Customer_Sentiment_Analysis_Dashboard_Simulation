"""Deterministic analytics for the Customer Sentiment Intelligence Dashboard.

Every number rendered in the UI is produced here from the supplied dataset by
plain aggregation - counts, means, rates and rankings. There is no model, no
prediction and no randomness: the same filter selection always yields the same
figures and the same insight wording.

The reference figures in :data:`EXPECTED_VALIDATION` exist only to *verify* the
calculations; they are never used as an output value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .data_cleaning import (
    AGE_BAND_ORDER,
    RATING_ORDER,
    RECOMMENDATION_ORDER,
    SENTIMENT_ORDER,
    UNKNOWN_CATEGORY,
)

# --------------------------------------------------------------------------- #
# Validation reference (verification only - never displayed as a result)
# --------------------------------------------------------------------------- #
EXPECTED_VALIDATION: dict[str, dict[str, Any]] = {
    "total_records": {"label": "Total Reviews", "expected": 23_486, "tolerance": 0, "kind": "count"},
    "records_with_text": {"label": "Reviews With Text", "expected": 22_641, "tolerance": 0, "kind": "count"},
    "missing_review_text": {"label": "Missing Review Text", "expected": 845, "tolerance": 0, "kind": "count"},
    "missing_title": {"label": "Missing Titles", "expected": 3_810, "tolerance": 0, "kind": "count"},
    "average_rating": {"label": "Average Rating", "expected": 4.196, "tolerance": 0.01, "kind": "rating"},
    "recommendation_rate": {"label": "Recommendation Rate", "expected": 82.24, "tolerance": 0.15, "kind": "percent"},
    "positive_reviews": {"label": "Positive Reviews", "expected": 18_208, "tolerance": 0, "kind": "count"},
    "positive_rate": {"label": "Positive Rate", "expected": 77.53, "tolerance": 0.15, "kind": "percent"},
    "neutral_reviews": {"label": "Neutral Reviews", "expected": 2_871, "tolerance": 0, "kind": "count"},
    "neutral_rate": {"label": "Neutral Rate", "expected": 12.22, "tolerance": 0.15, "kind": "percent"},
    "negative_reviews": {"label": "Negative Reviews", "expected": 2_407, "tolerance": 0, "kind": "count"},
    "negative_rate": {"label": "Negative Rate", "expected": 10.25, "tolerance": 0.15, "kind": "percent"},
    "total_positive_feedback": {"label": "Total Positive Feedback Count", "expected": 59_559, "tolerance": 0, "kind": "count"},
    "average_age": {"label": "Average Age", "expected": 43.20, "tolerance": 0.1, "kind": "rating"},
    "median_age": {"label": "Median Age", "expected": 41, "tolerance": 0, "kind": "count"},
}

#: Thresholds (percentage points vs the filtered baseline) for the performance
#: bands used in conditional formatting. Documented so the colours are auditable.
ATTENTION_GAP_PP = 5.0
AVERAGE_GAP_PP = 2.0

PERFORMANCE_ORDER = ("Strong", "Average", "Needs Attention")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _rate(numerator: float, denominator: float) -> float:
    """Percentage with a zero-safe denominator."""
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def _as_str(series: pd.Series) -> pd.Series:
    """Categorical-safe string view used for every group-by."""
    return series.astype("object").astype(str)


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else 0.0


# --------------------------------------------------------------------------- #
# Headline KPIs
# --------------------------------------------------------------------------- #
KPI_KEYS = (
    "total_records", "records_with_text", "missing_review_text", "missing_title",
    "text_coverage_rate", "average_rating", "median_rating",
    "recommendation_rate", "recommended_count", "not_recommended_count",
    "positive_reviews", "positive_rate", "neutral_reviews", "neutral_rate",
    "negative_reviews", "negative_rate", "total_positive_feedback",
    "average_positive_feedback", "average_age", "median_age",
    "average_review_length", "duplicate_review_rows", "unique_clothing_ids",
)


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Headline KPIs for the currently filtered frame."""
    total = int(len(df))
    if total == 0:
        return {key: 0.0 for key in KPI_KEYS}

    sentiment = _as_str(df["Sentiment"])
    positive = int((sentiment == "Positive").sum())
    neutral = int((sentiment == "Neutral").sum())
    negative = int((sentiment == "Negative").sum())
    classified = positive + neutral + negative

    valid_reco = (
        df["Valid Recommendation"]
        if "Valid Recommendation" in df.columns
        else pd.Series(True, index=df.index)
    )
    reco = pd.to_numeric(df.loc[valid_reco, "Recommended IND"], errors="coerce").dropna()
    recommended = int((reco == 1).sum())
    not_recommended = int((reco == 0).sum())

    with_text = int(df["Has Review Text"].sum())
    with_title = int(df["Has Title"].sum()) if "Has Title" in df.columns else total

    return {
        "total_records": total,
        "records_with_text": with_text,
        "missing_review_text": total - with_text,
        "missing_title": total - with_title,
        "text_coverage_rate": _rate(with_text, total),
        "average_rating": _safe_mean(df["Rating"]),
        "median_rating": float(pd.to_numeric(df["Rating"], errors="coerce").median()),
        "recommendation_rate": _rate(recommended, len(reco)),
        "recommended_count": recommended,
        "not_recommended_count": not_recommended,
        "positive_reviews": positive,
        "positive_rate": _rate(positive, classified),
        "neutral_reviews": neutral,
        "neutral_rate": _rate(neutral, classified),
        "negative_reviews": negative,
        "negative_rate": _rate(negative, classified),
        "total_positive_feedback": int(
            pd.to_numeric(df["Positive Feedback Count"], errors="coerce").fillna(0).sum()
        ),
        "average_positive_feedback": _safe_mean(df["Positive Feedback Count"]),
        "average_age": _safe_mean(df["Age"]),
        "median_age": float(pd.to_numeric(df["Age"], errors="coerce").median()),
        "average_review_length": _safe_mean(
            df.loc[df["Has Review Text"], "Review Length Words"]
        ),
        "duplicate_review_rows": int(df["Is Duplicate Review Text"].sum())
        if "Is Duplicate Review Text" in df.columns
        else 0,
        "unique_clothing_ids": int(df["Clothing ID"].nunique()),
    }


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = _as_str(df["Sentiment"]).value_counts()
    total = int(sum(int(counts.get(label, 0)) for label in SENTIMENT_ORDER))
    return pd.DataFrame(
        [
            {
                "Sentiment": label,
                "Reviews": int(counts.get(label, 0)),
                "Share %": _rate(int(counts.get(label, 0)), total),
            }
            for label in SENTIMENT_ORDER
        ]
    )


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = pd.to_numeric(df["Rating"], errors="coerce").value_counts()
    total = int(sum(int(counts.get(rating, 0)) for rating in RATING_ORDER))
    return pd.DataFrame(
        [
            {
                "Rating": rating,
                "Rating Label": f"{rating} star" + ("" if rating == 1 else "s"),
                "Reviews": int(counts.get(rating, 0)),
                "Share %": _rate(int(counts.get(rating, 0)), total),
            }
            for rating in RATING_ORDER
        ]
    )


def recommendation_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = _as_str(df["Recommendation Label"]).value_counts()
    total = int(sum(int(counts.get(label, 0)) for label in RECOMMENDATION_ORDER))
    return pd.DataFrame(
        [
            {
                "Recommendation": label,
                "Reviews": int(counts.get(label, 0)),
                "Share %": _rate(int(counts.get(label, 0)), total),
            }
            for label in RECOMMENDATION_ORDER
        ]
    )


def recommendation_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Recommendation behaviour within each sentiment group (long form)."""
    work = pd.DataFrame(
        {
            "Sentiment": _as_str(df["Sentiment"]),
            "Recommendation": _as_str(df["Recommendation Label"]),
        }
    )
    work = work[work["Sentiment"].isin(SENTIMENT_ORDER)]
    work = work[work["Recommendation"].isin(RECOMMENDATION_ORDER)]

    rows = []
    for sentiment in SENTIMENT_ORDER:
        subset = work[work["Sentiment"] == sentiment]
        total = len(subset)
        for recommendation in RECOMMENDATION_ORDER:
            count = int((subset["Recommendation"] == recommendation).sum())
            rows.append(
                {
                    "Sentiment": sentiment,
                    "Recommendation": recommendation,
                    "Reviews": count,
                    "Share %": _rate(count, total),
                }
            )
    return pd.DataFrame(rows)


def age_distribution(df: pd.DataFrame) -> pd.Series:
    """Raw customer ages, for the histogram."""
    return pd.to_numeric(df["Age"], errors="coerce").dropna()


# --------------------------------------------------------------------------- #
# Segment performance
# --------------------------------------------------------------------------- #
SEGMENT_COLUMNS = (
    "Reviews", "Share %", "Avg Rating", "Recommended Reviews",
    "Not Recommended Reviews", "Recommendation Rate", "Positive Reviews",
    "Positive Rate", "Neutral Reviews", "Neutral Rate", "Negative Reviews",
    "Negative Rate", "Helpful Votes", "Avg Helpful Votes", "Reviews With Text",
)


def segment_performance(
    df: pd.DataFrame,
    by: str,
    min_reviews: int = 0,
    order: Sequence[str] | None = None,
    sort_by: str | None = "Reviews",
) -> pd.DataFrame:
    """Volume, rating, recommendation and negative metrics for one dimension.

    ``min_reviews`` suppresses segments whose sample is too small to interpret;
    suppressed rows are absent, never silently merged into another segment.
    """
    if by not in df.columns or df.empty:
        return pd.DataFrame(columns=[by, *SEGMENT_COLUMNS])

    work = pd.DataFrame(
        {
            "_segment": _as_str(df[by]),
            "_sentiment": _as_str(df["Sentiment"]),
            "_rating": pd.to_numeric(df["Rating"], errors="coerce"),
            "_reco": pd.to_numeric(df["Recommended IND"], errors="coerce"),
            "_helpful": pd.to_numeric(
                df["Positive Feedback Count"], errors="coerce"
            ).fillna(0),
            "_has_text": df["Has Review Text"].astype(bool),
        }
    )

    total_reviews = len(work)
    rows = []
    for segment, group in work.groupby("_segment", sort=False):
        n_classified = int(group["_sentiment"].isin(SENTIMENT_ORDER).sum())
        negative = int((group["_sentiment"] == "Negative").sum())
        neutral = int((group["_sentiment"] == "Neutral").sum())
        positive = int((group["_sentiment"] == "Positive").sum())
        reco = group["_reco"].dropna()
        recommended = int((reco == 1).sum())
        rows.append(
            {
                by: segment,
                "Reviews": int(len(group)),
                "Share %": _rate(len(group), total_reviews),
                "Avg Rating": float(group["_rating"].mean())
                if group["_rating"].notna().any()
                else 0.0,
                "Recommended Reviews": recommended,
                "Not Recommended Reviews": int((reco == 0).sum()),
                "Recommendation Rate": _rate(recommended, len(reco)),
                "Positive Reviews": positive,
                "Positive Rate": _rate(positive, n_classified),
                "Neutral Reviews": neutral,
                "Neutral Rate": _rate(neutral, n_classified),
                "Negative Reviews": negative,
                "Negative Rate": _rate(negative, n_classified),
                "Helpful Votes": int(group["_helpful"].sum()),
                "Avg Helpful Votes": float(group["_helpful"].mean()),
                "Reviews With Text": int(group["_has_text"].sum()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if min_reviews > 0:
        out = out[out["Reviews"] >= min_reviews]
    if out.empty:
        return out
    if order is not None:
        ordering = {label: i for i, label in enumerate(order)}
        out["_order"] = out[by].map(ordering).fillna(len(ordering))
        out = out.sort_values(["_order", by]).drop(columns="_order")
    elif sort_by and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=False)
    return out.reset_index(drop=True)


def division_performance(df: pd.DataFrame, min_reviews: int = 0) -> pd.DataFrame:
    return segment_performance(df, "Division Name", min_reviews=min_reviews)


def department_performance(df: pd.DataFrame, min_reviews: int = 0) -> pd.DataFrame:
    return segment_performance(df, "Department Name", min_reviews=min_reviews)


def age_band_performance(df: pd.DataFrame, min_reviews: int = 0) -> pd.DataFrame:
    order = list(AGE_BAND_ORDER) + [UNKNOWN_CATEGORY]
    return segment_performance(df, "Age Band", min_reviews=min_reviews, order=order)


def top_classes_by_volume(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    table = segment_performance(df, "Class Name")
    if table.empty:
        return table
    return (
        table.sort_values("Reviews", ascending=False).head(top_n).reset_index(drop=True)
    )


def performance_band(
    negative_rate: float, recommendation_rate: float, baseline_negative: float,
    baseline_recommendation: float,
) -> str:
    """Classify a segment for conditional formatting.

    Needs Attention: negative rate at least 5 pp above the filtered baseline.
    Strong:          negative rate at least 2 pp below baseline **and**
                     recommendation rate at or above the baseline.
    Average:         everything in between.
    """
    gap = negative_rate - baseline_negative
    if gap >= ATTENTION_GAP_PP:
        return "Needs Attention"
    if gap <= -AVERAGE_GAP_PP and recommendation_rate >= baseline_recommendation:
        return "Strong"
    return "Average"


def class_performance_matrix(
    df: pd.DataFrame, min_reviews: int = 50, top_n: int | None = None
) -> pd.DataFrame:
    """Product-class matrix with the columns required by the case study."""
    kpis = compute_kpis(df)
    baseline_negative = kpis.get("negative_rate", 0.0)
    baseline_recommendation = kpis.get("recommendation_rate", 0.0)

    table = segment_performance(df, "Class Name", min_reviews=min_reviews)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "Product Class", "Review Count", "Avg Rating",
                "Recommended Reviews", "Recommendation Rate", "Negative Reviews",
                "Negative Rate", "Total Helpful Votes", "Performance",
            ]
        )

    table = table.rename(
        columns={"Class Name": "Product Class", "Reviews": "Review Count",
                 "Helpful Votes": "Total Helpful Votes"}
    )
    table["Performance"] = [
        performance_band(
            float(row["Negative Rate"]),
            float(row["Recommendation Rate"]),
            baseline_negative,
            baseline_recommendation,
        )
        for _, row in table.iterrows()
    ]
    columns = [
        "Product Class", "Review Count", "Avg Rating", "Recommended Reviews",
        "Recommendation Rate", "Negative Reviews", "Negative Rate",
        "Total Helpful Votes", "Performance",
    ]
    table = table[columns].sort_values("Review Count", ascending=False).reset_index(
        drop=True
    )
    return table.head(top_n) if top_n else table


# --------------------------------------------------------------------------- #
# Helpful (Positive Feedback Count) analysis
# --------------------------------------------------------------------------- #
def helpful_votes_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Helpful-vote behaviour per sentiment class.

    ``Positive Feedback Count`` is the number of readers who marked a review as
    helpful. It is a *usefulness* signal, not a positive-sentiment signal.
    """
    work = pd.DataFrame(
        {
            "Sentiment": _as_str(df["Sentiment"]),
            "Helpful": pd.to_numeric(
                df["Positive Feedback Count"], errors="coerce"
            ).fillna(0),
        }
    )
    rows = []
    for sentiment in SENTIMENT_ORDER:
        subset = work[work["Sentiment"] == sentiment]
        rows.append(
            {
                "Sentiment": sentiment,
                "Reviews": int(len(subset)),
                "Total Helpful Votes": int(subset["Helpful"].sum()),
                "Avg Helpful Votes": float(subset["Helpful"].mean())
                if len(subset)
                else 0.0,
                "Median Helpful Votes": float(subset["Helpful"].median())
                if len(subset)
                else 0.0,
                "Reviews With >=1 Vote": int((subset["Helpful"] >= 1).sum()),
                "Share of Helpful Votes %": 0.0,
            }
        )
    out = pd.DataFrame(rows)
    grand_total = out["Total Helpful Votes"].sum()
    out["Share of Helpful Votes %"] = out["Total Helpful Votes"].apply(
        lambda v: _rate(v, grand_total)
    )
    return out


def helpful_votes_by_class(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    table = segment_performance(df, "Class Name")
    if table.empty:
        return table
    table = table.rename(columns={"Class Name": "Product Class"})
    return (
        table.sort_values("Helpful Votes", ascending=False)
        .head(top_n)[
            [
                "Product Class", "Reviews", "Helpful Votes", "Avg Helpful Votes",
                "Negative Rate", "Avg Rating",
            ]
        ]
        .reset_index(drop=True)
    )


REVIEW_DISPLAY_COLUMNS = (
    "Clothing ID", "Age", "Age Band", "Title", "Review Text", "Rating",
    "Sentiment", "Recommendation Label", "Positive Feedback Count",
    "Division Name", "Department Name", "Class Name",
)


def most_helpful_reviews(
    df: pd.DataFrame, top_n: int = 25, negative_only: bool = False
) -> pd.DataFrame:
    """Highest-voted reviews, optionally restricted to negative sentiment."""
    subset = df[df["Has Review Text"]].copy()
    if negative_only:
        subset = subset[_as_str(subset["Sentiment"]) == "Negative"]
    if subset.empty:
        return pd.DataFrame(columns=list(REVIEW_DISPLAY_COLUMNS))
    subset["Positive Feedback Count"] = (
        pd.to_numeric(subset["Positive Feedback Count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    subset = subset.sort_values(
        ["Positive Feedback Count", "Review Length Words"], ascending=[False, False]
    ).head(top_n)
    available = [c for c in REVIEW_DISPLAY_COLUMNS if c in subset.columns]
    return subset[available].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Actionable Insights (calculated, never hardcoded)
# --------------------------------------------------------------------------- #
@dataclass
class ActionableInsight:
    """One evidence-based observation in the professor's three-part format."""

    title: str
    what: str          # What the data shows
    why: str           # Why it matters to the business
    focus: str         # Which product area requires attention
    tone: str = "neutral"  # positive | neutral | negative

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "what": self.what,
            "why": self.why,
            "focus": self.focus,
            "tone": self.tone,
        }


def _no_data_insight(scope: str) -> list[ActionableInsight]:
    return [
        ActionableInsight(
            title="Insufficient data in scope",
            what=f"The current filter combination leaves no {scope} with enough "
            f"records to analyse.",
            why="Reporting on a handful of records would produce rates that move "
            "by tens of percentage points on a single review.",
            focus="Widen the filters or lower the minimum category sample size in "
            "the sidebar.",
            tone="neutral",
        )
    ]


def executive_insights(df: pd.DataFrame, min_reviews: int = 50) -> list[ActionableInsight]:
    """Executive Overview insights, all derived from the filtered frame."""
    if df.empty:
        return _no_data_insight("department")

    kpis = compute_kpis(df)
    baseline = kpis["negative_rate"]
    departments = department_performance(df, min_reviews=min_reviews)
    if departments.empty:
        return _no_data_insight("department")

    insights: list[ActionableInsight] = []

    volume = departments.sort_values("Reviews", ascending=False).iloc[0]
    insights.append(
        ActionableInsight(
            title="Highest-volume department",
            what=f"<b>{volume['Department Name']}</b> accounts for "
            f"<b>{int(volume['Reviews']):,}</b> reviews "
            f"({volume['Share %']:.1f}% of the {int(kpis['total_records']):,} in "
            f"scope), with an average rating of {volume['Avg Rating']:.2f} and a "
            f"{volume['Recommendation Rate']:.1f}% recommendation rate.",
            why="This department dominates the feedback base, so its rates set "
            "the tone of the overall numbers and a change here moves more "
            "absolute reviews than anywhere else.",
            focus=f"{volume['Department Name']} - treat as the reference volume "
            f"for any category comparison.",
            tone="neutral",
        )
    )

    rated = departments.sort_values("Avg Rating", ascending=False).iloc[0]
    insights.append(
        ActionableInsight(
            title="Highest-rated department",
            what=f"<b>{rated['Department Name']}</b> holds the highest average "
            f"rating at <b>{rated['Avg Rating']:.2f}</b> out of 5 across "
            f"{int(rated['Reviews']):,} reviews, with a "
            f"{rated['Negative Rate']:.1f}% negative rate against the "
            f"{baseline:.1f}% filtered baseline.",
            why="It is the internal benchmark: whatever this department does on "
            "product selection and description quality is what the weaker "
            "departments are being measured against.",
            focus=f"{rated['Department Name']} - protect current standards and "
            f"document what is working.",
            tone="positive",
        )
    )

    worst = departments.sort_values("Negative Rate", ascending=False).iloc[0]
    insights.append(
        ActionableInsight(
            title="Department with the highest negative rate",
            what=f"<b>{worst['Department Name']}</b> has the highest negative-review "
            f"rate at <b>{worst['Negative Rate']:.1f}%</b> "
            f"({int(worst['Negative Reviews']):,} of {int(worst['Reviews']):,} "
            f"reviews), which is {worst['Negative Rate'] - baseline:+.1f} "
            f"percentage points versus the {baseline:.1f}% baseline. Average "
            f"rating {worst['Avg Rating']:.2f}.",
            why="A negative rate above the baseline is associated with weaker "
            "advocacy in the same segment. Because low-star reviews stay visible "
            "on the product page, they influence what future shoppers read first.",
            focus=f"{worst['Department Name']} - review the underlying product "
            f"classes and the accuracy of their descriptions.",
            tone="negative",
        )
    )

    advocate = departments.sort_values("Recommendation Rate", ascending=False).iloc[0]
    insights.append(
        ActionableInsight(
            title="Department with the highest recommendation rate",
            what=f"<b>{advocate['Department Name']}</b> leads on advocacy with a "
            f"<b>{advocate['Recommendation Rate']:.1f}%</b> recommendation rate "
            f"({int(advocate['Recommended Reviews']):,} of "
            f"{int(advocate['Reviews']):,} reviews), against the "
            f"{kpis['recommendation_rate']:.1f}% overall filtered rate.",
            why="Recommendation is the customer's own stated intent rather than an "
            "inferred score, which makes it the closest available proxy for "
            "repeat purchase in this dataset.",
            focus=f"{advocate['Department Name']} - use as the comparison standard "
            f"when setting targets for the other departments.",
            tone="positive",
        )
    )

    insights.append(
        ActionableInsight(
            title="Records containing review text",
            what=f"<b>{kpis['text_coverage_rate']:.1f}%</b> of the "
            f"{int(kpis['total_records']):,} records in scope contain review text "
            f"({int(kpis['records_with_text']):,} rows); "
            f"{int(kpis['missing_review_text']):,} rows are rating-only.",
            why="Rating, recommendation and category analytics use all rows, while "
            "review-text search and display can only use the rows that have text. "
            "Knowing the gap prevents the two denominators being confused.",
            focus="Review display and search - rating-only rows are excluded there "
            "and retained everywhere else.",
            tone="neutral",
        )
    )

    return insights


def product_insights(df: pd.DataFrame, min_reviews: int = 50) -> list[ActionableInsight]:
    """Product Performance insights at product-class level."""
    if df.empty:
        return _no_data_insight("product class")

    kpis = compute_kpis(df)
    baseline = kpis["negative_rate"]
    classes = segment_performance(df, "Class Name", min_reviews=min_reviews)
    if classes.empty:
        return _no_data_insight("product class")

    insights: list[ActionableInsight] = []

    volume = classes.sort_values("Reviews", ascending=False).iloc[0]
    insights.append(
        ActionableInsight(
            title="Class with the highest review volume",
            what=f"<b>{volume['Class Name']}</b> generates the most reviews - "
            f"<b>{int(volume['Reviews']):,}</b> ({volume['Share %']:.1f}% of "
            f"records in scope) - with an average rating of "
            f"{volume['Avg Rating']:.2f} and a {volume['Negative Rate']:.1f}% "
            f"negative rate.",
            why="Volume equals exposure. Even a small rate improvement in this "
            "class changes more absolute reviews than a large improvement in a "
            "small class.",
            focus=f"{volume['Class Name']} - highest leverage for any change to "
            f"sizing guidance or product copy.",
            tone="neutral",
        )
    )

    lowest_rated = classes.sort_values("Avg Rating", ascending=True).iloc[0]
    insights.append(
        ActionableInsight(
            title="Class with the lowest average rating",
            what=f"<b>{lowest_rated['Class Name']}</b> has the lowest average "
            f"rating at <b>{lowest_rated['Avg Rating']:.2f}</b> out of 5 across "
            f"{int(lowest_rated['Reviews']):,} reviews, with a "
            f"{lowest_rated['Recommendation Rate']:.1f}% recommendation rate.",
            why="A depressed average across a qualifying sample indicates a "
            "consistent shortfall against expectation rather than a few "
            "dissatisfied individuals.",
            focus=f"{lowest_rated['Class Name']} - compare the product "
            f"description against what customers actually received.",
            tone="negative",
        )
    )

    worst = classes.sort_values("Negative Rate", ascending=False).iloc[0]
    insights.append(
        ActionableInsight(
            title="Class with the highest negative-review rate",
            what=f"<b>{worst['Class Name']}</b> has the highest negative rate at "
            f"<b>{worst['Negative Rate']:.1f}%</b> "
            f"({int(worst['Negative Reviews']):,} of {int(worst['Reviews']):,} "
            f"reviews), {worst['Negative Rate'] - baseline:+.1f} percentage points "
            f"versus the {baseline:.1f}% baseline.",
            why="This is the class where dissatisfaction is most concentrated "
            "relative to its own size, so it is the clearest candidate for a "
            "targeted product or content review.",
            focus=f"{worst['Class Name']} - read its most helpful negative reviews "
            f"on the Customer & Review Insights tab.",
            tone="negative",
        )
    )

    advocate = classes.sort_values("Recommendation Rate", ascending=False).iloc[0]
    insights.append(
        ActionableInsight(
            title="Class with the strongest recommendation rate",
            what=f"<b>{advocate['Class Name']}</b> reaches a "
            f"<b>{advocate['Recommendation Rate']:.1f}%</b> recommendation rate "
            f"over {int(advocate['Reviews']):,} reviews, with an average rating of "
            f"{advocate['Avg Rating']:.2f} and only "
            f"{int(advocate['Negative Reviews']):,} negative reviews.",
            why="Classes at this level of advocacy show what the assortment looks "
            "like when expectation and delivery align, which is the practical "
            "target for the weaker classes.",
            focus=f"{advocate['Class Name']} - benchmark its description and "
            f"sizing detail against the weakest classes.",
            tone="positive",
        )
    )

    attention = classes[classes["Negative Rate"] - baseline >= ATTENTION_GAP_PP]
    if not attention.empty:
        insights.append(
            ActionableInsight(
                title="Classes flagged for attention",
                what=f"<b>{len(attention)}</b> product class(es) sit at least "
                f"{ATTENTION_GAP_PP:.0f} percentage points above the "
                f"{baseline:.1f}% baseline negative rate, together holding "
                f"<b>{int(attention['Negative Reviews'].sum()):,}</b> negative "
                f"reviews: "
                + ", ".join(
                    f"{row['Class Name']} ({row['Negative Rate']:.1f}%)"
                    for _, row in attention.sort_values(
                        "Negative Rate", ascending=False
                    ).head(5).iterrows()
                )
                + ".",
                why="These are the segments where the gap is wide enough not to be "
                "explained by sample noise at the current minimum sample size.",
                focus="The classes listed above - prioritise in that order by "
                "negative-review count.",
                tone="negative",
            )
        )

    return insights


def customer_insights(df: pd.DataFrame, min_reviews: int = 50) -> list[ActionableInsight]:
    """Customer & Review Insights observations."""
    if df.empty:
        return _no_data_insight("age band")

    kpis = compute_kpis(df)
    ages = age_band_performance(df, min_reviews=min_reviews)
    if ages.empty:
        return _no_data_insight("age band")

    insights: list[ActionableInsight] = []

    advocate = ages.sort_values("Recommendation Rate", ascending=False).iloc[0]
    laggard = ages.sort_values("Recommendation Rate", ascending=True).iloc[0]
    insights.append(
        ActionableInsight(
            title="Age band with the highest recommendation rate",
            what=f"The <b>{advocate['Age Band']}</b> cohort recommends most often "
            f"at <b>{advocate['Recommendation Rate']:.1f}%</b> across "
            f"{int(advocate['Reviews']):,} reviews, while "
            f"{laggard['Age Band']} is lowest at "
            f"{laggard['Recommendation Rate']:.1f}% - a spread of "
            f"{advocate['Recommendation Rate'] - laggard['Recommendation Rate']:.1f} "
            f"percentage points.",
            why="Age is the only customer attribute in this dataset, so cohort "
            "spread is the sole available read on whether the assortment suits "
            "some customers better than others.",
            focus=f"{laggard['Age Band']} cohort - check which classes it reviews "
            f"most and how those classes are described.",
            tone="positive",
        )
    )

    lowest_rated = ages.sort_values("Avg Rating", ascending=True).iloc[0]
    insights.append(
        ActionableInsight(
            title="Age band with the lowest average rating",
            what=f"The <b>{lowest_rated['Age Band']}</b> cohort gives the lowest "
            f"average rating at <b>{lowest_rated['Avg Rating']:.2f}</b> out of 5 "
            f"over {int(lowest_rated['Reviews']):,} reviews, with a "
            f"{lowest_rated['Negative Rate']:.1f}% negative rate against the "
            f"{kpis['negative_rate']:.1f}% baseline.",
            why="A cohort rating consistently below the others is associated with "
            "a fit or expectation mismatch in the products that cohort buys. The "
            "data shows the association, not the cause.",
            focus=f"{lowest_rated['Age Band']} cohort - review the sizing and "
            f"length detail on the classes it reviews most.",
            tone="negative",
        )
    )

    helpful = helpful_votes_by_sentiment(df)
    if not helpful.empty and helpful["Total Helpful Votes"].sum() > 0:
        top = helpful.sort_values("Total Helpful Votes", ascending=False).iloc[0]
        by_average = helpful.sort_values("Avg Helpful Votes", ascending=False).iloc[0]
        insights.append(
            ActionableInsight(
                title="Sentiment group receiving the most helpful votes",
                what=f"<b>{top['Sentiment']}</b> reviews collect the most helpful "
                f"votes in total - <b>{int(top['Total Helpful Votes']):,}</b> "
                f"({top['Share of Helpful Votes %']:.1f}% of all votes) across "
                f"{int(top['Reviews']):,} reviews. Per review, "
                f"<b>{by_average['Sentiment']}</b> reviews are voted most useful "
                f"at {by_average['Avg Helpful Votes']:.2f} votes each.",
                why="Helpful votes measure how useful other shoppers found a "
                "review, not how positive it was. A high average on critical "
                "reviews means that criticism is the content prospective buyers "
                "are reading most.",
                focus="The highest-voted reviews listed below - read them before "
                "changing any product description.",
                tone="neutral",
            )
        )

    top_negative = most_helpful_reviews(df, top_n=1, negative_only=True)
    if not top_negative.empty:
        record = top_negative.iloc[0]
        excerpt = str(record["Review Text"])
        if len(excerpt) > 260:
            excerpt = excerpt[:257].rstrip() + "..."
        insights.append(
            ActionableInsight(
                title="Most helpful negative review",
                what=f"The single most-voted negative review has "
                f"<b>{int(record['Positive Feedback Count']):,}</b> helpful votes: "
                f"{int(record['Rating'])} stars, "
                f"{record['Department Name']} / {record['Class Name']}, customer "
                f"age {int(record['Age'])}. &ldquo;{excerpt}&rdquo;",
                why="This is the most-read piece of criticism in the current "
                "scope. Its specific objection is the one most likely to be "
                "shaping purchase decisions for this class.",
                focus=f"{record['Class Name']} in {record['Department Name']} - "
                f"verify the objection raised here against the live product page.",
                tone="negative",
            )
        )

    return insights


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compare calculated values against the published reference figures.

    The frame supplied here must be the *unfiltered* cleaned dataset.
    """
    kpis = compute_kpis(df)
    rows = []
    for key, spec in EXPECTED_VALIDATION.items():
        actual = float(kpis.get(key, 0.0))
        expected = float(spec["expected"])
        tolerance = float(spec["tolerance"])
        delta = actual - expected
        passed = abs(delta) <= tolerance if tolerance else actual == expected
        if spec["kind"] == "percent":
            formatted_actual, formatted_expected = f"{actual:,.2f}%", f"{expected:,.2f}%"
        elif spec["kind"] == "rating":
            formatted_actual, formatted_expected = f"{actual:,.3f}", f"{expected:,.3f}"
        else:
            formatted_actual, formatted_expected = f"{actual:,.0f}", f"{expected:,.0f}"
        rows.append(
            {
                "Metric": spec["label"],
                "Calculated From Data": formatted_actual,
                "Expected Reference": formatted_expected,
                "Difference": f"{delta:+,.3f}" if tolerance else f"{delta:+,.0f}",
                "Status": "Match" if passed else "Deviation",
                "_passed": bool(passed),
            }
        )
    return pd.DataFrame(rows)


def validation_summary(df: pd.DataFrame) -> dict[str, Any]:
    table = validation_table(df)
    failures = table[~table["_passed"]]
    return {
        "checks": int(len(table)),
        "passed": int(table["_passed"].sum()),
        "failed": int(len(failures)),
        "failed_metrics": failures["Metric"].tolist(),
        "table": table.drop(columns="_passed"),
        "all_passed": bool(failures.empty),
    }


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def apply_filters(
    df: pd.DataFrame,
    divisions: Iterable[str] | None = None,
    departments: Iterable[str] | None = None,
    classes: Iterable[str] | None = None,
    ratings: Iterable[int] | None = None,
    sentiments: Iterable[str] | None = None,
    recommendations: Iterable[str] | None = None,
    age_bands: Iterable[str] | None = None,
    has_review_text: str = "All records",
    keyword: str = "",
) -> pd.DataFrame:
    """Apply the synchronized global filters. An empty selection means 'all'."""
    mask = pd.Series(True, index=df.index)

    def add(column: str, values: Iterable[str] | None) -> pd.Series:
        selected = list(values or [])
        if not selected:
            return pd.Series(True, index=df.index)
        return _as_str(df[column]).isin([str(v) for v in selected])

    mask &= add("Division Name", divisions)
    mask &= add("Department Name", departments)
    mask &= add("Class Name", classes)
    mask &= add("Sentiment", sentiments)
    mask &= add("Recommendation Label", recommendations)
    mask &= add("Age Band", age_bands)

    rating_values = list(ratings or [])
    if rating_values:
        mask &= pd.to_numeric(df["Rating"], errors="coerce").isin(
            [int(r) for r in rating_values]
        )

    if has_review_text == "With review text only":
        mask &= df["Has Review Text"]
    elif has_review_text == "Without review text only":
        mask &= ~df["Has Review Text"]

    term = (keyword or "").strip()
    if term:
        haystack = (
            df["Review Text"].astype(str) + " " + df["Title"].astype(str)
        ).str.lower()
        mask &= haystack.str.contains(term.lower(), regex=False, na=False)

    return df.loc[mask].copy()


def search_reviews(df: pd.DataFrame, term: str) -> pd.DataFrame:
    """Plain case-insensitive substring search over Title and Review Text.

    This is text *retrieval*, not text analysis: no tokenising, stemming, topic
    extraction or scoring is applied to the review wording anywhere in this
    application.
    """
    value = (term or "").strip()
    if not value:
        return df
    haystack = (
        df["Review Text"].astype(str) + " " + df["Title"].astype(str)
    ).str.lower()
    return df[haystack.str.contains(value.lower(), regex=False, na=False)]


def filter_summary(
    filters: dict[str, Any], filtered_rows: int, total_rows: int
) -> list[str]:
    """Readable description of the active filters for the sidebar."""
    lines: list[str] = []
    label_map = {
        "divisions": "Division",
        "departments": "Department",
        "classes": "Class",
        "ratings": "Rating",
        "sentiments": "Sentiment",
        "recommendations": "Recommendation",
        "age_bands": "Age Band",
    }
    for key, label in label_map.items():
        values = filters.get(key) or []
        if values:
            preview = ", ".join(str(v) for v in list(values)[:3])
            if len(values) > 3:
                preview += f" +{len(values) - 3} more"
            lines.append(f"<b>{label}:</b> {preview}")

    if filters.get("has_review_text") and filters["has_review_text"] != "All records":
        lines.append(f"<b>Has Review Text:</b> {filters['has_review_text']}")
    if filters.get("min_sample", 0):
        lines.append(
            f"<b>Min category sample:</b> {filters['min_sample']} reviews"
        )
    if not lines:
        lines.append("No filters applied - showing the complete dataset.")
    return lines


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for the numeric structured columns."""
    columns = ["Age", "Rating", "Positive Feedback Count", "Review Length Words"]
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "Measure": column,
                "Count": int(series.size),
                "Mean": round(float(series.mean()), 2),
                "Median": round(float(series.median()), 2),
                "Std Dev": round(float(series.std(ddof=1)), 2)
                if series.size > 1
                else 0.0,
                "Min": round(float(series.min()), 2),
                "P25": round(float(series.quantile(0.25)), 2),
                "P75": round(float(series.quantile(0.75)), 2),
                "Max": round(float(series.max()), 2),
            }
        )
    return pd.DataFrame(rows)
