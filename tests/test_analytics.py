"""Tests for the analytical calculations.

Two things are checked here:

1. **Correctness against the published reference figures** - the headline KPIs
   calculated from the real file must reproduce the documented totals for the
   Women's E-Commerce Clothing Reviews dataset.
2. **Internal consistency** - segment tables, distributions, filters and the
   generated insights must agree with the KPIs and with each other under any
   filter combination.

These are fast, pure-pandas checks. There is no model to train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import analytics, data_cleaning, data_loader


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def clean() -> pd.DataFrame:
    raw, _ = data_loader.load_default_dataset()
    frame, _audit = data_cleaning.clean_dataset(raw)
    return frame


@pytest.fixture(scope="session")
def kpis(clean: pd.DataFrame) -> dict[str, float]:
    return analytics.compute_kpis(clean)


# --------------------------------------------------------------------------- #
# Validation against the published reference figures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(analytics.EXPECTED_VALIDATION.keys()))
def test_kpi_matches_the_expected_reference_value(
    kpis: dict[str, float], key: str
) -> None:
    spec = analytics.EXPECTED_VALIDATION[key]
    actual = kpis[key]
    expected = float(spec["expected"])
    tolerance = float(spec["tolerance"])
    if tolerance:
        assert actual == pytest.approx(expected, abs=tolerance), (
            f"{spec['label']}: calculated {actual!r}, expected ~{expected!r}"
        )
    else:
        assert actual == expected, (
            f"{spec['label']}: calculated {actual!r}, expected {expected!r}"
        )


def test_all_validation_checks_pass(clean: pd.DataFrame) -> None:
    summary = analytics.validation_summary(clean)
    assert summary["checks"] == 15
    assert summary["failed"] == 0, f"Deviations: {summary['failed_metrics']}"
    assert summary["passed"] == summary["checks"]
    assert summary["all_passed"]


def test_validation_table_never_shows_a_reference_value_as_the_result(
    clean: pd.DataFrame,
) -> None:
    """The 'Calculated' column must come from the data, not from the constant."""
    table = analytics.validation_table(clean)
    assert set(table.columns) == {
        "Metric",
        "Calculated From Data",
        "Expected Reference",
        "Difference",
        "Status",
        "_passed",
    }
    assert (table["Status"] == "Match").all()
    # Average Age differs from the rounded reference in the third decimal, which
    # proves the value is computed rather than copied.
    average_age = table.loc[table["Metric"] == "Average Age"].iloc[0]
    assert average_age["Calculated From Data"] != average_age["Expected Reference"]


def test_validation_flags_a_deviating_dataset(clean: pd.DataFrame) -> None:
    """A different dataset must be reported as deviating, not silently accepted."""
    subset = clean.head(5_000)
    summary = analytics.validation_summary(subset)
    assert not summary["all_passed"]
    assert summary["failed"] > 0
    assert "Total Reviews" in summary["failed_metrics"]


# --------------------------------------------------------------------------- #
# KPI internal consistency
# --------------------------------------------------------------------------- #
def test_sentiment_counts_and_rates_are_consistent(kpis: dict[str, float]) -> None:
    total_classified = (
        kpis["positive_reviews"] + kpis["neutral_reviews"] + kpis["negative_reviews"]
    )
    assert total_classified == kpis["total_records"]
    rate_sum = kpis["positive_rate"] + kpis["neutral_rate"] + kpis["negative_rate"]
    assert rate_sum == pytest.approx(100.0, abs=1e-6)


def test_recommendation_counts_and_rate_are_consistent(kpis: dict[str, float]) -> None:
    assert (
        kpis["recommended_count"] + kpis["not_recommended_count"]
        == kpis["total_records"]
    )
    expected = kpis["recommended_count"] / kpis["total_records"] * 100
    assert kpis["recommendation_rate"] == pytest.approx(expected, abs=1e-9)


def test_text_coverage_is_consistent(kpis: dict[str, float]) -> None:
    assert (
        kpis["records_with_text"] + kpis["missing_review_text"]
        == kpis["total_records"]
    )
    expected = kpis["records_with_text"] / kpis["total_records"] * 100
    assert kpis["text_coverage_rate"] == pytest.approx(expected, abs=1e-9)


def test_kpis_on_an_empty_frame_are_zero_not_an_error(clean: pd.DataFrame) -> None:
    empty = clean.iloc[0:0]
    result = analytics.compute_kpis(empty)
    assert set(result.keys()) == set(analytics.KPI_KEYS)
    assert all(value == 0.0 for value in result.values())


def test_helpful_votes_kpi_matches_the_column_total(clean: pd.DataFrame) -> None:
    kpis = analytics.compute_kpis(clean)
    assert kpis["total_positive_feedback"] == int(
        clean["Positive Feedback Count"].sum()
    )


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def test_sentiment_distribution_matches_the_kpis(
    clean: pd.DataFrame, kpis: dict[str, float]
) -> None:
    table = analytics.sentiment_distribution(clean)
    assert table["Sentiment"].tolist() == list(data_cleaning.SENTIMENT_ORDER)
    assert int(table["Reviews"].sum()) == int(kpis["total_records"])
    assert table["Share %"].sum() == pytest.approx(100.0, abs=1e-6)
    by_label = dict(zip(table["Sentiment"], table["Reviews"]))
    assert by_label["Positive"] == kpis["positive_reviews"]
    assert by_label["Neutral"] == kpis["neutral_reviews"]
    assert by_label["Negative"] == kpis["negative_reviews"]


def test_rating_distribution_is_ordered_one_to_five(clean: pd.DataFrame) -> None:
    table = analytics.rating_distribution(clean)
    assert table["Rating"].tolist() == [1, 2, 3, 4, 5]
    assert int(table["Reviews"].sum()) == len(clean)
    assert table["Share %"].sum() == pytest.approx(100.0, abs=1e-6)
    assert table["Rating Label"].tolist() == [
        "1 star",
        "2 stars",
        "3 stars",
        "4 stars",
        "5 stars",
    ]


def test_rating_distribution_agrees_with_the_sentiment_mapping(
    clean: pd.DataFrame, kpis: dict[str, float]
) -> None:
    table = analytics.rating_distribution(clean).set_index("Rating")["Reviews"]
    assert table[1] + table[2] == kpis["negative_reviews"]
    assert table[3] == kpis["neutral_reviews"]
    assert table[4] + table[5] == kpis["positive_reviews"]


def test_recommendation_distribution_matches_the_kpis(
    clean: pd.DataFrame, kpis: dict[str, float]
) -> None:
    table = analytics.recommendation_distribution(clean)
    assert table["Recommendation"].tolist() == list(
        data_cleaning.RECOMMENDATION_ORDER
    )
    by_label = dict(zip(table["Recommendation"], table["Reviews"]))
    assert by_label["Recommended"] == kpis["recommended_count"]
    assert by_label["Not Recommended"] == kpis["not_recommended_count"]
    assert table["Share %"].sum() == pytest.approx(100.0, abs=1e-6)


def test_recommendation_by_sentiment_shares_sum_within_each_group(
    clean: pd.DataFrame,
) -> None:
    table = analytics.recommendation_by_sentiment(clean)
    assert len(table) == 6  # 3 sentiments x 2 recommendation states
    assert int(table["Reviews"].sum()) == len(clean)
    for sentiment in data_cleaning.SENTIMENT_ORDER:
        group = table[table["Sentiment"] == sentiment]
        assert group["Share %"].sum() == pytest.approx(100.0, abs=1e-6)


def test_negative_reviews_rarely_recommend(clean: pd.DataFrame) -> None:
    """A sanity check that the two independent fields tell a coherent story."""
    table = analytics.recommendation_by_sentiment(clean)
    negative_recommended = table[
        (table["Sentiment"] == "Negative") & (table["Recommendation"] == "Recommended")
    ].iloc[0]
    positive_recommended = table[
        (table["Sentiment"] == "Positive") & (table["Recommendation"] == "Recommended")
    ].iloc[0]
    assert negative_recommended["Share %"] < 25.0
    assert positive_recommended["Share %"] > 90.0


def test_age_distribution_matches_the_age_column(clean: pd.DataFrame) -> None:
    ages = analytics.age_distribution(clean)
    assert len(ages) == len(clean)
    assert float(ages.mean()) == pytest.approx(43.20, abs=0.1)
    assert float(ages.median()) == 41.0


# --------------------------------------------------------------------------- #
# Segment performance
# --------------------------------------------------------------------------- #
def test_segment_performance_volumes_sum_to_the_total(clean: pd.DataFrame) -> None:
    for dimension in ("Division Name", "Department Name", "Class Name", "Age Band"):
        table = analytics.segment_performance(clean, dimension)
        assert int(table["Reviews"].sum()) == len(clean), dimension
        assert table["Share %"].sum() == pytest.approx(100.0, abs=1e-6), dimension


def test_segment_performance_counts_sum_within_each_segment(
    clean: pd.DataFrame,
) -> None:
    table = analytics.segment_performance(clean, "Department Name")
    sentiment_total = (
        table["Positive Reviews"] + table["Neutral Reviews"] + table["Negative Reviews"]
    )
    assert (sentiment_total == table["Reviews"]).all()
    reco_total = table["Recommended Reviews"] + table["Not Recommended Reviews"]
    assert (reco_total == table["Reviews"]).all()


def test_segment_performance_rates_are_within_bounds(clean: pd.DataFrame) -> None:
    table = analytics.segment_performance(clean, "Class Name")
    for column in ("Recommendation Rate", "Positive Rate", "Neutral Rate", "Negative Rate"):
        assert table[column].between(0, 100).all(), column
    assert table["Avg Rating"].between(1, 5).all()


def test_segment_performance_rates_are_recomputed_per_segment(
    clean: pd.DataFrame,
) -> None:
    """Rates must be calculated inside the segment, not inherited from the total."""
    table = analytics.segment_performance(clean, "Department Name")
    row = table.iloc[0]
    department = row["Department Name"]
    subset = clean[clean["Department Name"].astype(str) == department]
    assert int(row["Reviews"]) == len(subset)
    assert float(row["Avg Rating"]) == pytest.approx(
        float(subset["Rating"].mean()), abs=1e-9
    )
    expected_negative = (
        (subset["Sentiment"].astype("object").astype(str) == "Negative").sum()
        / len(subset)
        * 100
    )
    assert float(row["Negative Rate"]) == pytest.approx(expected_negative, abs=1e-9)


def test_minimum_sample_size_suppresses_small_segments(clean: pd.DataFrame) -> None:
    unfiltered = analytics.segment_performance(clean, "Class Name", min_reviews=0)
    limited = analytics.segment_performance(clean, "Class Name", min_reviews=50)
    assert len(limited) < len(unfiltered)
    assert (limited["Reviews"] >= 50).all()
    # Suppressed rows are dropped, never merged into another segment.
    assert set(limited["Class Name"]) <= set(unfiltered["Class Name"])


def test_age_band_performance_is_returned_in_band_order(clean: pd.DataFrame) -> None:
    table = analytics.age_band_performance(clean)
    expected = [
        band
        for band in data_cleaning.AGE_BAND_ORDER
        if band in set(table["Age Band"])
    ]
    assert table["Age Band"].tolist() == expected
    assert int(table["Reviews"].sum()) == len(clean)


def test_top_classes_by_volume_is_sorted_and_capped(clean: pd.DataFrame) -> None:
    table = analytics.top_classes_by_volume(clean, top_n=10)
    assert len(table) == 10
    assert table["Reviews"].is_monotonic_decreasing


def test_segment_performance_on_a_missing_column_returns_an_empty_frame(
    clean: pd.DataFrame,
) -> None:
    table = analytics.segment_performance(clean, "Nonexistent Column")
    assert table.empty


# --------------------------------------------------------------------------- #
# Performance bands and the product-class matrix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("negative_rate", "recommendation_rate", "expected"),
    [
        (20.0, 60.0, "Needs Attention"),   # +10 pp above baseline
        (15.3, 90.0, "Needs Attention"),   # +5.3 pp, attention regardless of advocacy
        (12.0, 90.0, "Average"),           # +2 pp
        (10.0, 90.0, "Average"),           # at baseline
        (7.0, 90.0, "Strong"),             # -3 pp and strong advocacy
        (7.0, 70.0, "Average"),            # -3 pp but weak advocacy
    ],
)
def test_performance_band(negative_rate, recommendation_rate, expected) -> None:
    assert (
        analytics.performance_band(
            negative_rate, recommendation_rate, baseline_negative=10.0,
            baseline_recommendation=82.0,
        )
        == expected
    )


def test_class_performance_matrix_structure(clean: pd.DataFrame) -> None:
    matrix = analytics.class_performance_matrix(clean, min_reviews=50)
    assert list(matrix.columns) == [
        "Product Class",
        "Review Count",
        "Avg Rating",
        "Recommended Reviews",
        "Recommendation Rate",
        "Negative Reviews",
        "Negative Rate",
        "Total Helpful Votes",
        "Performance",
    ]
    assert not matrix.empty
    assert (matrix["Review Count"] >= 50).all()
    assert matrix["Review Count"].is_monotonic_decreasing
    assert set(matrix["Performance"]) <= set(analytics.PERFORMANCE_ORDER)


def test_class_performance_matrix_recommended_count_matches_the_rate(
    clean: pd.DataFrame,
) -> None:
    matrix = analytics.class_performance_matrix(clean, min_reviews=50)
    implied = matrix["Recommended Reviews"] / matrix["Review Count"] * 100
    assert np.allclose(implied, matrix["Recommendation Rate"], atol=1e-6)


def test_class_performance_matrix_is_empty_when_nothing_qualifies(
    clean: pd.DataFrame,
) -> None:
    matrix = analytics.class_performance_matrix(clean, min_reviews=100_000)
    assert matrix.empty
    assert "Performance" in matrix.columns


# --------------------------------------------------------------------------- #
# Helpful votes
# --------------------------------------------------------------------------- #
def test_helpful_votes_by_sentiment_totals_match_the_dataset(
    clean: pd.DataFrame, kpis: dict[str, float]
) -> None:
    table = analytics.helpful_votes_by_sentiment(clean)
    assert table["Sentiment"].tolist() == list(data_cleaning.SENTIMENT_ORDER)
    assert int(table["Reviews"].sum()) == len(clean)
    assert int(table["Total Helpful Votes"].sum()) == int(
        kpis["total_positive_feedback"]
    )
    assert table["Share of Helpful Votes %"].sum() == pytest.approx(100.0, abs=1e-6)


def test_helpful_votes_are_not_treated_as_positive_sentiment(
    clean: pd.DataFrame,
) -> None:
    """Negative reviews must be able to hold helpful votes, and they do."""
    table = analytics.helpful_votes_by_sentiment(clean).set_index("Sentiment")
    assert table.loc["Negative", "Total Helpful Votes"] > 0
    assert table.loc["Negative", "Avg Helpful Votes"] > 0
    # Per review, criticism is voted at least as useful as praise in this data.
    assert (
        table.loc["Negative", "Avg Helpful Votes"]
        > table.loc["Positive", "Avg Helpful Votes"]
    )


def test_helpful_votes_by_class_is_sorted_and_capped(clean: pd.DataFrame) -> None:
    table = analytics.helpful_votes_by_class(clean, top_n=10)
    assert len(table) == 10
    assert table["Helpful Votes"].is_monotonic_decreasing


def test_most_helpful_reviews_only_returns_rows_with_text(
    clean: pd.DataFrame,
) -> None:
    table = analytics.most_helpful_reviews(clean, top_n=25)
    assert len(table) == 25
    assert table["Positive Feedback Count"].is_monotonic_decreasing
    assert (table["Review Text"] != data_cleaning.MISSING_REVIEW_TEXT).all()
    assert list(table.columns) == list(analytics.REVIEW_DISPLAY_COLUMNS)


def test_most_helpful_negative_reviews_are_all_negative(clean: pd.DataFrame) -> None:
    table = analytics.most_helpful_reviews(clean, top_n=10, negative_only=True)
    assert len(table) == 10
    assert (table["Sentiment"].astype("object").astype(str) == "Negative").all()
    assert (table["Rating"] <= 2).all()


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_no_filters_returns_the_whole_dataset(clean: pd.DataFrame) -> None:
    assert len(analytics.apply_filters(clean)) == len(clean)


def test_single_dimension_filters(clean: pd.DataFrame) -> None:
    result = analytics.apply_filters(clean, departments=["Dresses"])
    assert len(result) > 0
    assert set(result["Department Name"].astype(str)) == {"Dresses"}

    result = analytics.apply_filters(clean, ratings=[1, 2])
    assert set(result["Rating"].unique()) == {1, 2}
    assert set(result["Sentiment"].astype("object").astype(str)) == {"Negative"}

    result = analytics.apply_filters(clean, sentiments=["Positive"])
    assert set(result["Rating"].unique()) == {4, 5}

    result = analytics.apply_filters(clean, recommendations=["Not Recommended"])
    assert set(result["Recommended IND"].unique()) == {0}

    result = analytics.apply_filters(clean, age_bands=["18-29"])
    assert result["Age"].between(18, 29).all()


def test_combined_filters_are_an_intersection(clean: pd.DataFrame) -> None:
    result = analytics.apply_filters(
        clean,
        departments=["Tops"],
        sentiments=["Negative"],
        age_bands=["30-39", "40-49"],
    )
    assert set(result["Department Name"].astype(str)) == {"Tops"}
    assert set(result["Sentiment"].astype("object").astype(str)) == {"Negative"}
    assert result["Age"].between(30, 49).all()
    assert len(result) < len(analytics.apply_filters(clean, departments=["Tops"]))


def test_has_review_text_filter(clean: pd.DataFrame, kpis: dict[str, float]) -> None:
    with_text = analytics.apply_filters(clean, has_review_text="With review text only")
    without = analytics.apply_filters(
        clean, has_review_text="Without review text only"
    )
    assert len(with_text) == int(kpis["records_with_text"])
    assert len(without) == int(kpis["missing_review_text"])
    assert len(with_text) + len(without) == len(clean)
    assert with_text["Has Review Text"].all()
    assert not without["Has Review Text"].any()


def test_filters_drive_the_kpis(clean: pd.DataFrame) -> None:
    """The whole point of the filters: the KPIs must actually change."""
    baseline = analytics.compute_kpis(clean)
    filtered = analytics.compute_kpis(
        analytics.apply_filters(clean, departments=["Dresses"])
    )
    assert filtered["total_records"] < baseline["total_records"]
    assert filtered["average_rating"] != baseline["average_rating"]
    assert filtered["negative_rate"] != baseline["negative_rate"]


def test_impossible_filter_combination_returns_an_empty_frame(
    clean: pd.DataFrame,
) -> None:
    result = analytics.apply_filters(clean, ratings=[5], sentiments=["Negative"])
    assert result.empty


def test_filters_never_mutate_the_source_frame(clean: pd.DataFrame) -> None:
    before = len(clean)
    analytics.apply_filters(clean, departments=["Tops"], ratings=[1])
    assert len(clean) == before


def test_search_reviews_is_a_plain_substring_match(clean: pd.DataFrame) -> None:
    result = analytics.search_reviews(clean, "runs small")
    assert len(result) > 0
    haystack = (
        result["Review Text"].astype(str) + " " + result["Title"].astype(str)
    ).str.lower()
    assert haystack.str.contains("runs small").all()
    # Case-insensitive, and an empty term is a no-op.
    assert len(analytics.search_reviews(clean, "RUNS SMALL")) == len(result)
    assert len(analytics.search_reviews(clean, "   ")) == len(clean)


def test_search_for_a_nonsense_term_returns_nothing(clean: pd.DataFrame) -> None:
    assert analytics.search_reviews(clean, "zzzznotaword").empty


def test_filter_summary_reports_the_active_filters(clean: pd.DataFrame) -> None:
    lines = analytics.filter_summary(
        {"departments": ["Tops"], "ratings": [1, 2], "min_sample": 50},
        filtered_rows=100,
        total_rows=len(clean),
    )
    joined = " ".join(lines)
    assert "Tops" in joined
    assert "Rating" in joined
    assert "50" in joined


def test_filter_summary_with_no_filters(clean: pd.DataFrame) -> None:
    lines = analytics.filter_summary(
        {"min_sample": 0}, filtered_rows=len(clean), total_rows=len(clean)
    )
    assert any("No filters applied" in line for line in lines)


# --------------------------------------------------------------------------- #
# Actionable Insights - calculated, never hardcoded
# --------------------------------------------------------------------------- #
INSIGHT_BUILDERS = (
    analytics.executive_insights,
    analytics.product_insights,
    analytics.customer_insights,
)


@pytest.mark.parametrize("builder", INSIGHT_BUILDERS)
def test_insights_are_generated_in_the_three_part_format(builder, clean) -> None:
    insights = builder(clean, min_reviews=50)
    assert len(insights) >= 4
    for insight in insights:
        assert isinstance(insight, analytics.ActionableInsight)
        assert insight.title
        assert len(insight.what) > 40
        assert len(insight.why) > 40
        assert len(insight.focus) > 10
        assert insight.tone in {"positive", "neutral", "negative"}


@pytest.mark.parametrize("builder", INSIGHT_BUILDERS)
def test_insights_quote_figures_from_the_data(builder, clean) -> None:
    """Each insight must cite a number, which only calculation can supply."""
    for insight in builder(clean, min_reviews=50):
        assert any(character.isdigit() for character in insight.what), insight.title


@pytest.mark.parametrize("builder", INSIGHT_BUILDERS)
def test_insight_wording_changes_with_the_filters(builder, clean) -> None:
    """Hardcoded sentences would be identical under a different scope."""
    full = builder(clean, min_reviews=50)
    scoped = builder(
        analytics.apply_filters(clean, departments=["Tops", "Bottoms"]),
        min_reviews=50,
    )
    assert [i.what for i in full] != [i.what for i in scoped]


@pytest.mark.parametrize("builder", INSIGHT_BUILDERS)
def test_insights_degrade_gracefully_on_an_empty_frame(builder, clean) -> None:
    insights = builder(clean.iloc[0:0], min_reviews=50)
    assert len(insights) == 1
    assert "Insufficient data" in insights[0].title


@pytest.mark.parametrize("builder", INSIGHT_BUILDERS)
def test_insights_are_deterministic(builder, clean) -> None:
    """No randomness and no generated prose - two runs must be identical."""
    first = [i.as_dict() for i in builder(clean, min_reviews=50)]
    second = [i.as_dict() for i in builder(clean, min_reviews=50)]
    assert first == second


def test_executive_insight_names_the_real_worst_department(clean: pd.DataFrame) -> None:
    departments = analytics.department_performance(clean, min_reviews=50)
    worst = departments.sort_values("Negative Rate", ascending=False).iloc[0]
    insights = analytics.executive_insights(clean, min_reviews=50)
    match = next(
        i for i in insights if i.title == "Department with the highest negative rate"
    )
    assert str(worst["Department Name"]) in match.what
    assert f"{worst['Negative Rate']:.1f}%" in match.what


def test_product_insight_names_the_real_lowest_rated_class(clean: pd.DataFrame) -> None:
    classes = analytics.segment_performance(clean, "Class Name", min_reviews=50)
    lowest = classes.sort_values("Avg Rating", ascending=True).iloc[0]
    insights = analytics.product_insights(clean, min_reviews=50)
    match = next(
        i for i in insights if i.title == "Class with the lowest average rating"
    )
    assert str(lowest["Class Name"]) in match.what
    assert f"{lowest['Avg Rating']:.2f}" in match.what


def test_customer_insight_names_the_real_best_age_band(clean: pd.DataFrame) -> None:
    ages = analytics.age_band_performance(clean, min_reviews=50)
    best = ages.sort_values("Recommendation Rate", ascending=False).iloc[0]
    insights = analytics.customer_insights(clean, min_reviews=50)
    match = next(
        i
        for i in insights
        if i.title == "Age band with the highest recommendation rate"
    )
    assert str(best["Age Band"]) in match.what


def test_insights_avoid_causal_language(clean: pd.DataFrame) -> None:
    """The data shows association only; the wording must not claim causation."""
    forbidden = (" causes ", " caused by ", " because of the ", " proves ",
                 " will increase ", " will reduce ", " guarantees ")
    for builder in INSIGHT_BUILDERS:
        for insight in builder(clean, min_reviews=50):
            text = f" {insight.what} {insight.why} {insight.focus} ".lower()
            for phrase in forbidden:
                assert phrase not in text, f"{insight.title}: '{phrase.strip()}'"


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def test_numeric_summary_covers_the_structured_measures(clean: pd.DataFrame) -> None:
    table = analytics.numeric_summary(clean)
    assert set(table["Measure"]) == {
        "Age",
        "Rating",
        "Positive Feedback Count",
        "Review Length Words",
    }
    age_row = table[table["Measure"] == "Age"].iloc[0]
    assert age_row["Mean"] == pytest.approx(43.20, abs=0.05)
    assert age_row["Median"] == 41
    assert age_row["Count"] == len(clean)
    rating_row = table[table["Measure"] == "Rating"].iloc[0]
    assert rating_row["Min"] == 1
    assert rating_row["Max"] == 5
