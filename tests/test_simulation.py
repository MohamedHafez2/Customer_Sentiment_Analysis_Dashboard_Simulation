"""Unit tests for the Simulation & Decision Lab what-if calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import data_cleaning, data_loader, simulation


@pytest.fixture(scope="session")
def clean() -> pd.DataFrame:
    raw, _ = data_loader.load_default_dataset()
    frame, _audit = data_cleaning.clean_dataset(raw)
    return frame


@pytest.fixture
def default_assumptions() -> simulation.ScenarioAssumptions:
    return simulation.ScenarioAssumptions()


# --------------------------------------------------------------------------- #
# 1. Default simulation calculations
# --------------------------------------------------------------------------- #
def test_default_simulation_calculations(
    clean: pd.DataFrame, default_assumptions: simulation.ScenarioAssumptions
) -> None:
    result = simulation.run_simulation(clean, assumptions=default_assumptions)
    assert result.n == 23_486
    assert result.neg == 2_407
    assert result.neu == 2_871
    assert result.pos == 18_208

    # REACHED_NEG = round(2407 * 0.60) = 1444
    assert result.reached_neg == round(2407 * 0.60)
    # SUCCESSFUL_NEG = round(1444 * 0.50)
    assert result.successful_neg == round(result.reached_neg * 0.50)
    assert result.neg_to_pos == round(result.successful_neg * 0.70)
    assert result.neg_to_neu == result.successful_neg - result.neg_to_pos
    assert result.neu_to_pos == round(2871 * 0.20)
    assert result.sim_neg == result.neg - result.successful_neg
    assert result.sim_neu == result.neu + result.neg_to_neu - result.neu_to_pos
    assert result.sim_pos == result.pos + result.neg_to_pos + result.neu_to_pos
    assert result.advocacy_conversions == round(4172 * 0.25)
    assert result.sim_recommended == result.recommended + result.advocacy_conversions
    assert result.sim_avg_rating > result.avg_rating
    assert result.successfully_improved_reviews == (
        result.successful_neg + result.neu_to_pos
    )


# --------------------------------------------------------------------------- #
# 2. Zero-intervention scenario
# --------------------------------------------------------------------------- #
def test_zero_intervention_matches_baseline(clean: pd.DataFrame) -> None:
    zero = simulation.ScenarioAssumptions(
        negative_reached_pct=0,
        intervention_success_pct=0,
        neg_to_pos_of_success_pct=0,
        neu_to_pos_pct=0,
        advocacy_conversion_pct=0,
    )
    result = simulation.run_simulation(clean, assumptions=zero)
    assert result.successful_neg == 0
    assert result.neg_to_pos == 0
    assert result.neg_to_neu == 0
    assert result.neu_to_pos == 0
    assert result.advocacy_conversions == 0
    assert result.sim_neg == result.neg
    assert result.sim_neu == result.neu
    assert result.sim_pos == result.pos
    assert result.sim_recommended == result.recommended
    assert result.sim_not_recommended == result.not_recommended
    assert result.sim_avg_rating == pytest.approx(result.avg_rating, abs=1e-12)
    assert result.sim_positive_rate == pytest.approx(
        result.baseline_positive_rate, abs=1e-12
    )
    assert result.sim_recommendation_rate == pytest.approx(
        result.baseline_recommendation_rate, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# 3. 100% intervention scenario
# --------------------------------------------------------------------------- #
def test_full_intervention_scenario(clean: pd.DataFrame) -> None:
    full = simulation.ScenarioAssumptions(
        negative_reached_pct=100,
        intervention_success_pct=100,
        neg_to_pos_of_success_pct=100,
        neu_to_pos_pct=100,
        advocacy_conversion_pct=100,
    )
    result = simulation.run_simulation(clean, assumptions=full)
    assert result.reached_neg == result.neg
    assert result.successful_neg == result.neg
    assert result.neg_to_pos == result.neg
    assert result.neg_to_neu == 0
    assert result.neu_to_pos == result.neu
    assert result.sim_neg == 0
    assert result.sim_neu == 0
    assert result.sim_pos == result.n
    assert result.sim_not_recommended == 0
    assert result.sim_recommended == result.n
    assert result.sim_negative_rate == 0.0
    assert result.sim_positive_rate == pytest.approx(100.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. Empty filtered dataset
# --------------------------------------------------------------------------- #
def test_empty_filtered_dataset(
    clean: pd.DataFrame, default_assumptions: simulation.ScenarioAssumptions
) -> None:
    empty = clean.iloc[0:0]
    result = simulation.run_simulation(empty, assumptions=default_assumptions)
    assert result.n == 0
    assert result.sim_neg == 0
    assert result.sim_neu == 0
    assert result.sim_pos == 0
    assert result.sim_avg_rating == 0.0
    assert result.sim_recommendation_rate == 0.0
    assert simulation.finite_kpis(result)
    table = simulation.department_simulation_table(empty, assumptions=default_assumptions)
    assert table.empty
    insights = simulation.simulation_insights(result, table)
    assert len(insights) >= 1


# --------------------------------------------------------------------------- #
# 5. Scope with zero negative reviews
# --------------------------------------------------------------------------- #
def test_zero_negative_reviews(
    clean: pd.DataFrame, default_assumptions: simulation.ScenarioAssumptions
) -> None:
    positive_only = clean[clean["Sentiment"].astype(str) == "Positive"].copy()
    assert len(positive_only) > 0
    assert int((positive_only["Sentiment"].astype(str) == "Negative").sum()) == 0
    result = simulation.run_simulation(
        positive_only, assumptions=default_assumptions
    )
    assert result.neg == 0
    assert result.successful_neg == 0
    assert result.neg_to_pos == 0
    assert result.neg_to_neu == 0
    assert result.sim_neg == 0
    assert result.sim_neg + result.sim_neu + result.sim_pos == result.n
    assert simulation.finite_kpis(result)


# --------------------------------------------------------------------------- #
# 6. Scope with zero neutral reviews
# --------------------------------------------------------------------------- #
def test_zero_neutral_reviews(
    clean: pd.DataFrame, default_assumptions: simulation.ScenarioAssumptions
) -> None:
    no_neutral = clean[clean["Sentiment"].astype(str) != "Neutral"].copy()
    assert int((no_neutral["Sentiment"].astype(str) == "Neutral").sum()) == 0
    result = simulation.run_simulation(no_neutral, assumptions=default_assumptions)
    assert result.neu == 0
    assert result.neu_to_pos == 0
    assert result.sim_neg + result.sim_neu + result.sim_pos == result.n
    assert simulation.finite_kpis(result)


# --------------------------------------------------------------------------- #
# 7. Sentiment-count reconciliation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "assumptions",
    [
        simulation.ScenarioAssumptions(),
        simulation.ScenarioAssumptions(0, 0, 0, 0, 0),
        simulation.ScenarioAssumptions(100, 100, 100, 100, 100),
        simulation.ScenarioAssumptions(33, 67, 25, 40, 10),
        simulation.ScenarioAssumptions(1, 1, 1, 1, 1),
    ],
)
def test_sentiment_count_reconciliation(
    clean: pd.DataFrame, assumptions: simulation.ScenarioAssumptions
) -> None:
    result = simulation.run_simulation(clean, assumptions=assumptions)
    assert result.sim_neg + result.sim_neu + result.sim_pos == result.n
    assert result.sim_neg >= 0
    assert result.sim_neu >= 0
    assert result.sim_pos >= 0


# --------------------------------------------------------------------------- #
# 8. Recommendation-count reconciliation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "assumptions",
    [
        simulation.ScenarioAssumptions(),
        simulation.ScenarioAssumptions(0, 0, 0, 0, 0),
        simulation.ScenarioAssumptions(100, 100, 50, 50, 100),
        simulation.ScenarioAssumptions(40, 40, 40, 40, 40),
    ],
)
def test_recommendation_count_reconciliation(
    clean: pd.DataFrame, assumptions: simulation.ScenarioAssumptions
) -> None:
    result = simulation.run_simulation(clean, assumptions=assumptions)
    assert result.recommended + result.not_recommended == result.n
    assert result.sim_recommended + result.sim_not_recommended == result.n
    assert result.sim_recommended == result.recommended + result.advocacy_conversions
    assert result.sim_not_recommended == (
        result.not_recommended - result.advocacy_conversions
    )


# --------------------------------------------------------------------------- #
# 9. Department-level calculations
# --------------------------------------------------------------------------- #
def test_department_level_calculations(
    clean: pd.DataFrame, default_assumptions: simulation.ScenarioAssumptions
) -> None:
    table = simulation.department_simulation_table(
        clean, assumptions=default_assumptions, min_reviews=0
    )
    assert not table.empty
    assert set(clean["Department Name"].astype(str).unique()).issubset(
        set(table["Department"])
    )
    assert table["Reviews"].sum() == len(clean)
    for _, row in table.iterrows():
        dept = clean[clean["Department Name"].astype(str) == row["Department"]]
        expected = simulation.run_simulation(dept, assumptions=default_assumptions)
        assert int(row["Reviews"]) == expected.n
        assert int(row["Successful Negative Cases"]) == expected.successful_neg
        assert int(row["Remaining Negative Reviews"]) == expected.sim_neg
        assert float(row["Simulated Negative Rate"]) == pytest.approx(
            expected.sim_negative_rate, abs=1e-9
        )

    impact = simulation.department_impact_chart_data(
        clean, assumptions=default_assumptions, min_reviews=50
    )
    assert not impact.empty
    assert (impact["Reviews"] >= 50).all()
    assert impact["Negative Reduction"].is_monotonic_decreasing or len(impact) == 1


# --------------------------------------------------------------------------- #
# 10. Priority classification
# --------------------------------------------------------------------------- #
def test_priority_classification_rules() -> None:
    assert simulation.classify_priority(12.0, 8.0, 10.0) == "High"
    assert simulation.classify_priority(12.0, 8.0, 3.0) == "Medium"
    assert simulation.classify_priority(5.0, 8.0, 10.0) == "Medium"
    assert simulation.classify_priority(5.0, 8.0, 3.0) == "Low"
    # Boundary: equal rate is not "above"
    assert simulation.classify_priority(8.0, 8.0, 10.0) == "Medium"
    assert simulation.classify_priority(8.0, 8.0, 4.9) == "Low"
    assert simulation.classify_priority(9.0, 8.0, 5.0) == "High"


def test_priority_on_department_table(
    clean: pd.DataFrame, default_assumptions: simulation.ScenarioAssumptions
) -> None:
    table = simulation.department_simulation_table(
        clean, assumptions=default_assumptions
    )
    overall = simulation.run_simulation(clean, assumptions=default_assumptions)
    for _, row in table.iterrows():
        expected = simulation.classify_priority(
            float(row["Simulated Negative Rate"]),
            overall.sim_negative_rate,
            float(row["Share %"]),
        )
        assert row["Scenario Priority"] == expected


# --------------------------------------------------------------------------- #
# 11. No NaN or infinite KPI values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "frame_builder",
    [
        lambda clean: clean,
        lambda clean: clean.iloc[0:0],
        lambda clean: clean[clean["Sentiment"].astype(str) == "Positive"],
        lambda clean: clean[clean["Sentiment"].astype(str) != "Neutral"],
        lambda clean: clean[clean["Department Name"].astype(str) == "Trend"],
    ],
)
def test_no_nan_or_infinite_kpi_values(
    clean: pd.DataFrame, frame_builder
) -> None:
    frame = frame_builder(clean)
    for assumptions in (
        simulation.ScenarioAssumptions(),
        simulation.ScenarioAssumptions(0, 0, 0, 0, 0),
        simulation.ScenarioAssumptions(100, 100, 100, 100, 100),
    ):
        result = simulation.run_simulation(frame, assumptions=assumptions)
        assert simulation.finite_kpis(result)
        for value in result.to_dict().values():
            if isinstance(value, (float, int, np.floating, np.integer)):
                assert math.isfinite(float(value)), value


def test_resolve_scope_and_categories(clean: pd.DataFrame) -> None:
    depts = simulation.available_categories(clean, "By Department")
    assert "Tops" in depts or "Dresses" in depts or len(depts) > 0
    scoped = simulation.resolve_scope(clean, "By Department", depts[:1])
    assert len(scoped) > 0
    assert set(scoped["Department Name"].astype(str).unique()) == {depts[0]}
    all_scope = simulation.resolve_scope(clean, "All filtered data", depts)
    assert len(all_scope) == len(clean)


def test_chart_frames_shapes(
    clean: pd.DataFrame, default_assumptions: simulation.ScenarioAssumptions
) -> None:
    result = simulation.run_simulation(clean, assumptions=default_assumptions)
    sentiment = simulation.sentiment_comparison_frame(result)
    assert len(sentiment) == 6
    kpi = simulation.kpi_improvement_frame(result)
    assert list(kpi["Metric"]) == [
        "Average Rating",
        "Positive Rate",
        "Negative Rate",
        "Recommendation Rate",
    ]
    movement = simulation.movement_waterfall_frame(result)
    assert len(movement) == 4
    assert int(movement["Count"].sum()) >= result.successful_neg
