"""Deterministic what-if simulation for the Simulation & Decision Lab tab.

All outputs are scenario estimates driven by user-defined intervention rates
applied to the currently filtered review frame. Nothing here is a prediction,
a trained model, or an observed future outcome.

Sentiment uses the same rating-based classification as the rest of the
dashboard (1-2 Negative, 3 Neutral, 4-5 Positive).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from html import escape
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .analytics import _as_str, _rate

# --------------------------------------------------------------------------- #
# Defaults and labels
# --------------------------------------------------------------------------- #
SIM_DEFAULTS: dict[str, Any] = {
    "sim_scope": "All filtered data",
    "sim_categories": [],
    "sim_neg_reached": 60,
    "sim_success_rate": 50,
    "sim_neg_to_pos": 70,
    "sim_neu_to_pos": 20,
    "sim_advocacy": 25,
}

SCOPE_OPTIONS: tuple[str, ...] = (
    "All filtered data",
    "By Department",
    "By Division",
    "By Product Class",
)

SCOPE_COLUMN: dict[str, str | None] = {
    "All filtered data": None,
    "By Department": "Department Name",
    "By Division": "Division Name",
    "By Product Class": "Class Name",
}

PRIORITY_ORDER = ("High", "Medium", "Low")


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScenarioAssumptions:
    """User-defined intervention rates (percentages, 0-100)."""

    negative_reached_pct: float = 60.0
    intervention_success_pct: float = 50.0
    neg_to_pos_of_success_pct: float = 70.0
    neu_to_pos_pct: float = 20.0
    advocacy_conversion_pct: float = 25.0

    def as_fractions(self) -> dict[str, float]:
        return {
            "negative_reached": self.negative_reached_pct / 100.0,
            "intervention_success": self.intervention_success_pct / 100.0,
            "neg_to_pos_of_success": self.neg_to_pos_of_success_pct / 100.0,
            "neu_to_pos": self.neu_to_pos_pct / 100.0,
            "advocacy_conversion": self.advocacy_conversion_pct / 100.0,
        }


@dataclass
class SimulationResult:
    """Baseline and simulated KPIs for one scope, plus movement counts."""

    n: int = 0
    # Baseline sentiment
    neg: int = 0
    neu: int = 0
    pos: int = 0
    # Baseline recommendation
    recommended: int = 0
    not_recommended: int = 0
    # Baseline rating
    avg_rating: float = 0.0
    rating_sum: float = 0.0
    # Movement
    reached_neg: int = 0
    successful_neg: int = 0
    neg_to_pos: int = 0
    neg_to_neu: int = 0
    neu_to_pos: int = 0
    advocacy_conversions: int = 0
    # Simulated sentiment
    sim_neg: int = 0
    sim_neu: int = 0
    sim_pos: int = 0
    # Simulated recommendation
    sim_recommended: int = 0
    sim_not_recommended: int = 0
    # Simulated rating
    sim_avg_rating: float = 0.0
    sim_rating_sum: float = 0.0
    # Rates (percent)
    baseline_positive_rate: float = 0.0
    baseline_neutral_rate: float = 0.0
    baseline_negative_rate: float = 0.0
    baseline_recommendation_rate: float = 0.0
    sim_positive_rate: float = 0.0
    sim_neutral_rate: float = 0.0
    sim_negative_rate: float = 0.0
    sim_recommendation_rate: float = 0.0
    # Meta
    scope_label: str = "All filtered data"
    assumptions: dict[str, float] = field(default_factory=dict)

    @property
    def successfully_improved_reviews(self) -> int:
        """Negative cases successfully improved plus neutrals moved to positive."""
        return int(self.successful_neg + self.neu_to_pos)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["successfully_improved_reviews"] = self.successfully_improved_reviews
        return payload


@dataclass
class SimulationInsight:
    """Rule-based scenario insight (not LLM-generated)."""

    title: str
    what: str
    why: str
    action: str
    tone: str = "neutral"


# --------------------------------------------------------------------------- #
# Scope helpers
# --------------------------------------------------------------------------- #
def resolve_scope(
    df: pd.DataFrame,
    scope: str,
    categories: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Restrict the filtered frame to the selected simulation scope."""
    if df is None or df.empty:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()

    column = SCOPE_COLUMN.get(scope)
    if column is None:
        return df.copy()

    selected = [str(c) for c in (categories or []) if str(c).strip()]
    if not selected:
        return df.copy()

    if column not in df.columns:
        return df.iloc[0:0].copy()

    mask = _as_str(df[column]).isin(selected)
    return df.loc[mask].copy()


def available_categories(df: pd.DataFrame, scope: str) -> list[str]:
    """Sorted category labels present in the current filtered frame."""
    column = SCOPE_COLUMN.get(scope)
    if column is None or df is None or df.empty or column not in df.columns:
        return []
    values = _as_str(df[column]).dropna().unique().tolist()
    return sorted(str(v) for v in values if str(v).strip())


def scope_description(
    scope: str,
    categories: Sequence[str] | None,
    n: int,
) -> str:
    """Human-readable scope line for the purpose panel."""
    if scope == "All filtered data" or not categories:
        return f"All filtered data ({n:,} reviews)"
    preview = ", ".join(str(c) for c in list(categories)[:4])
    extra = f" +{len(categories) - 4} more" if len(categories) > 4 else ""
    return f"{scope}: {preview}{extra} ({n:,} reviews)"


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #
def _empty_result(
    scope_label: str = "All filtered data",
    assumptions: ScenarioAssumptions | None = None,
) -> SimulationResult:
    assumptions = assumptions or ScenarioAssumptions()
    return SimulationResult(
        scope_label=scope_label,
        assumptions={
            "negative_reached_pct": assumptions.negative_reached_pct,
            "intervention_success_pct": assumptions.intervention_success_pct,
            "neg_to_pos_of_success_pct": assumptions.neg_to_pos_of_success_pct,
            "neu_to_pos_pct": assumptions.neu_to_pos_pct,
            "advocacy_conversion_pct": assumptions.advocacy_conversion_pct,
        },
    )


def run_simulation(
    df: pd.DataFrame,
    assumptions: ScenarioAssumptions | None = None,
    scope_label: str = "All filtered data",
) -> SimulationResult:
    """Apply deterministic what-if rates to one review frame.

    Sentiment-count identity always holds: ``SIM_NEG + SIM_NEU + SIM_POS = N``.
    Recommendation identity always holds:
    ``SIM_RECOMMENDED + SIM_NOT_RECOMMENDED = N`` when recommendation labels are
    fully populated (as in the cleaned dataset).
    """
    assumptions = assumptions or ScenarioAssumptions()
    if df is None or len(df) == 0:
        return _empty_result(scope_label=scope_label, assumptions=assumptions)

    fracs = assumptions.as_fractions()
    n = int(len(df))

    sentiment = _as_str(df["Sentiment"])
    neg = int((sentiment == "Negative").sum())
    neu = int((sentiment == "Neutral").sum())
    pos = int((sentiment == "Positive").sum())
    # Guard: unclassified rows (should not occur after cleaning) stay Positive-
    # bucketed only for rate denominators by using classified = neg+neu+pos.
    classified = neg + neu + pos
    if classified != n:
        # Keep totals equal to N by attributing residual to Positive.
        pos = n - neg - neu

    ratings = pd.to_numeric(df["Rating"], errors="coerce")
    rating_sum = float(ratings.fillna(0).sum())
    avg_rating = float(ratings.mean()) if ratings.notna().any() else 0.0

    reco = pd.to_numeric(df["Recommended IND"], errors="coerce")
    recommended = int((reco == 1).sum())
    not_recommended = int((reco == 0).sum())
    # If any missing recommendation labels exist, keep N identity by treating
    # residual as recommended for simulation bookkeeping only when both sides
    # already sum to less than N (rare after cleaning).
    reco_known = recommended + not_recommended
    if reco_known < n and reco_known > 0:
        recommended = n - not_recommended
    elif reco_known == 0:
        recommended = 0
        not_recommended = 0

    reached_neg = int(round(neg * fracs["negative_reached"]))
    reached_neg = max(0, min(reached_neg, neg))
    successful_neg = int(round(reached_neg * fracs["intervention_success"]))
    successful_neg = max(0, min(successful_neg, reached_neg))
    neg_to_pos = int(round(successful_neg * fracs["neg_to_pos_of_success"]))
    neg_to_pos = max(0, min(neg_to_pos, successful_neg))
    neg_to_neu = successful_neg - neg_to_pos
    neu_to_pos = int(round(neu * fracs["neu_to_pos"]))
    neu_to_pos = max(0, min(neu_to_pos, neu))

    sim_neg = neg - successful_neg
    sim_neu = neu + neg_to_neu - neu_to_pos
    sim_pos = pos + neg_to_pos + neu_to_pos

    # Hard reconciliation: totals must equal N (rounding / edge-case guard).
    total_sim = sim_neg + sim_neu + sim_pos
    if total_sim != n:
        sim_pos += n - total_sim

    advocacy_conversions = int(round(not_recommended * fracs["advocacy_conversion"]))
    advocacy_conversions = max(0, min(advocacy_conversions, not_recommended))
    sim_recommended = recommended + advocacy_conversions
    sim_not_recommended = not_recommended - advocacy_conversions
    if recommended + not_recommended == n:
        # Keep identity when baseline recommendation is complete.
        if sim_recommended + sim_not_recommended != n:
            sim_not_recommended = n - sim_recommended

    sim_rating_sum = _simulated_rating_sum(
        df,
        successful_neg=successful_neg,
        neg_to_pos=neg_to_pos,
        neg_to_neu=neg_to_neu,
        neu_to_pos=neu_to_pos,
        baseline_sum=rating_sum,
    )
    sim_avg_rating = float(sim_rating_sum / n) if n else 0.0

    denom = float(n) if n else 0.0
    return SimulationResult(
        n=n,
        neg=neg,
        neu=neu,
        pos=pos,
        recommended=recommended,
        not_recommended=not_recommended,
        avg_rating=avg_rating,
        rating_sum=rating_sum,
        reached_neg=reached_neg,
        successful_neg=successful_neg,
        neg_to_pos=neg_to_pos,
        neg_to_neu=neg_to_neu,
        neu_to_pos=neu_to_pos,
        advocacy_conversions=advocacy_conversions,
        sim_neg=sim_neg,
        sim_neu=sim_neu,
        sim_pos=sim_pos,
        sim_recommended=sim_recommended,
        sim_not_recommended=sim_not_recommended,
        sim_avg_rating=sim_avg_rating,
        sim_rating_sum=sim_rating_sum,
        baseline_positive_rate=_rate(pos, n),
        baseline_neutral_rate=_rate(neu, n),
        baseline_negative_rate=_rate(neg, n),
        baseline_recommendation_rate=_rate(recommended, n) if n else 0.0,
        sim_positive_rate=_rate(sim_pos, n),
        sim_neutral_rate=_rate(sim_neu, n),
        sim_negative_rate=_rate(sim_neg, n),
        sim_recommendation_rate=_rate(sim_recommended, n) if n else 0.0,
        scope_label=scope_label,
        assumptions={
            "negative_reached_pct": assumptions.negative_reached_pct,
            "intervention_success_pct": assumptions.intervention_success_pct,
            "neg_to_pos_of_success_pct": assumptions.neg_to_pos_of_success_pct,
            "neu_to_pos_pct": assumptions.neu_to_pos_pct,
            "advocacy_conversion_pct": assumptions.advocacy_conversion_pct,
        },
    )


def _simulated_rating_sum(
    df: pd.DataFrame,
    *,
    successful_neg: int,
    neg_to_pos: int,
    neg_to_neu: int,
    neu_to_pos: int,
    baseline_sum: float,
) -> float:
    """Replace rating contribution of converted cases only (deterministic).

    Selection order is the frame's natural index order so the same filter and
    assumptions always yield the same simulated average rating.

    * Each of the first ``neg_to_pos`` successfully improved negatives becomes 4.
    * The remaining ``neg_to_neu`` successfully improved negatives become 3.
    * Each of the first ``neu_to_pos`` neutrals moves from 3 to 4.
    * All other ratings are unchanged.
    """
    if df is None or len(df) == 0:
        return 0.0

    work = df.copy()
    work["_rating"] = pd.to_numeric(work["Rating"], errors="coerce")
    work["_sentiment"] = _as_str(work["Sentiment"])

    delta = 0.0

    if successful_neg > 0:
        neg_rows = work.loc[work["_sentiment"] == "Negative"].sort_index()
        chosen = neg_rows.iloc[:successful_neg]
        to_pos = chosen.iloc[:neg_to_pos]
        to_neu = chosen.iloc[neg_to_pos : neg_to_pos + neg_to_neu]
        if len(to_pos):
            old = to_pos["_rating"].fillna(1.0)
            delta += float((4.0 - old).sum())
        if len(to_neu):
            old = to_neu["_rating"].fillna(1.0)
            delta += float((3.0 - old).sum())

    if neu_to_pos > 0:
        neu_rows = work.loc[work["_sentiment"] == "Neutral"].sort_index()
        chosen = neu_rows.iloc[:neu_to_pos]
        if len(chosen):
            # Neutral is defined as rating 3; still use observed rating if present.
            old = chosen["_rating"].fillna(3.0)
            delta += float((4.0 - old).sum())

    return float(baseline_sum + delta)


# --------------------------------------------------------------------------- #
# Department / segment tables
# --------------------------------------------------------------------------- #
def classify_priority(
    remaining_negative_rate: float,
    overall_sim_negative_rate: float,
    review_share_pct: float,
) -> str:
    """Scenario Priority: High / Medium / Low."""
    above_overall = remaining_negative_rate > overall_sim_negative_rate
    material_volume = review_share_pct >= 5.0
    if above_overall and material_volume:
        return "High"
    if above_overall or material_volume:
        return "Medium"
    return "Low"


def department_simulation_table(
    df: pd.DataFrame,
    assumptions: ScenarioAssumptions | None = None,
    min_reviews: int = 0,
) -> pd.DataFrame:
    """One simulation row per department in the filtered frame."""
    columns = [
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
        "Improved Cases",
        "Share %",
    ]
    if df is None or df.empty or "Department Name" not in df.columns:
        return pd.DataFrame(columns=columns)

    assumptions = assumptions or ScenarioAssumptions()
    overall = run_simulation(df, assumptions=assumptions)
    total = max(int(len(df)), 1)

    rows: list[dict[str, Any]] = []
    for dept, group in df.groupby(_as_str(df["Department Name"]), sort=True):
        if len(group) < min_reviews:
            continue
        result = run_simulation(
            group, assumptions=assumptions, scope_label=str(dept)
        )
        share = _rate(result.n, total)
        priority = classify_priority(
            result.sim_negative_rate,
            overall.sim_negative_rate,
            share,
        )
        rows.append(
            {
                "Department": str(dept),
                "Reviews": result.n,
                "Baseline Average Rating": result.avg_rating,
                "Simulated Average Rating": result.sim_avg_rating,
                "Baseline Positive Rate": result.baseline_positive_rate,
                "Simulated Positive Rate": result.sim_positive_rate,
                "Baseline Negative Rate": result.baseline_negative_rate,
                "Simulated Negative Rate": result.sim_negative_rate,
                "Baseline Recommendation Rate": result.baseline_recommendation_rate,
                "Simulated Recommendation Rate": result.sim_recommendation_rate,
                "Successful Negative Cases": result.successful_neg,
                "Remaining Negative Reviews": result.sim_neg,
                "Scenario Priority": priority,
                "Improved Cases": result.successfully_improved_reviews,
                "Share %": share,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    table = pd.DataFrame(rows)
    priority_rank = table["Scenario Priority"].map(
        {p: i for i, p in enumerate(PRIORITY_ORDER)}
    )
    return (
        table.assign(_prio=priority_rank)
        .sort_values(
            ["_prio", "Remaining Negative Reviews", "Reviews"],
            ascending=[True, False, False],
        )
        .drop(columns="_prio")
        .reset_index(drop=True)
    )


def department_impact_chart_data(
    df: pd.DataFrame,
    assumptions: ScenarioAssumptions | None = None,
    min_reviews: int = 50,
) -> pd.DataFrame:
    """Departments with enough volume, ranked by expected negative-count reduction."""
    table = department_simulation_table(
        df, assumptions=assumptions, min_reviews=min_reviews
    )
    if table.empty:
        return table

    out = table.copy()
    out["Negative Reduction"] = out["Successful Negative Cases"].astype(int)
    out["Baseline Neg Rate"] = out["Baseline Negative Rate"]
    out["Simulated Neg Rate"] = out["Simulated Negative Rate"]
    return out.sort_values("Negative Reduction", ascending=False).reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- #
# Chart-ready frames
# --------------------------------------------------------------------------- #
def sentiment_comparison_frame(result: SimulationResult) -> pd.DataFrame:
    """Long-form baseline vs simulated sentiment counts and rates."""
    if result.n == 0:
        return pd.DataFrame(
            columns=["Sentiment", "Series", "Reviews", "Share %"]
        )
    rows = []
    for label, base_count, sim_count in (
        ("Negative", result.neg, result.sim_neg),
        ("Neutral", result.neu, result.sim_neu),
        ("Positive", result.pos, result.sim_pos),
    ):
        rows.append(
            {
                "Sentiment": label,
                "Series": "Dataset Baseline",
                "Reviews": base_count,
                "Share %": _rate(base_count, result.n),
            }
        )
        rows.append(
            {
                "Sentiment": label,
                "Series": "Simulated Scenario",
                "Reviews": sim_count,
                "Share %": _rate(sim_count, result.n),
            }
        )
    return pd.DataFrame(rows)


def kpi_improvement_frame(result: SimulationResult) -> pd.DataFrame:
    """Metrics for the dual-axis / subplot KPI comparison chart."""
    return pd.DataFrame(
        [
            {
                "Metric": "Average Rating",
                "Kind": "rating",
                "Baseline": result.avg_rating,
                "Simulated": result.sim_avg_rating,
                "Change": result.sim_avg_rating - result.avg_rating,
            },
            {
                "Metric": "Positive Rate",
                "Kind": "percent",
                "Baseline": result.baseline_positive_rate,
                "Simulated": result.sim_positive_rate,
                "Change": result.sim_positive_rate - result.baseline_positive_rate,
            },
            {
                "Metric": "Negative Rate",
                "Kind": "percent",
                "Baseline": result.baseline_negative_rate,
                "Simulated": result.sim_negative_rate,
                "Change": result.sim_negative_rate - result.baseline_negative_rate,
            },
            {
                "Metric": "Recommendation Rate",
                "Kind": "percent",
                "Baseline": result.baseline_recommendation_rate,
                "Simulated": result.sim_recommendation_rate,
                "Change": result.sim_recommendation_rate
                - result.baseline_recommendation_rate,
            },
        ]
    )


def movement_waterfall_frame(result: SimulationResult) -> pd.DataFrame:
    """Review-movement steps for the waterfall chart."""
    return pd.DataFrame(
        [
            {
                "Step": "Negative cases improved",
                "Count": result.successful_neg,
                "Direction": "outflow",
            },
            {
                "Step": "Negative to Neutral",
                "Count": result.neg_to_neu,
                "Direction": "shift",
            },
            {
                "Step": "Negative to Positive",
                "Count": result.neg_to_pos,
                "Direction": "shift",
            },
            {
                "Step": "Neutral to Positive",
                "Count": result.neu_to_pos,
                "Direction": "shift",
            },
        ]
    )


# --------------------------------------------------------------------------- #
# Insights (rule-based)
# --------------------------------------------------------------------------- #
def simulation_insights(
    result: SimulationResult,
    department_table: pd.DataFrame,
) -> list[SimulationInsight]:
    """Build 3-5 deterministic insight cards from calculated scenario results."""
    if result.n == 0:
        return [
            SimulationInsight(
                title="No reviews in simulation scope",
                what="The current filter and scope selection leave zero reviews "
                "to simulate.",
                why="Without a review base, scenario rates cannot be estimated.",
                action="Management could consider widening global filters or "
                "selecting additional categories in the simulation scope.",
                tone="neutral",
            )
        ]

    insights: list[SimulationInsight] = []

    neg_reduction_pp = result.baseline_negative_rate - result.sim_negative_rate
    insights.append(
        SimulationInsight(
            title="Expected negative-rate reduction",
            what=(
                f"The scenario estimates the negative rate could move from "
                f"<b>{result.baseline_negative_rate:.1f}%</b> to "
                f"<b>{result.sim_negative_rate:.1f}%</b> "
                f"({neg_reduction_pp:+.1f} pp), after successfully improving "
                f"<b>{result.successful_neg:,}</b> of "
                f"<b>{result.neg:,}</b> baseline negative reviews."
            ),
            why="A lower share of low-star reviews on product pages is associated "
            "with a healthier perceived quality signal for shoppers still deciding.",
            action="Management could consider prioritising departments with both "
            "high volume and high negative counts for investigation and service "
            "recovery.",
            tone="positive" if neg_reduction_pp > 0 else "neutral",
        )
    )

    pos_increase_pp = result.sim_positive_rate - result.baseline_positive_rate
    insights.append(
        SimulationInsight(
            title="Expected positive-rate increase",
            what=(
                f"Under the selected assumptions, the positive rate could rise "
                f"from <b>{result.baseline_positive_rate:.1f}%</b> to "
                f"<b>{result.sim_positive_rate:.1f}%</b> "
                f"({pos_increase_pp:+.1f} pp), combining "
                f"<b>{result.neg_to_pos:,}</b> negative-to-positive and "
                f"<b>{result.neu_to_pos:,}</b> neutral-to-positive movements."
            ),
            why="Positive reviews reinforce advocacy and are the most common "
            "sentiment class in this dataset, so even modest shifts change the "
            "headline mix customers see.",
            action="Management could consider improving product descriptions and "
            "sizing guidance for categories that currently generate many neutral "
            "or negative ratings.",
            tone="positive" if pos_increase_pp > 0 else "neutral",
        )
    )

    reco_increase_pp = (
        result.sim_recommendation_rate - result.baseline_recommendation_rate
    )
    insights.append(
        SimulationInsight(
            title="Expected recommendation-rate increase",
            what=(
                f"The scenario estimates recommendation rate could move from "
                f"<b>{result.baseline_recommendation_rate:.1f}%</b> to "
                f"<b>{result.sim_recommendation_rate:.1f}%</b> "
                f"({reco_increase_pp:+.1f} pp) if "
                f"<b>{result.advocacy_conversions:,}</b> of "
                f"<b>{result.not_recommended:,}</b> not-recommended cases convert."
            ),
            why="Recommendation is the customer's stated advocacy intent in this "
            "dataset and is tracked separately from rating-based sentiment.",
            action="Management could consider routing recovered customers to "
            "Customer Service follow-up and CRM retention outreach after a "
            "successful intervention.",
            tone="positive" if reco_increase_pp > 0 else "neutral",
        )
    )

    if not department_table.empty:
        addressable = department_table.sort_values(
            "Successful Negative Cases", ascending=False
        ).iloc[0]
        insights.append(
            SimulationInsight(
                title="Largest addressable negative volume",
                what=(
                    f"Under the selected assumptions, <b>{escape(str(addressable['Department']))}</b> "
                    f"shows the largest number of successfully improved negative "
                    f"cases (<b>{int(addressable['Successful Negative Cases']):,}</b> "
                    f"of <b>{int(addressable['Reviews']):,}</b> reviews)."
                ),
                why="Absolute volume determines how many customers can be reached "
                "with the same intervention rates; high-volume pockets move overall "
                "KPIs more than small categories.",
                action="Management could consider reviewing quality complaints with "
                "Product and Supplier Management for this department first.",
                tone="neutral",
            )
        )

        risk = department_table.sort_values(
            ["Simulated Negative Rate", "Remaining Negative Reviews"],
            ascending=[False, False],
        ).iloc[0]
        insights.append(
            SimulationInsight(
                title="Highest remaining simulated risk",
                what=(
                    f"After the scenario, <b>{escape(str(risk['Department']))}</b> "
                    f"retains the highest simulated negative rate at "
                    f"<b>{float(risk['Simulated Negative Rate']):.1f}%</b> "
                    f"({int(risk['Remaining Negative Reviews']):,} remaining "
                    f"negative reviews; priority "
                    f"<b>{escape(str(risk['Scenario Priority']))}</b>)."
                ),
                why="Categories that stay elevated after a uniform intervention "
                "assumption may need deeper product or assortment review beyond "
                "standard service recovery.",
                action="Management could consider investigating high-negative-rate "
                "product classes with the Department Manager and E-commerce "
                "Content Team.",
                tone="negative",
            )
        )

    return insights[:5]


# --------------------------------------------------------------------------- #
# KPI delta helpers for UI
# --------------------------------------------------------------------------- #
def kpi_delta(
    simulated: float,
    baseline: float,
    *,
    higher_is_better: bool,
    is_percent: bool = True,
) -> dict[str, Any]:
    """Absolute change, direction marker, and accent colour for a simulated KPI."""
    change = float(simulated) - float(baseline)
    if abs(change) < 1e-12:
        direction = "unchanged"
        accent = "neutral"
        arrow = "●"
    elif (change > 0 and higher_is_better) or (change < 0 and not higher_is_better):
        direction = "improved"
        accent = "positive"
        arrow = "▲"
    else:
        direction = "deteriorated"
        accent = "negative"
        arrow = "▼"

    if is_percent:
        change_text = f"{change:+.2f} pp"
        value_text = f"{simulated:.2f}%"
    else:
        change_text = f"{change:+.3f}"
        value_text = f"{simulated:.3f}"

    return {
        "value_text": value_text,
        "change": change,
        "change_text": change_text,
        "direction": direction,
        "accent": accent,
        "arrow": arrow,
        "label": "Scenario Estimate",
    }


def finite_kpis(result: SimulationResult) -> bool:
    """True when every numeric KPI is finite (no NaN / Inf)."""
    values = [
        result.avg_rating,
        result.sim_avg_rating,
        result.baseline_positive_rate,
        result.sim_positive_rate,
        result.baseline_neutral_rate,
        result.sim_neutral_rate,
        result.baseline_negative_rate,
        result.sim_negative_rate,
        result.baseline_recommendation_rate,
        result.sim_recommendation_rate,
    ]
    arr = np.asarray(values, dtype=float)
    return bool(np.isfinite(arr).all())


def methodology_markdown() -> str:
    """Documented formulas for the expandable methodology section."""
    return """
### Methodology

- **Rating-based sentiment classification**: Negative = Rating 1–2, Neutral = Rating 3, Positive = Rating 4–5 (same definition as the rest of this dashboard).
- **Deterministic what-if calculations**: every count uses `round()` on the selected percentage assumptions; there is no randomness and no model training.
- **Current filtered data** (and optional category scope) is the baseline; global sidebar filters apply first.
- **No model training and no external data** are used.

### Core formulas

Let `N`, `NEG`, `NEU`, `POS` be review counts in the simulation scope.

- `REACHED_NEG = round(NEG × Negative Reviews Reached %)`
- `SUCCESSFUL_NEG = round(REACHED_NEG × Intervention Success Rate %)`
- `NEG_TO_POS = round(SUCCESSFUL_NEG × Successful Negative Cases Moved to Positive %)`
- `NEG_TO_NEU = SUCCESSFUL_NEG − NEG_TO_POS`
- `NEU_TO_POS = round(NEU × Neutral-to-Positive Conversion Rate %)`
- `SIM_NEG = NEG − SUCCESSFUL_NEG`
- `SIM_NEU = NEU + NEG_TO_NEU − NEU_TO_POS`
- `SIM_POS = POS + NEG_TO_POS + NEU_TO_POS`
- Identity: `SIM_NEG + SIM_NEU + SIM_POS = N`

Recommendation:

- `ADVOCACY_CONVERSIONS = round(NOT_RECOMMENDED × Advocacy Conversion Rate %)`
- `SIM_RECOMMENDED = recommended + ADVOCACY_CONVERSIONS`
- `SIM_NOT_RECOMMENDED = NOT_RECOMMENDED − ADVOCACY_CONVERSIONS`

Simulated average rating (deterministic replacement of converted cases only):

- Successfully improved negatives selected in stable index order: first `NEG_TO_POS` become rating **4**, remaining `NEG_TO_NEU` become rating **3**.
- First `NEU_TO_POS` neutrals move from their observed rating (typically 3) to **4**.
- Unchanged records keep their existing ratings.
- `SIM_AVG_RATING = (baseline rating sum + Σ(new − old) for converted cases) / N`

### Assumptions

- Intervention rates are **user-defined management assumptions**.
- Successfully improved reviews move according to the selected conversion settings.
- The same assumptions are applied consistently across departments in comparative tables.
- Results represent a **scenario**, not observed future behaviour.

### Limitations

- The dataset contains **no review date**, so this is not a time forecast.
- Rating-based sentiment does not capture all language nuances in review text.
- Recommendation and sentiment are **separate** measures.
- Small categories can produce unstable percentages.
- No financial ROI is estimated without verified cost and revenue inputs.
- The scenario does **not** prove that an intervention causes the simulated outcome.
"""
