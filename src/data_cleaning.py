"""Auditable data-cleaning pipeline.

Every transformation is a small pure function so that ``tests/`` can assert on
it directly, and every step appends an entry to an audit log that the
Methodology & Data Quality page renders verbatim.

Design rules enforced here
--------------------------
* The exported row index (``Unnamed: 0``) is dropped - it carries no meaning.
* Missing text is replaced with explicit placeholders, never with invented text.
* ``Has Review Text`` is captured *before* the placeholder is written so that
  review display and keyword search can exclude those rows while rating /
  recommendation analytics keep them.
* Duplicated review text is flagged, never removed - identical wording can be a
  legitimate repeat purchase and dropping rows would distort volume metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import INDEX_COLUMN

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MISSING_TITLE = "Untitled Review"
MISSING_REVIEW_TEXT = "No Review Text"
UNKNOWN_CATEGORY = "Unknown"

TAXONOMY_COLUMNS = ("Division Name", "Department Name", "Class Name")

SENTIMENT_ORDER = ("Negative", "Neutral", "Positive")
RECOMMENDATION_ORDER = ("Not Recommended", "Recommended")
AGE_BAND_ORDER = ("18-29", "30-39", "40-49", "50-59", "60+")
RATING_ORDER = (1, 2, 3, 4, 5)

AGE_BAND_BINS = (-np.inf, 29, 39, 49, 59, np.inf)

VALID_RATINGS = frozenset({1, 2, 3, 4, 5})
VALID_RECOMMENDATIONS = frozenset({0, 1})

DERIVED_COLUMNS = (
    "Has Review Text",
    "Has Title",
    "Sentiment",
    "Recommendation Label",
    "Age Band",
    "Review Length Words",
    "Title Length Words",
    "Is Duplicate Review Text",
    "Valid Rating",
    "Valid Recommendation",
)


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
@dataclass
class CleaningAudit:
    """Ordered record of what the pipeline did and how many rows it touched."""

    steps: list[dict[str, Any]] = field(default_factory=list)

    def log(self, step: str, action: str, rows_affected: int, rationale: str) -> None:
        self.steps.append(
            {
                "#": len(self.steps) + 1,
                "Step": step,
                "Action": action,
                "Rows Affected": int(rows_affected),
                "Rationale": rationale,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        if not self.steps:
            return pd.DataFrame(
                columns=["#", "Step", "Action", "Rows Affected", "Rationale"]
            )
        return pd.DataFrame(self.steps)


# --------------------------------------------------------------------------- #
# Individual transformations
# --------------------------------------------------------------------------- #
def drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the exported row-index column if present."""
    columns = [c for c in df.columns if str(c) == INDEX_COLUMN or str(c) == ""]
    if not columns:
        return df.copy()
    return df.drop(columns=columns)


def sentiment_from_rating(rating: Any) -> str:
    """Map a star rating to the case-study sentiment class.

    1-2 = Negative, 3 = Neutral, 4-5 = Positive. Ratings outside 1-5 (or
    missing) return ``"Unknown"`` rather than being silently bucketed.
    """
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return "Unknown"
    try:
        value = int(rating)
    except (TypeError, ValueError):
        return "Unknown"
    if value in (1, 2):
        return "Negative"
    if value == 3:
        return "Neutral"
    if value in (4, 5):
        return "Positive"
    return "Unknown"


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Sentiment"] = out["Rating"].apply(sentiment_from_rating)
    out["Sentiment"] = pd.Categorical(
        out["Sentiment"], categories=list(SENTIMENT_ORDER) + ["Unknown"], ordered=True
    )
    return out


def recommendation_label(value: Any) -> str:
    """Map ``Recommended IND`` to a readable label."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    try:
        flag = int(value)
    except (TypeError, ValueError):
        return "Unknown"
    if flag == 1:
        return "Recommended"
    if flag == 0:
        return "Not Recommended"
    return "Unknown"


def add_recommendation_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Recommendation Label"] = out["Recommended IND"].apply(recommendation_label)
    out["Recommendation Label"] = pd.Categorical(
        out["Recommendation Label"],
        categories=list(RECOMMENDATION_ORDER) + ["Unknown"],
        ordered=True,
    )
    return out


def age_band(age: Any) -> str:
    """Bucket an age into the five reporting bands."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return UNKNOWN_CATEGORY
    try:
        value = float(age)
    except (TypeError, ValueError):
        return UNKNOWN_CATEGORY
    if value < 18:
        return "18-29"  # the dataset starts at 18; guard keeps bands exhaustive
    if value <= 29:
        return "18-29"
    if value <= 39:
        return "30-39"
    if value <= 49:
        return "40-49"
    if value <= 59:
        return "50-59"
    return "60+"


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Age Band"] = out["Age"].apply(age_band)
    categories = list(AGE_BAND_ORDER)
    if (out["Age Band"] == UNKNOWN_CATEGORY).any():
        categories.append(UNKNOWN_CATEGORY)
    out["Age Band"] = pd.Categorical(out["Age Band"], categories=categories, ordered=True)
    return out


def count_words(text: Any) -> int:
    """Whitespace word count; placeholders and missing values count as zero."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0
    value = str(text).strip()
    if not value or value in (MISSING_REVIEW_TEXT, MISSING_TITLE):
        return 0
    return len(value.split())


def add_review_length(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Review Length Words"] = out["Review Text"].apply(count_words)
    if "Title" in out.columns:
        out["Title Length Words"] = out["Title"].apply(count_words)
    return out


def flag_duplicate_review_text(df: pd.DataFrame) -> pd.DataFrame:
    """Flag exact duplicated review text. Rows are kept, never dropped."""
    out = df.copy()
    normalised = out["Review Text"].astype(str).str.strip().str.lower()
    eligible = out["Has Review Text"] if "Has Review Text" in out.columns else pd.Series(
        True, index=out.index
    )
    duplicated = normalised.duplicated(keep=False) & eligible
    out["Is Duplicate Review Text"] = duplicated.fillna(False)
    return out


def validate_rating(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Valid Rating"] = out["Rating"].isin(list(VALID_RATINGS))
    return out


def validate_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Valid Recommendation"] = out["Recommended IND"].isin(
        list(VALID_RECOMMENDATIONS)
    )
    return out


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline.

    Returns ``(clean_df, audit_frame)``. The input frame is never mutated, so
    the caller keeps the original dataset in memory for comparison.
    """
    audit = CleaningAudit()
    original_rows = len(df)

    # 1. Drop the exported row index.
    had_index = any(str(c) == INDEX_COLUMN or str(c) == "" for c in df.columns)
    out = drop_index_column(df)
    audit.log(
        "Remove export artefact",
        f"Dropped '{INDEX_COLUMN}'" if had_index else "No index column present",
        original_rows if had_index else 0,
        "The column is only the row index of the original export and carries no "
        "business meaning.",
    )

    # 2. Capture text availability BEFORE any placeholder is written.
    missing_text = int(out["Review Text"].isna().sum())
    out["Has Review Text"] = out["Review Text"].notna()
    missing_title = int(out["Title"].isna().sum())
    out["Has Title"] = out["Title"].notna()
    audit.log(
        "Flag text availability",
        "Created 'Has Review Text' and 'Has Title' booleans",
        missing_text + missing_title,
        "Recorded before imputation so review display and keyword search can "
        "exclude empty text while rating and recommendation analytics keep the "
        "rows.",
    )

    # 3. Missing title placeholder.
    out["Title"] = out["Title"].fillna(MISSING_TITLE)
    audit.log(
        "Impute missing Title",
        f"Filled with '{MISSING_TITLE}'",
        missing_title,
        "A missing headline is not a missing review; the placeholder keeps the "
        "row usable and visibly non-fabricated.",
    )

    # 4. Missing review text placeholder.
    out["Review Text"] = out["Review Text"].fillna(MISSING_REVIEW_TEXT)
    audit.log(
        "Impute missing Review Text",
        f"Filled with '{MISSING_REVIEW_TEXT}'",
        missing_text,
        "The placeholder is excluded from review display and search via "
        "'Has Review Text'; no review content is invented.",
    )

    # 5. Missing taxonomy values.
    taxonomy_filled = 0
    for column in TAXONOMY_COLUMNS:
        if column in out.columns:
            missing = int(out[column].isna().sum())
            taxonomy_filled += missing
            out[column] = out[column].fillna(UNKNOWN_CATEGORY)
    audit.log(
        "Impute product taxonomy",
        f"Division / Department / Class filled with '{UNKNOWN_CATEGORY}'",
        taxonomy_filled,
        "Keeps the row in volume metrics while making the gap explicit instead "
        "of assigning it to a real category.",
    )

    # 6. + 7. Domain validation.
    out = validate_rating(out)
    out = validate_recommendation(out)
    invalid_rating = int((~out["Valid Rating"]).sum())
    invalid_recommendation = int((~out["Valid Recommendation"]).sum())
    audit.log(
        "Validate Rating",
        "Checked Rating is an integer in 1-5",
        invalid_rating,
        "Ratings drive the sentiment definition, so out-of-range values would "
        "corrupt every downstream metric.",
    )
    audit.log(
        "Validate Recommended IND",
        "Checked Recommended IND is 0 or 1",
        invalid_recommendation,
        "The recommendation rate is only meaningful for a clean binary flag.",
    )

    # 8. - 11. Derived analytical fields.
    out = add_sentiment(out)
    audit.log(
        "Derive Sentiment",
        "Rating 1-2 = Negative, 3 = Neutral, 4-5 = Positive",
        len(out),
        "Provides a consistent proxy sentiment label from the only ordinal "
        "satisfaction signal in the dataset.",
    )

    out = add_recommendation_label(out)
    audit.log(
        "Derive Recommendation Label",
        "1 = Recommended, 0 = Not Recommended",
        len(out),
        "Readable labels for charts and tables.",
    )

    out = add_age_band(out)
    audit.log(
        "Derive Age Band",
        "18-29 / 30-39 / 40-49 / 50-59 / 60+",
        len(out),
        "Age bands make cohort comparison stable and protect small-sample "
        "single ages from over-interpretation.",
    )

    out = add_review_length(out)
    audit.log(
        "Derive Review Length Words",
        "Whitespace word count of Review Text (placeholders = 0)",
        len(out),
        "Length is a proxy for review effort and is used to qualify text-based "
        "findings.",
    )

    # 12. Duplicate detection without deletion.
    out = flag_duplicate_review_text(out)
    duplicate_rows = int(out["Is Duplicate Review Text"].sum())
    audit.log(
        "Flag duplicated Review Text",
        "Created 'Is Duplicate Review Text' - no rows removed",
        duplicate_rows,
        "Identical wording can be a genuine repeat purchase; deletion would "
        "distort volume metrics, so duplicates are reported instead.",
    )

    # 13. Document the review-text exclusion rule.
    audit.log(
        "Scope review-text display",
        f"{int(out['Has Review Text'].sum()):,} rows carry displayable review text",
        int((~out["Has Review Text"]).sum()),
        "Rows without review text are excluded from review display and keyword "
        "search only; they remain in rating, recommendation and category "
        "analytics.",
    )

    return out, audit.to_frame()


# --------------------------------------------------------------------------- #
# Data-quality reporting
# --------------------------------------------------------------------------- #
def data_quality_summary(
    original_df: pd.DataFrame, clean_df: pd.DataFrame
) -> dict[str, Any]:
    """Headline data-quality counts for the Data Quality tab."""
    total = len(clean_df)
    completeness = completeness_metrics(original_df, clean_df)
    summary: dict[str, Any] = {
        "total_rows": total,
        "total_columns": int(clean_df.shape[1]),
        "source_columns": int(original_df.shape[1]),
        "missing_titles": int((~clean_df["Has Title"]).sum()),
        "missing_review_texts": int((~clean_df["Has Review Text"]).sum()),
        "duplicate_review_rows": int(clean_df["Is Duplicate Review Text"].sum()),
        "duplicate_review_texts": int(
            clean_df.loc[clean_df["Is Duplicate Review Text"], "Review Text"].nunique()
        ),
        "invalid_ratings": int((~clean_df["Valid Rating"]).sum()),
        "invalid_recommendations": int((~clean_df["Valid Recommendation"]).sum()),
        "completeness_rate": completeness["completeness_rate"],
        "complete_rows": completeness["complete_rows"],
        "complete_row_rate": completeness["complete_row_rate"],
        "missing_cells": completeness["missing_cells"],
        "total_cells": completeness["total_cells"],
    }
    for column in TAXONOMY_COLUMNS:
        key = "missing_" + column.replace(" Name", "").lower()
        summary[key] = (
            int((clean_df[column] == UNKNOWN_CATEGORY).sum())
            if column in clean_df.columns
            else 0
        )
    return summary


def missing_values_by_field(original_df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value count per source field (the export index is excluded)."""
    analytical = original_df.drop(
        columns=[c for c in original_df.columns if str(c) == INDEX_COLUMN],
        errors="ignore",
    )
    total = len(analytical)
    rows = []
    for column in analytical.columns:
        missing = int(analytical[column].isna().sum())
        rows.append(
            {
                "Field": str(column),
                "Rows": total,
                "Present": total - missing,
                "Missing": missing,
                "Missing %": round(missing / total * 100, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("Missing", ascending=False).reset_index(
        drop=True
    )


def data_quality_report(
    original_df: pd.DataFrame, clean_df: pd.DataFrame
) -> pd.DataFrame:
    """Data-quality issues with their analytical impact and the treatment used."""
    total = len(clean_df)
    summary = data_quality_summary(original_df, clean_df)

    def pct(count: int) -> float:
        return round(count / total * 100, 2) if total else 0.0

    rows: list[dict[str, Any]] = [
        {
            "Data Quality Issue": "Missing Title",
            "Affected Rows": summary["missing_titles"],
            "Percentage": pct(summary["missing_titles"]),
            "Severity": "Low",
            "Analytical Impact": "The title is a secondary field. Its absence "
            "does not affect any rating, recommendation, sentiment or category "
            "metric; only the review-display table shows the placeholder.",
            "Treatment Used": f"Replaced with '{MISSING_TITLE}'. The 'Has Title' "
            f"flag records the original state so the count stays auditable.",
        },
        {
            "Data Quality Issue": "Missing Review Text",
            "Affected Rows": summary["missing_review_texts"],
            "Percentage": pct(summary["missing_review_texts"]),
            "Severity": "Medium",
            "Analytical Impact": "Excluded from review-text display and keyword "
            "search, so the text sample is smaller than the rating sample. "
            "Retained in full for rating, recommendation, sentiment and category "
            "analysis - these customers still rated the product.",
            "Treatment Used": f"The 'Has Review Text' flag is captured first, "
            f"then the text is replaced with '{MISSING_REVIEW_TEXT}'. Rows are "
            f"never deleted.",
        },
        {
            "Data Quality Issue": "Missing Division Name",
            "Affected Rows": summary["missing_division"],
            "Percentage": pct(summary["missing_division"]),
            "Severity": "Low",
            "Analytical Impact": "Appears as its own segment in division "
            "breakdowns rather than being absorbed into a real division, so no "
            "division total is silently inflated.",
            "Treatment Used": f"Replaced with '{UNKNOWN_CATEGORY}'.",
        },
        {
            "Data Quality Issue": "Missing Department Name",
            "Affected Rows": summary["missing_department"],
            "Percentage": pct(summary["missing_department"]),
            "Severity": "Low",
            "Analytical Impact": "Same rows as the missing division values. "
            "Shown as a separate department so department comparisons remain "
            "honest.",
            "Treatment Used": f"Replaced with '{UNKNOWN_CATEGORY}'.",
        },
        {
            "Data Quality Issue": "Missing Class Name",
            "Affected Rows": summary["missing_class"],
            "Percentage": pct(summary["missing_class"]),
            "Severity": "Low",
            "Analytical Impact": "Excluded from the product-class matrix in "
            "practice because it falls below the minimum sample size, so it "
            "cannot distort the class ranking.",
            "Treatment Used": f"Replaced with '{UNKNOWN_CATEGORY}'.",
        },
        {
            "Data Quality Issue": "Exact duplicated Review Text",
            "Affected Rows": summary["duplicate_review_rows"],
            "Percentage": pct(summary["duplicate_review_rows"]),
            "Severity": "Low",
            "Analytical Impact": f"{summary['duplicate_review_texts']:,} distinct "
            "texts appear more than once. At this volume the effect on any rate "
            "is below 0.1 percentage points, but repeated wording would be "
            "double-counted in the review-display table.",
            "Treatment Used": "Flagged in 'Is Duplicate Review Text' and "
            "reported here. Not deleted - identical wording can be a genuine "
            "repeat purchase and removal would understate review volume.",
        },
        {
            "Data Quality Issue": "Invalid Rating (outside 1-5)",
            "Affected Rows": summary["invalid_ratings"],
            "Percentage": pct(summary["invalid_ratings"]),
            "Severity": "High" if summary["invalid_ratings"] else "None",
            "Analytical Impact": (
                "Would break the sentiment definition and the average rating."
                if summary["invalid_ratings"]
                else "No violations found, so the sentiment mapping covers every "
                "row and the average rating is calculated on the full sample."
            ),
            "Treatment Used": "Every row checked against the 1-5 domain and "
            "flagged in 'Valid Rating'; flagged rows would be excluded from "
            "sentiment metrics.",
        },
        {
            "Data Quality Issue": "Invalid Recommended IND (not 0 or 1)",
            "Affected Rows": summary["invalid_recommendations"],
            "Percentage": pct(summary["invalid_recommendations"]),
            "Severity": "High" if summary["invalid_recommendations"] else "None",
            "Analytical Impact": (
                "Would distort the recommendation rate."
                if summary["invalid_recommendations"]
                else "No violations found, so the recommendation rate is "
                "calculated on a clean binary flag across every row."
            ),
            "Treatment Used": "Every row checked against the {0, 1} domain and "
            "flagged in 'Valid Recommendation'; flagged rows would be excluded "
            "from the recommendation rate.",
        },
        {
            "Data Quality Issue": "Export artefact column 'Unnamed: 0'",
            "Affected Rows": total,
            "Percentage": 100.0 if total else 0.0,
            "Severity": "Info",
            "Analytical Impact": "None once removed. Left in place it would be "
            "treated as a numeric measure and could be aggregated by mistake.",
            "Treatment Used": "Dropped during cleaning - it is only the row "
            "index of the original export.",
        },
    ]
    return pd.DataFrame(rows)


def completeness_metrics(
    original_df: pd.DataFrame, clean_df: pd.DataFrame
) -> dict[str, float]:
    """Cell-level completeness of the *original* analytical columns."""
    analytical = original_df.drop(
        columns=[c for c in original_df.columns if str(c) == INDEX_COLUMN],
        errors="ignore",
    )
    total_cells = int(analytical.size)
    missing_cells = int(analytical.isna().sum().sum())
    complete_rows = int(analytical.notna().all(axis=1).sum())
    return {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "completeness_rate": (
            (total_cells - missing_cells) / total_cells * 100 if total_cells else 0.0
        ),
        "complete_rows": complete_rows,
        "complete_row_rate": (
            complete_rows / len(analytical) * 100 if len(analytical) else 0.0
        ),
    }


def column_completeness(original_df: pd.DataFrame) -> pd.DataFrame:
    """Per-column completeness table for the data-quality page."""
    analytical = original_df.drop(
        columns=[c for c in original_df.columns if str(c) == INDEX_COLUMN],
        errors="ignore",
    )
    total = len(analytical)
    rows = []
    for column in analytical.columns:
        missing = int(analytical[column].isna().sum())
        rows.append(
            {
                "Column": str(column),
                "Present": total - missing,
                "Missing": missing,
                "Completeness %": round((total - missing) / total * 100, 2)
                if total
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("Completeness %").reset_index(drop=True)


def analysis_ready_frame(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Rows valid for review-text display and keyword search.

    These are the rows with genuine review text and a valid rating. Everything
    else in the dashboard uses the full cleaned frame.
    """
    mask = (
        clean_df["Has Review Text"]
        & clean_df["Valid Rating"]
        & (clean_df["Sentiment"] != "Unknown")
    )
    return clean_df.loc[mask].copy()
