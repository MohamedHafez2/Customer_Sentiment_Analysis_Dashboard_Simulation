"""Tests for ingestion and the auditable cleaning pipeline.

These run against the real dataset in ``data/`` - no mock or synthetic records
are substituted for it. Small hand-built frames are used only to probe edge
cases (invalid ratings, missing ages) that the real file does not contain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_cleaning, data_loader


# --------------------------------------------------------------------------- #
# Fixtures - the real dataset, loaded once for the whole session
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    frame, _ = data_loader.load_default_dataset()
    return frame


@pytest.fixture(scope="session")
def clean_bundle(raw_dataset: pd.DataFrame):
    return data_cleaning.clean_dataset(raw_dataset)


@pytest.fixture(scope="session")
def clean_dataset(clean_bundle) -> pd.DataFrame:
    return clean_bundle[0]


@pytest.fixture(scope="session")
def audit_frame(clean_bundle) -> pd.DataFrame:
    return clean_bundle[1]


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def test_bundled_dataset_file_exists():
    assert data_loader.DEFAULT_DATA_PATH.exists(), (
        f"The analytical source file must be present at "
        f"{data_loader.DEFAULT_DATA_PATH}"
    )


def test_dataset_loads_with_expected_schema(raw_dataset: pd.DataFrame):
    report = data_loader.describe_schema(raw_dataset)
    assert report.missing_required == []
    assert report.unexpected == []
    assert report.is_expected_schema
    assert report.has_index_column


def test_raw_column_order_matches_specification(raw_dataset: pd.DataFrame):
    assert list(raw_dataset.columns) == list(data_loader.EXPECTED_COLUMNS)


def test_raw_row_count(raw_dataset: pd.DataFrame):
    assert len(raw_dataset) == 23_486


def test_numeric_columns_are_numeric(raw_dataset: pd.DataFrame):
    for column in data_loader.NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(raw_dataset[column]), column


def test_unsupported_extension_is_rejected():
    with pytest.raises(data_loader.DatasetError):
        data_loader.read_any(b"", "reviews.json")


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(data_loader.DatasetError):
        data_loader.load_dataset(tmp_path / "does_not_exist.csv")


def test_excel_roundtrip_produces_the_same_frame(raw_dataset: pd.DataFrame, tmp_path):
    """XLSX support is a stated requirement, so it is exercised end to end."""
    sample = raw_dataset.head(400)
    path = tmp_path / "sample.xlsx"
    sample.to_excel(path, index=False)

    reloaded = data_loader.load_dataset(path)
    report = data_loader.describe_schema(reloaded)
    assert report.missing_required == []
    assert len(reloaded) == len(sample)

    clean, _ = data_cleaning.clean_dataset(reloaded)
    assert len(clean) == len(sample)
    assert "Sentiment" in clean.columns


def test_profile_dataset_covers_every_column(raw_dataset: pd.DataFrame):
    profile = data_loader.profile_dataset(raw_dataset)
    assert len(profile) == raw_dataset.shape[1]
    assert set(profile.columns) >= {"Column", "Dtype", "Missing", "Unique"}


# --------------------------------------------------------------------------- #
# Step 1 - index column removal, immutability
# --------------------------------------------------------------------------- #
def test_export_index_column_is_dropped(clean_dataset: pd.DataFrame):
    assert data_loader.INDEX_COLUMN not in clean_dataset.columns


def test_all_required_columns_survive_cleaning(clean_dataset: pd.DataFrame):
    for column in data_loader.REQUIRED_COLUMNS:
        assert column in clean_dataset.columns, column


def test_cleaning_preserves_every_row(
    raw_dataset: pd.DataFrame, clean_dataset: pd.DataFrame
):
    assert len(clean_dataset) == len(raw_dataset)


def test_cleaning_does_not_mutate_the_original(raw_dataset: pd.DataFrame):
    before_columns = list(raw_dataset.columns)
    before_missing = raw_dataset.isna().sum().to_dict()
    data_cleaning.clean_dataset(raw_dataset)
    assert list(raw_dataset.columns) == before_columns
    assert raw_dataset.isna().sum().to_dict() == before_missing


def test_drop_index_column_is_a_noop_without_the_column():
    frame = pd.DataFrame({"Rating": [5], "Review Text": ["good"]})
    assert list(data_cleaning.drop_index_column(frame).columns) == [
        "Rating",
        "Review Text",
    ]


# --------------------------------------------------------------------------- #
# Steps 2 to 5 - missing-value handling
# --------------------------------------------------------------------------- #
def test_text_availability_is_captured_before_imputation(
    raw_dataset: pd.DataFrame, clean_dataset: pd.DataFrame
):
    expected_missing_text = int(raw_dataset["Review Text"].isna().sum())
    expected_missing_title = int(raw_dataset["Title"].isna().sum())

    assert int((~clean_dataset["Has Review Text"]).sum()) == expected_missing_text
    assert int((~clean_dataset["Has Title"]).sum()) == expected_missing_title
    assert expected_missing_text == 845
    assert expected_missing_title == 3_810


def test_missing_title_uses_the_placeholder(clean_dataset: pd.DataFrame):
    placeholder_rows = clean_dataset["Title"] == data_cleaning.MISSING_TITLE
    assert int(placeholder_rows.sum()) == 3_810
    assert clean_dataset.loc[placeholder_rows, "Has Title"].eq(False).all()


def test_missing_review_text_uses_the_placeholder(clean_dataset: pd.DataFrame):
    placeholder_rows = clean_dataset["Review Text"] == data_cleaning.MISSING_REVIEW_TEXT
    assert int(placeholder_rows.sum()) == 845
    assert clean_dataset.loc[placeholder_rows, "Has Review Text"].eq(False).all()


def test_no_missing_values_remain_in_the_analytical_columns(
    clean_dataset: pd.DataFrame,
):
    for column in data_loader.REQUIRED_COLUMNS:
        assert clean_dataset[column].isna().sum() == 0, column


def test_missing_taxonomy_becomes_unknown(
    raw_dataset: pd.DataFrame, clean_dataset: pd.DataFrame
):
    for column in data_cleaning.TAXONOMY_COLUMNS:
        expected = int(raw_dataset[column].isna().sum())
        actual = int((clean_dataset[column] == data_cleaning.UNKNOWN_CATEGORY).sum())
        assert actual == expected, column
        assert clean_dataset[column].isna().sum() == 0


# --------------------------------------------------------------------------- #
# Steps 6 and 7 - domain validation
# --------------------------------------------------------------------------- #
def test_every_rating_is_within_one_to_five(clean_dataset: pd.DataFrame):
    assert clean_dataset["Valid Rating"].all()
    assert set(clean_dataset["Rating"].unique()) <= data_cleaning.VALID_RATINGS


def test_every_recommendation_flag_is_binary(clean_dataset: pd.DataFrame):
    assert clean_dataset["Valid Recommendation"].all()
    assert (
        set(clean_dataset["Recommended IND"].unique())
        <= data_cleaning.VALID_RECOMMENDATIONS
    )


def test_out_of_range_values_are_flagged_not_dropped():
    frame = pd.DataFrame(
        {
            "Rating": [5, 0, 7, np.nan],
            "Recommended IND": [1, 2, 0, np.nan],
        }
    )
    rated = data_cleaning.validate_rating(frame)
    recommended = data_cleaning.validate_recommendation(frame)
    assert rated["Valid Rating"].tolist() == [True, False, False, False]
    assert recommended["Valid Recommendation"].tolist() == [True, False, True, False]
    assert len(rated) == 4  # nothing removed


# --------------------------------------------------------------------------- #
# Step 8 - sentiment derivation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (1, "Negative"),
        (2, "Negative"),
        (3, "Neutral"),
        (4, "Positive"),
        (5, "Positive"),
        (1.0, "Negative"),
        (5.0, "Positive"),
    ],
)
def test_sentiment_from_rating(rating, expected):
    assert data_cleaning.sentiment_from_rating(rating) == expected


@pytest.mark.parametrize("rating", [0, 6, -1, None, np.nan, "five", ""])
def test_sentiment_from_invalid_rating_is_unknown(rating):
    assert data_cleaning.sentiment_from_rating(rating) == "Unknown"


def test_sentiment_matches_the_rating_bands_on_the_real_data(
    clean_dataset: pd.DataFrame,
):
    sentiment = clean_dataset["Sentiment"].astype("object").astype(str)
    rating = clean_dataset["Rating"]
    assert (sentiment[rating.isin([1, 2])] == "Negative").all()
    assert (sentiment[rating == 3] == "Neutral").all()
    assert (sentiment[rating.isin([4, 5])] == "Positive").all()
    assert "Unknown" not in set(sentiment)


def test_sentiment_counts_partition_the_dataset(clean_dataset: pd.DataFrame):
    counts = clean_dataset["Sentiment"].astype("object").astype(str).value_counts()
    assert counts.get("Positive", 0) == 18_208
    assert counts.get("Neutral", 0) == 2_871
    assert counts.get("Negative", 0) == 2_407
    assert counts.sum() == len(clean_dataset)


# --------------------------------------------------------------------------- #
# Step 9 - recommendation label
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, "Recommended"), (0, "Not Recommended"), (1.0, "Recommended")],
)
def test_recommendation_label(value, expected):
    assert data_cleaning.recommendation_label(value) == expected


@pytest.mark.parametrize("value", [None, np.nan, 2, -1, "yes"])
def test_recommendation_label_invalid_is_unknown(value):
    assert data_cleaning.recommendation_label(value) == "Unknown"


def test_recommendation_label_matches_the_flag(clean_dataset: pd.DataFrame):
    label = clean_dataset["Recommendation Label"].astype("object").astype(str)
    flag = clean_dataset["Recommended IND"]
    assert (label[flag == 1] == "Recommended").all()
    assert (label[flag == 0] == "Not Recommended").all()


# --------------------------------------------------------------------------- #
# Step 10 - age bands
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (18, "18-29"),
        (29, "18-29"),
        (30, "30-39"),
        (39, "30-39"),
        (40, "40-49"),
        (49, "40-49"),
        (50, "50-59"),
        (59, "50-59"),
        (60, "60+"),
        (99, "60+"),
    ],
)
def test_age_band_boundaries(age, expected):
    assert data_cleaning.age_band(age) == expected


@pytest.mark.parametrize("age", [None, np.nan, "forty"])
def test_age_band_missing_is_unknown(age):
    assert data_cleaning.age_band(age) == data_cleaning.UNKNOWN_CATEGORY


def test_age_bands_are_exhaustive_on_the_real_data(clean_dataset: pd.DataFrame):
    bands = set(clean_dataset["Age Band"].astype("object").astype(str))
    assert bands <= set(data_cleaning.AGE_BAND_ORDER) | {
        data_cleaning.UNKNOWN_CATEGORY
    }
    assert data_cleaning.UNKNOWN_CATEGORY not in bands  # Age is fully populated
    counts = clean_dataset["Age Band"].astype("object").astype(str).value_counts()
    assert counts.sum() == len(clean_dataset)


def test_age_band_is_consistent_with_the_age_column(clean_dataset: pd.DataFrame):
    recomputed = clean_dataset["Age"].apply(data_cleaning.age_band)
    assert (
        recomputed == clean_dataset["Age Band"].astype("object").astype(str)
    ).all()


# --------------------------------------------------------------------------- #
# Step 11 - review length
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("three little words", 3),
        ("  padded   spacing  here ", 3),
        ("", 0),
        (None, 0),
        (np.nan, 0),
        (data_cleaning.MISSING_REVIEW_TEXT, 0),
        (data_cleaning.MISSING_TITLE, 0),
    ],
)
def test_count_words(text, expected):
    assert data_cleaning.count_words(text) == expected


def test_placeholder_rows_have_zero_length(clean_dataset: pd.DataFrame):
    no_text = clean_dataset[~clean_dataset["Has Review Text"]]
    assert (no_text["Review Length Words"] == 0).all()


def test_real_text_rows_have_positive_length(clean_dataset: pd.DataFrame):
    with_text = clean_dataset[clean_dataset["Has Review Text"]]
    assert (with_text["Review Length Words"] > 0).all()
    assert with_text["Review Length Words"].max() > 50


# --------------------------------------------------------------------------- #
# Step 12 - duplicates are flagged, never removed
# --------------------------------------------------------------------------- #
def test_duplicates_are_flagged_without_deleting_rows():
    frame = pd.DataFrame(
        {
            "Review Text": ["Same text", "same TEXT ", "Unique text", "Another"],
            "Has Review Text": [True, True, True, True],
        }
    )
    flagged = data_cleaning.flag_duplicate_review_text(frame)
    assert len(flagged) == 4
    assert flagged["Is Duplicate Review Text"].tolist() == [True, True, False, False]


def test_placeholder_text_is_not_counted_as_duplicate(clean_dataset: pd.DataFrame):
    """845 identical placeholders must not be reported as 845 duplicates."""
    no_text = clean_dataset[~clean_dataset["Has Review Text"]]
    assert not no_text["Is Duplicate Review Text"].any()


def test_duplicate_flag_is_consistent_with_the_text(clean_dataset: pd.DataFrame):
    flagged = clean_dataset[clean_dataset["Is Duplicate Review Text"]]
    if not flagged.empty:
        normalised = flagged["Review Text"].astype(str).str.strip().str.lower()
        # Every flagged row must share its text with at least one other row.
        assert (normalised.value_counts() >= 2).all()


# --------------------------------------------------------------------------- #
# Step 13 - NLP / ML scope
# --------------------------------------------------------------------------- #
def test_displayable_text_frame_excludes_only_the_textless_rows(
    clean_dataset: pd.DataFrame,
):
    ready = data_cleaning.analysis_ready_frame(clean_dataset)
    assert len(ready) == 22_641
    assert ready["Has Review Text"].all()
    assert (ready["Review Text"] != data_cleaning.MISSING_REVIEW_TEXT).all()


def test_textless_rows_remain_in_the_full_frame(clean_dataset: pd.DataFrame):
    """The 845 rating-only rows must stay available for structured analytics."""
    textless = clean_dataset[~clean_dataset["Has Review Text"]]
    assert len(textless) == 845
    assert textless["Rating"].notna().all()
    assert textless["Recommended IND"].notna().all()
    assert (
        textless["Sentiment"].astype("object").astype(str) != "Unknown"
    ).all()


# --------------------------------------------------------------------------- #
# Derived columns and the audit trail
# --------------------------------------------------------------------------- #
def test_every_derived_column_is_created(clean_dataset: pd.DataFrame):
    for column in data_cleaning.DERIVED_COLUMNS:
        assert column in clean_dataset.columns, column


def test_audit_trail_documents_every_step(audit_frame: pd.DataFrame):
    assert list(audit_frame.columns) == [
        "#",
        "Step",
        "Action",
        "Rows Affected",
        "Rationale",
    ]
    assert len(audit_frame) == 13
    assert audit_frame["#"].tolist() == list(range(1, 14))
    assert audit_frame["Rationale"].str.len().gt(20).all()
    assert (audit_frame["Rows Affected"] >= 0).all()


def test_audit_reports_the_real_imputation_counts(audit_frame: pd.DataFrame):
    by_step = dict(zip(audit_frame["Step"], audit_frame["Rows Affected"]))
    assert by_step["Impute missing Title"] == 3_810
    assert by_step["Impute missing Review Text"] == 845
    assert by_step["Validate Rating"] == 0
    assert by_step["Validate Recommended IND"] == 0


# --------------------------------------------------------------------------- #
# Data-quality reporting
# --------------------------------------------------------------------------- #
def test_data_quality_summary_counts_match_the_real_data(
    raw_dataset: pd.DataFrame, clean_dataset: pd.DataFrame
):
    summary = data_cleaning.data_quality_summary(raw_dataset, clean_dataset)
    assert summary["total_rows"] == 23_486
    assert summary["source_columns"] == 11
    assert summary["total_columns"] > 11  # derived analytical fields were added
    assert summary["missing_titles"] == 3_810
    assert summary["missing_review_texts"] == 845
    assert summary["missing_division"] == 14
    assert summary["missing_department"] == 14
    assert summary["missing_class"] == 14
    assert summary["invalid_ratings"] == 0
    assert summary["invalid_recommendations"] == 0
    assert summary["duplicate_review_rows"] >= 0
    assert 95 < summary["completeness_rate"] <= 100


def test_data_quality_report_structure(
    raw_dataset: pd.DataFrame, clean_dataset: pd.DataFrame
):
    report = data_cleaning.data_quality_report(raw_dataset, clean_dataset)
    assert list(report.columns) == [
        "Data Quality Issue",
        "Affected Rows",
        "Percentage",
        "Severity",
        "Analytical Impact",
        "Treatment Used",
    ]
    issues = set(report["Data Quality Issue"])
    for expected in (
        "Missing Title",
        "Missing Review Text",
        "Missing Division Name",
        "Missing Department Name",
        "Missing Class Name",
        "Exact duplicated Review Text",
        "Invalid Rating (outside 1-5)",
        "Invalid Recommended IND (not 0 or 1)",
    ):
        assert expected in issues, expected
    # Every row must explain both the impact and what was done about it.
    assert report["Analytical Impact"].str.len().gt(40).all()
    assert report["Treatment Used"].str.len().gt(20).all()


def test_data_quality_report_counts_match_the_real_data(
    raw_dataset: pd.DataFrame, clean_dataset: pd.DataFrame
):
    report = data_cleaning.data_quality_report(raw_dataset, clean_dataset).set_index(
        "Data Quality Issue"
    )
    assert report.loc["Missing Title", "Affected Rows"] == 3_810
    assert report.loc["Missing Review Text", "Affected Rows"] == 845
    assert report.loc["Missing Division Name", "Affected Rows"] == 14
    assert report.loc["Invalid Rating (outside 1-5)", "Affected Rows"] == 0
    assert report.loc["Invalid Recommended IND (not 0 or 1)", "Affected Rows"] == 0
    assert report.loc["Missing Title", "Percentage"] == pytest.approx(16.22, abs=0.01)


def test_missing_values_by_field(raw_dataset: pd.DataFrame):
    table = data_cleaning.missing_values_by_field(raw_dataset)
    assert list(table.columns) == ["Field", "Rows", "Present", "Missing", "Missing %"]
    assert data_loader.INDEX_COLUMN not in set(table["Field"])
    assert len(table) == 10  # 11 source columns minus the export index
    assert table["Missing"].is_monotonic_decreasing
    by_field = dict(zip(table["Field"], table["Missing"]))
    assert by_field["Title"] == 3_810
    assert by_field["Review Text"] == 845
    assert by_field["Rating"] == 0
    assert (table["Present"] + table["Missing"] == table["Rows"]).all()


def test_completeness_metrics(raw_dataset: pd.DataFrame, clean_dataset: pd.DataFrame):
    metrics = data_cleaning.completeness_metrics(raw_dataset, clean_dataset)
    assert 0 < metrics["completeness_rate"] <= 100
    assert metrics["completeness_rate"] > 95  # only Title/Text/taxonomy are sparse
    assert metrics["complete_rows"] <= len(clean_dataset)
    expected_missing = int(
        raw_dataset.drop(columns=["Unnamed: 0"]).isna().sum().sum()
    )
    assert metrics["missing_cells"] == expected_missing


def test_column_completeness_orders_worst_first(raw_dataset: pd.DataFrame):
    table = data_cleaning.column_completeness(raw_dataset)
    assert table.iloc[0]["Column"] == "Title"  # 3,810 missing is the worst column
    assert table["Completeness %"].is_monotonic_increasing
    assert data_loader.INDEX_COLUMN not in set(table["Column"])
