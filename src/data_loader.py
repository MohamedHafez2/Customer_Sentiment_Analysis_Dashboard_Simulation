"""Dataset ingestion for the Customer Sentiment Intelligence Dashboard.

Supports CSV, XLSX and XLS. The bundled Kaggle extract in ``data/`` is loaded
automatically at start-up; a sidebar uploader can replace it at runtime.

Nothing in this module fabricates records. If a file cannot be parsed the error
is surfaced to the caller so the UI can show it instead of silently continuing.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_FILENAME = "Womens Clothing E-Commerce Reviews.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / DEFAULT_FILENAME

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

INDEX_COLUMN = "Unnamed: 0"

#: The schema of the real Kaggle dataset, in file order.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "Unnamed: 0",
    "Clothing ID",
    "Age",
    "Title",
    "Review Text",
    "Rating",
    "Recommended IND",
    "Positive Feedback Count",
    "Division Name",
    "Department Name",
    "Class Name",
)

#: Columns the analytical layer cannot operate without.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Clothing ID",
    "Age",
    "Title",
    "Review Text",
    "Rating",
    "Recommended IND",
    "Positive Feedback Count",
    "Division Name",
    "Department Name",
    "Class Name",
)

STRUCTURED_COLUMNS: tuple[str, ...] = (
    "Clothing ID",
    "Age",
    "Rating",
    "Recommended IND",
    "Positive Feedback Count",
    "Division Name",
    "Department Name",
    "Class Name",
)

UNSTRUCTURED_COLUMNS: tuple[str, ...] = ("Title", "Review Text")

NUMERIC_COLUMNS: tuple[str, ...] = (
    "Clothing ID",
    "Age",
    "Rating",
    "Recommended IND",
    "Positive Feedback Count",
)

TEXT_COLUMNS: tuple[str, ...] = (
    "Title",
    "Review Text",
    "Division Name",
    "Department Name",
    "Class Name",
)


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be read or is structurally unusable."""


# --------------------------------------------------------------------------- #
# Schema reporting
# --------------------------------------------------------------------------- #
@dataclass
class SchemaReport:
    """Outcome of comparing an ingested file against the expected schema."""

    columns: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    has_index_column: bool = False
    row_count: int = 0

    @property
    def is_expected_schema(self) -> bool:
        return not self.missing_required and not self.unexpected

    @property
    def is_usable(self) -> bool:
        return not self.missing_required

    def messages(self) -> list[str]:
        """Human-readable schema warnings (empty when the schema matches)."""
        out: list[str] = []
        if self.missing_required:
            out.append(
                "Missing required column(s): "
                + ", ".join(f"`{c}`" for c in self.missing_required)
            )
        if self.unexpected:
            out.append(
                "Unrecognised column(s) ignored by the analytics: "
                + ", ".join(f"`{c}`" for c in self.unexpected)
            )
        return out


def describe_schema(df: pd.DataFrame) -> SchemaReport:
    """Compare ``df`` with :data:`EXPECTED_COLUMNS`."""
    columns = [str(c) for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    unexpected = [c for c in columns if c not in EXPECTED_COLUMNS]
    return SchemaReport(
        columns=columns,
        missing_required=missing,
        unexpected=unexpected,
        has_index_column=INDEX_COLUMN in columns,
        row_count=int(len(df)),
    )


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def _read_csv(source: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:  # pragma: no cover - encoding fallback
            last_error = exc
            continue
    raise DatasetError(f"Could not decode the CSV file: {last_error}")


def _read_excel(source: Any, extension: str) -> pd.DataFrame:
    engine = "openpyxl" if extension == ".xlsx" else "xlrd"
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        book = pd.ExcelFile(source, engine=engine)
    except Exception as exc:  # pragma: no cover - depends on optional engines
        raise DatasetError(f"Could not open the Excel workbook: {exc}") from exc

    # Prefer the first sheet that actually contains the review schema.
    chosen = book.sheet_names[0]
    for name in book.sheet_names:
        head = book.parse(name, nrows=5)
        cols = {str(c) for c in head.columns}
        if {"Review Text", "Rating"}.issubset(cols):
            chosen = name
            break
    return book.parse(chosen)


def read_any(source: Any, filename: str) -> pd.DataFrame:
    """Read ``source`` (path or file-like) using the extension of ``filename``."""
    extension = os.path.splitext(str(filename))[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DatasetError(
            f"Unsupported file type '{extension or filename}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if extension == ".csv":
        return _read_csv(source)
    return _read_excel(source, extension)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the known columns to analysable dtypes without dropping rows."""
    out = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in TEXT_COLUMNS:
        if column in out.columns:
            # Keep genuine missing values as NaN so the cleaning audit can count
            # them; only normalise the values that are present.
            present = out[column].notna()
            out.loc[present, column] = (
                out.loc[present, column].astype(str).str.strip()
            )
            # Whitespace-only strings are missing values in practice.
            blank = present & out[column].astype(str).str.len().eq(0)
            out.loc[blank, column] = pd.NA
    return out


def load_dataset(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load a dataset from disk and coerce dtypes."""
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError(f"Dataset not found: {file_path}")
    frame = read_any(str(file_path), file_path.name)
    return coerce_types(frame)


def load_default_dataset() -> tuple[pd.DataFrame, str]:
    """Load the bundled dataset from ``data/``.

    Falls back to any other supported file in ``data/`` so the app still starts
    if the file was renamed. Returns ``(dataframe, source_label)``.
    """
    if DEFAULT_DATA_PATH.exists():
        return load_dataset(DEFAULT_DATA_PATH), DEFAULT_DATA_PATH.name

    candidates = sorted(
        p
        for p in DATA_DIR.glob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$")
    )
    if not candidates:
        raise DatasetError(
            f"No dataset found in {DATA_DIR}. Expected '{DEFAULT_FILENAME}'."
        )
    return load_dataset(candidates[0]), candidates[0].name


def load_uploaded_dataset(uploaded_file: BinaryIO) -> tuple[pd.DataFrame, str]:
    """Load a Streamlit ``UploadedFile`` (or any file-like object with a name)."""
    name = getattr(uploaded_file, "name", "uploaded_file.csv")
    payload = uploaded_file.read()
    frame = read_any(io.BytesIO(payload), name)
    return coerce_types(frame), name


def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile used on the Methodology & Data Quality page."""
    rows = []
    total = len(df)
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "Column": str(column),
                "Dtype": str(series.dtype),
                "Non-Null": total - missing,
                "Missing": missing,
                "Missing %": round(missing / total * 100, 2) if total else 0.0,
                "Unique": int(series.nunique(dropna=True)),
                "Sample Value": _first_sample(series),
            }
        )
    return pd.DataFrame(rows)


def _first_sample(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return ""
    value = str(non_null.iloc[0])
    return value if len(value) <= 70 else value[:67] + "..."
