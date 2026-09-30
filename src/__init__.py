"""Customer Sentiment Intelligence Dashboard - analytical source package.

This is a descriptive and diagnostic analytics application. It reads, cleans,
aggregates, filters and visualises the supplied review dataset. It contains no
machine-learning, NLP or predictive component: sentiment here is a rating-based
business classification, not a model output.

Modules
-------
data_loader      Multi-format ingestion (CSV / XLSX / XLS) and schema checks.
data_cleaning    Auditable cleaning pipeline and data-quality reporting.
analytics        Deterministic KPI, segment, validation and insight calculations.
simulation       Deterministic what-if scenario calculations (no ML / prediction).
guided_tour      First-time English Guided Tour (UI onboarding only).
charts           Plotly figure builders with a single shared visual language.
styles           Design tokens and custom CSS.
"""

__all__ = [
    "analytics",
    "charts",
    "data_cleaning",
    "data_loader",
    "guided_tour",
    "simulation",
    "styles",
]

__version__ = "1.0.0"
