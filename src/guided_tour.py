"""English Guided Tour for first-time dashboard users.

UI onboarding only: no analytics, filters, charts or simulation logic live here.
Tour progress is stored in Streamlit session state so ordinary reruns (filter
changes, sliders, tab switches) never restart the walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from urllib.parse import quote

import streamlit as st


def _html(markup: str) -> str:
    """Collapse indented HTML so Streamlit does not treat it as a code block."""
    return " ".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )

# Session-state keys (stable contract used by app.py)
STATE_STEP = "guided_tour_current_step"
STATE_ACTIVE = "guided_tour_active"


@dataclass(frozen=True)
class TourStep:
    """One coach-mark step in the Guided Tour."""

    title: str
    body: str
    # Optional dashboard tab to open while this step is shown
    tab: str | None = None
    # Stable custom anchor id rendered in the layout (data-tour-anchor)
    anchor: str | None = None
    tip: str = ""


# Complete walkthrough of the dashboard for a first-time business user.
TOUR_STEPS: tuple[TourStep, ...] = (
    TourStep(
        title="Welcome to the Customer Sentiment Intelligence Dashboard",
        body=(
            "This guided tour walks you through the full dashboard in plain "
            "business English. Every figure you will see is calculated from the "
            "supplied Women's Clothing E-Commerce Reviews dataset — there is no "
            "machine-learning prediction and no external scoring service."
        ),
        anchor="tour-welcome",
        tip="You can leave the tour at any time with Skip Tour or the Escape key.",
    ),
    TourStep(
        title="How sentiment is defined here",
        body=(
            "Sentiment in this dashboard is a rating-based business classification, "
            "not a text model:\n\n"
            "• Negative — Rating 1 or 2\n"
            "• Neutral — Rating 3\n"
            "• Positive — Rating 4 or 5\n\n"
            "Recommendation (Recommended / Not Recommended) is tracked separately "
            "from sentiment."
        ),
        anchor="tour-welcome",
    ),
    TourStep(
        title="Data source",
        body=(
            "The supplied review file loads automatically. Use the sidebar upload "
            "control only if you need to replace it with another CSV or Excel file "
            "that follows the same schema. Validation checks confirm the loaded "
            "file against published reference totals."
        ),
        anchor="tour-data-source",
        tip="Look at the Data Source block at the top of the sidebar.",
    ),
    TourStep(
        title="Presentation Mode",
        body=(
            "Presentation Mode prepares a compact, screenshot-ready layout for "
            "slides. It hides the sidebar and tightens spacing so KPIs, charts and "
            "insights remain clear on a 1920×1080 capture. You can exit it at any "
            "time from the main page."
        ),
        anchor="tour-presentation",
    ),
    TourStep(
        title="Global Filters",
        body=(
            "Global Filters narrow every tab at once: Division, Department, Class, "
            "Rating, Sentiment, Recommendation, Age Band, text availability and "
            "minimum category sample size. An empty selection means “all”. "
            "Clear All restores the defaults without affecting the Guided Tour."
        ),
        anchor="tour-filters",
        tip="Filters change the numbers; they do not restart this tour.",
    ),
    TourStep(
        title="Current Scope and download",
        body=(
            "Current Scope shows how many reviews remain after filters, and lists "
            "the active selections. Download Filtered Data exports exactly that "
            "working set as CSV for offline review or audit."
        ),
        anchor="tour-scope",
    ),
    TourStep(
        title="Dashboard views",
        body=(
            "Use the view selector under the header to move between five areas:\n\n"
            "1. Executive Overview\n"
            "2. Product Performance\n"
            "3. Customer & Review Insights\n"
            "4. Data Quality\n"
            "5. Simulation & Decision Lab\n\n"
            "Only the selected view is rendered, which keeps charts sharp and fast."
        ),
        anchor="tour-tabs",
        tab="Executive Overview",
    ),
    TourStep(
        title="Executive Overview",
        body=(
            "Start here for headline KPIs: volume, average rating, sentiment mix "
            "and recommendation rate. Charts summarise the filtered population, "
            "and Actionable Insight cards explain what the data shows, why it "
            "matters, and where to focus — all derived from the same aggregations."
        ),
        tab="Executive Overview",
        anchor="tour-tabs",
    ),
    TourStep(
        title="Product Performance",
        body=(
            "Compare Divisions, Departments and product Classes on rating, "
            "recommendation and negative rate. Performance bands (Strong / Average "
            "/ Needs Attention) are relative to the filtered baseline so colours "
            "stay meaningful when you change scope."
        ),
        tab="Product Performance",
        anchor="tour-tabs",
    ),
    TourStep(
        title="Customer & Review Insights",
        body=(
            "Explore age-band patterns, helpful-vote concentration and individual "
            "reviews. Review text is used for search and display only — the "
            "dashboard does not run NLP, topic models or sentiment prediction on "
            "the wording."
        ),
        tab="Customer & Review Insights",
        anchor="tour-tabs",
    ),
    TourStep(
        title="Data Quality",
        body=(
            "Inspect missing values, completeness, cleaning audit steps and "
            "validation against the published reference figures (for example "
            "23,486 total reviews). Use this tab when you need to demonstrate that "
            "the numbers are auditable and reproducible."
        ),
        tab="Data Quality",
        anchor="tour-tabs",
    ),
    TourStep(
        title="Simulation & Decision Lab — purpose",
        body=(
            "This tab is a business what-if lab. It estimates how management "
            "interventions could change sentiment, rating and recommendation KPIs "
            "under your assumptions. Outputs are scenario estimates — not "
            "predictions and not realised results."
        ),
        tab="Simulation & Decision Lab",
        anchor="tour-tabs",
    ),
    TourStep(
        title="Scenario controls and results",
        body=(
            "Set reach, success and conversion rates, then compare Baseline versus "
            "Simulated KPI cards, charts, rule-based insights and the department "
            "results table. Reset Scenario restores default assumptions. Download "
            "the simulation table when you need to share a planning scenario."
        ),
        tab="Simulation & Decision Lab",
        anchor="tour-tabs",
        tip="Changing simulation sliders recalculates results; it does not restart the tour.",
    ),
    TourStep(
        title="Reading results with confidence",
        body=(
            "Remember the boundaries of this dataset: there is no review date, so "
            "nothing here is a time forecast; no revenue or cost fields, so no ROI "
            "is invented; and associations in the charts are not proof of cause "
            "and effect. The Simulation disclaimer states this explicitly."
        ),
        tab="Simulation & Decision Lab",
        anchor="tour-tabs",
    ),
    TourStep(
        title="You are ready to explore",
        body=(
            "That completes the Guided Tour. Use the sidebar filters to focus a "
            "segment, move through the five views, and open Simulation & Decision "
            "Lab when you want a structured what-if discussion. You can open this "
            "tour again at any time with ▶ Start Guided Tour in the header."
        ),
        anchor="tour-welcome",
        tip="Click Finish to close the tour. Open it again any time with ▶ Start Guided Tour in the header.",
    ),
)


def step_count() -> int:
    return len(TOUR_STEPS)


def init_tour_state() -> None:
    """Initialise tour flags. The tour never auto-starts."""
    st.session_state.setdefault(STATE_STEP, 0)
    st.session_state.setdefault(STATE_ACTIVE, False)


def start_tour() -> None:
    """Header launcher: open the tour from Step 1."""
    st.session_state[STATE_ACTIVE] = True
    st.session_state[STATE_STEP] = 0


def skip_tour() -> None:
    st.session_state[STATE_ACTIVE] = False


def finish_tour() -> None:
    st.session_state[STATE_ACTIVE] = False


def close_tour() -> None:
    """Alias used by Escape / Close — same as Skip."""
    skip_tour()


def _clamp_step() -> int:
    total = step_count()
    if total <= 0:
        return 0
    current = int(st.session_state.get(STATE_STEP, 0) or 0)
    current = max(0, min(current, total - 1))
    st.session_state[STATE_STEP] = current
    return current


def current_step() -> TourStep:
    return TOUR_STEPS[_clamp_step()]


def is_active() -> bool:
    return bool(st.session_state.get(STATE_ACTIVE))


def apply_step_tab_navigation() -> None:
    """Open the tab associated with the active step before the tab control renders."""
    if not is_active():
        return
    step = current_step()
    if not step.tab:
        return
    # Keep segmented_control and our active_tab mirror aligned.
    st.session_state["active_tab"] = step.tab
    st.session_state["tab_selector"] = step.tab


def tour_anchor(anchor_id: str, active_anchor: str | None = None) -> None:
    """Render a stable custom anchor the coach panel can refer to."""
    active_class = (
        " csid-tour-anchor-active"
        if active_anchor and active_anchor == anchor_id
        else ""
    )
    st.markdown(
        f'<div class="csid-tour-anchor{active_class}" '
        f'data-tour-anchor="{escape(anchor_id)}" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )


def _go_back() -> None:
    st.session_state[STATE_STEP] = max(0, _clamp_step() - 1)


def _go_next() -> None:
    index = _clamp_step()
    if index >= step_count() - 1:
        finish_tour()
    else:
        st.session_state[STATE_STEP] = index + 1


def render_escape_handler() -> None:
    """Allow Escape to skip the tour without relying on Streamlit CSS classes."""
    if not is_active():
        return
    # Invisible iframe script talks to the parent document to click Skip Tour.
    script = """<!DOCTYPE html><html><body><script>
(function () {
  const parentWin = window.parent;
  if (!parentWin || parentWin.__csidTourEscBound) { return; }
  parentWin.__csidTourEscBound = true;
  parentWin.document.addEventListener('keydown', function (event) {
    if (event.key !== 'Escape') { return; }
    const buttons = parentWin.document.querySelectorAll('button');
    for (const button of buttons) {
      const label = (button.textContent || '').trim();
      if (label === 'Skip Tour') {
        button.click();
        break;
      }
    }
  });
})();
</script></body></html>"""
    st.iframe(
        f"data:text/html;charset=utf-8,{quote(script)}",
        height=1,
        width=1,
    )


def render_start_button(*, presentation: bool = False) -> None:
    """Permanent header launcher — the only way to open the Guided Tour."""
    # Kept visible in Presentation Mode: compact header placement avoids crowding
    # KPI and chart screenshots, so the launcher is not hidden.
    _ = presentation
    st.button(
        "▶ Start Guided Tour",
        key="guided_tour_start",
        on_click=start_tour,
        width="stretch",
        help="Open a step-by-step guide to the dashboard.",
    )

def render_tour_panel() -> None:
    """Professional coach panel with Back / Next / Skip / Finish controls."""
    if not is_active():
        return

    index = _clamp_step()
    step = TOUR_STEPS[index]
    total = step_count()
    is_first = index == 0
    is_last = index == total - 1

    tip_html = (
        f'<p class="csid-tour-tip">{escape(step.tip)}</p>' if step.tip else ""
    )
    body_html = "<br>".join(escape(line) for line in step.body.split("\n"))

    st.markdown(
        _html(
            f"""
            <div class="csid-tour-panel" role="dialog" aria-label="Guided Tour">
              <div class="csid-tour-panel-head">
                <span class="csid-tour-badge">Guided Tour</span>
                <span class="csid-tour-progress">Step {index + 1} of {total}</span>
              </div>
              <h4 class="csid-tour-title">{escape(step.title)}</h4>
              <div class="csid-tour-body">{body_html}</div>
              {tip_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    controls = st.columns([1.1, 1.1, 1.4, 1.1, 1.2])
    with controls[0]:
        st.button(
            "Back",
            key="guided_tour_back",
            on_click=_go_back,
            disabled=is_first,
            width="stretch",
            help="Return to the previous tour step.",
        )
    with controls[1]:
        if is_last:
            st.button(
                "Finish",
                key="guided_tour_finish",
                on_click=finish_tour,
                type="primary",
                width="stretch",
                help="Close the Guided Tour.",
            )
        else:
            st.button(
                "Next",
                key="guided_tour_next",
                on_click=_go_next,
                type="primary",
                width="stretch",
                help="Continue to the next tour step.",
            )
    with controls[2]:
        st.button(
            "Skip Tour",
            key="guided_tour_skip",
            on_click=skip_tour,
            width="stretch",
            help="Close the tour. Use ▶ Start Guided Tour in the header to open it again.",
        )
    with controls[3]:
        st.button(
            "Close",
            key="guided_tour_close",
            on_click=close_tour,
            width="stretch",
            help="Close the Guided Tour.",
        )
    with controls[4]:
        st.caption("Esc also closes · Start again from the header")

    render_escape_handler()


def active_anchor_id() -> str | None:
    if not is_active():
        return None
    return current_step().anchor
