"""Tests for the Guided Tour onboarding module (UI only)."""

from __future__ import annotations

from src import guided_tour


def test_tour_has_complete_english_steps() -> None:
    assert guided_tour.step_count() >= 12
    titles = [step.title for step in guided_tour.TOUR_STEPS]
    assert any("Welcome" in title for title in titles)
    assert any("Executive Overview" in title for title in titles)
    assert any("Product Performance" in title for title in titles)
    assert any("Customer & Review Insights" in title for title in titles)
    assert any("Data Quality" in title for title in titles)
    assert any("Simulation" in title for title in titles)
    for step in guided_tour.TOUR_STEPS:
        assert step.title.strip()
        assert step.body.strip()
        assert "http://" not in step.body.lower()
        assert "Restart Guided Tour in the sidebar" not in step.body


def test_tour_state_keys_are_stable() -> None:
    assert guided_tour.STATE_STEP == "guided_tour_current_step"
    assert guided_tour.STATE_ACTIVE == "guided_tour_active"
    assert not hasattr(guided_tour, "STATE_STARTED")
    assert not hasattr(guided_tour, "STATE_COMPLETED")


def test_start_and_close_update_session_flags(monkeypatch) -> None:
    state: dict = {}

    class _Session(dict):
        def setdefault(self, key, value):
            return super().setdefault(key, value)

    session = _Session()
    monkeypatch.setattr(guided_tour.st, "session_state", session)

    guided_tour.init_tour_state()
    assert session[guided_tour.STATE_ACTIVE] is False
    assert session[guided_tour.STATE_STEP] == 0

    guided_tour.start_tour()
    assert session[guided_tour.STATE_ACTIVE] is True
    assert session[guided_tour.STATE_STEP] == 0

    session[guided_tour.STATE_STEP] = 4
    guided_tour.skip_tour()
    assert session[guided_tour.STATE_ACTIVE] is False

    guided_tour.start_tour()
    assert session[guided_tour.STATE_ACTIVE] is True
    assert session[guided_tour.STATE_STEP] == 0

    guided_tour.finish_tour()
    assert session[guided_tour.STATE_ACTIVE] is False
