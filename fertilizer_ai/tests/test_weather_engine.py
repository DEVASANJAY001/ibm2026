"""
tests/test_weather_engine.py
─────────────────────────────
Tests for the weather engine.
Uses real Open-Meteo API — requires internet connection.
Mocks the API for the offline/failure test.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.weather_engine import get_weather, geocode, _window_score, _estimate_soil_moisture


# ─── Unit tests (no internet needed) ─────────────────────────────────────────

class TestWindowScore:
    def test_perfect_conditions(self):
        ws = _window_score(0.0, 0.0, 28.0, 10.0, 60.0)
        assert ws["score"] == 100
        assert ws["rating"] == "excellent"
        assert ws["apply_now"] is True

    def test_heavy_rain_deducts(self):
        ws = _window_score(10.0, 30.0, 28.0, 10.0, 60.0)
        assert ws["score"] < 70
        assert ws["apply_now"] is False

    def test_heat_stress_deducts(self):
        ws = _window_score(0.0, 0.0, 40.0, 10.0, 60.0)
        assert ws["score"] < 100

    def test_score_never_negative(self):
        ws = _window_score(50.0, 100.0, 45.0, 60.0, 99.0)
        assert ws["score"] >= 0

    def test_rating_categories(self):
        assert _window_score(0, 0, 25, 5, 50)["rating"] == "excellent"
        assert _window_score(3, 5, 30, 10, 70)["rating"] in ("good", "fair", "excellent")
        assert _window_score(15, 40, 42, 50, 99)["rating"] == "poor"


class TestSoilMoisture:
    def test_waterlogged(self):
        r = _estimate_soil_moisture(100.0, 20.0)
        assert r["level"] == "waterlogged"
        assert r["rain_minus_et_mm"] == 80.0

    def test_very_dry(self):
        r = _estimate_soil_moisture(0.0, 50.0)
        assert r["level"] == "very_dry"

    def test_optimal(self):
        r = _estimate_soil_moisture(30.0, 25.0)
        assert r["level"] == "optimal"


# ─── API offline fallback test ────────────────────────────────────────────────

def test_weather_returns_valid_dict_on_failure():
    """Engine must return a valid dict even when the API is completely unreachable."""
    with patch("engines.weather_engine._LIVE") as mock_live, \
         patch("engines.weather_engine._SESSION") as mock_session:

        mock_live.get.side_effect = ConnectionError("Network unreachable")
        mock_session.get.side_effect = ConnectionError("Network unreachable")

        result = get_weather(11.0, 77.0)
        # Must not raise; must have source field
        assert "source" in result
        assert result["source"] == "unavailable"
        assert "advisories" in result


# ─── Live API tests (requires internet) ──────────────────────────────────────

@pytest.mark.skipif(
    not __import__("os").environ.get("ONLINE_TESTS"),
    reason="Set ONLINE_TESTS=1 to run live API tests"
)
class TestLiveAPI:
    def test_geocode_coimbatore(self):
        loc = geocode("Coimbatore")
        assert "error" not in loc
        assert abs(loc["lat"] - 11.0) < 1.0
        assert loc["country"] == "India"

    def test_get_weather_structure(self):
        w = get_weather(11.0168, 76.9558)
        required_keys = [
            "source", "current", "forecast_7d", "rain_3d_mm",
            "rain_7d_mm", "tmax_3d_c", "et0_3d_mm",
            "application_window", "soil_moisture_estimate",
            "advisories", "trace",
        ]
        for k in required_keys:
            assert k in w, f"Missing key: {k}"

    def test_application_window_has_score(self):
        w = get_weather(11.0168, 76.9558)
        aw = w["application_window"]
        assert 0 <= aw["score"] <= 100
        assert aw["rating"] in ("excellent", "good", "fair", "poor")
        assert isinstance(aw["apply_now"], bool)

    def test_forecast_7_days(self):
        w = get_weather(11.0168, 76.9558)
        assert len(w["forecast_7d"]) == 7
        for day in w["forecast_7d"]:
            assert "date" in day
            assert "rain_mm" in day
            assert "tmax_c" in day

    def test_current_conditions_populated(self):
        w = get_weather(11.0168, 76.9558)
        cur = w["current"]
        assert "temperature_c" in cur
        assert cur.get("temperature_c") is not None
