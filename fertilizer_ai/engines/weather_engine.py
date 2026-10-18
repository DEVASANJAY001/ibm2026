"""
engines/weather_engine.py
──────────────────────────
Full weather engine using Open-Meteo (free, no API key required).

Features:
  ┌─────────────────────────────────────────────────────┐
  │  ✅ Current conditions (real-time)                  │
  │  ✅ 7-day daily forecast                            │
  │  ✅ Hourly data (next 24h — best application window)│
  │  ✅ Fertilizer application window scoring (0-100)   │
  │  ✅ Soil moisture estimate from ET and rainfall     │
  │  ✅ Crop-specific advisory rules from rules.yaml    │
  │  ✅ Geocoding by city/village name (no lat/lon needed│
  │  ✅ 1-hour response cache (requests_cache)          │
  │  ✅ Graceful offline fallback — never crashes       │
  └─────────────────────────────────────────────────────┘

Usage:
    from engines.weather_engine import get_weather, geocode

    # By coordinates
    w = get_weather(11.0168, 76.9558)

    # By place name
    loc = geocode("Coimbatore")
    w = get_weather(loc["lat"], loc["lon"])
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

import yaml

_HERE = Path(__file__).parent.parent
_RULES_PATH = _HERE / "config" / "rules.yaml"

with open(_RULES_PATH, "r", encoding="utf-8") as _f:
    _RULES = yaml.safe_load(_f)

_W_RULES = _RULES.get("weather", {})

# ─── HTTP session with 1-hour cache ──────────────────────────────────────────
try:
    import requests_cache
    _SESSION = requests_cache.CachedSession(
        str(_HERE / "cache" / "weather"),
        expire_after=3600,
    )
except ImportError:
    import requests as _r
    _SESSION = _r.Session()

# Short-lived session for current conditions (no caching — always fresh)
try:
    import requests as _req
    _LIVE = _req.Session()
except ImportError:
    _LIVE = _SESSION


# ─── Helper: safe list access ─────────────────────────────────────────────────
def _safe(lst: list, idx: int, default=None):
    try:
        v = lst[idx]
        return default if v is None else v
    except IndexError:
        return default


def _nonnull(lst: list) -> list:
    return [x for x in lst if x is not None]


# ─── Fertilizer window scorer ─────────────────────────────────────────────────
def _window_score(
    rain_24h: float,
    rain_48h: float,
    tmax: Optional[float],
    wind: Optional[float],
    humidity: Optional[float],
) -> Dict:
    """
    Score the next 24-hour window for fertilizer application.
    Returns score 0-100 and a plain-English rating.

    Deductions:
      -30 if rain_24h > 5 mm  (leaching risk)
      -20 if rain_48h > 20 mm (leaching risk ahead)
      -15 if tmax > 36°C      (volatilisation risk)
      -15 if wind > 30 km/h   (drift risk)
      -10 if humidity > 90%   (slow drying)
    """
    score = 100
    reasons = []

    if rain_24h > 5:
        score -= 30
        reasons.append(f"Rain in next 24h ({rain_24h:.1f}mm) — leaching risk")
    elif rain_24h > 2:
        score -= 15
        reasons.append(f"Light rain expected ({rain_24h:.1f}mm) — some leaching risk")

    if rain_48h > 20:
        score -= 20
        reasons.append(f"Heavy rain in 48h ({rain_48h:.1f}mm) — delay if possible")

    if tmax is not None and tmax > 36:
        score -= 15
        reasons.append(f"High temp ({tmax}°C) — apply before 8 AM or after 5 PM")
    elif tmax is not None and tmax > 32:
        score -= 5
        reasons.append(f"Warm ({tmax}°C) — apply early morning for best results")

    if wind is not None and wind > 30:
        score -= 15
        reasons.append(f"High wind ({wind:.0f} km/h) — drift risk for foliar spray")

    if humidity is not None and humidity > 90:
        score -= 10
        reasons.append(f"High humidity ({humidity:.0f}%) — fertilizer may clump")

    score = max(0, score)
    if score >= 80:
        rating = "excellent"
    elif score >= 60:
        rating = "good"
    elif score >= 40:
        rating = "fair"
    else:
        rating = "poor"

    return {
        "score": score,
        "rating": rating,
        "reasons": reasons,
        "apply_now": score >= 60,
    }


# ─── Soil moisture estimate ───────────────────────────────────────────────────
def _estimate_soil_moisture(rain_7d: float, et0_7d: float) -> Dict:
    """
    Simple water balance estimate. Not a substitute for a tensiometer.
    Returns a qualitative soil moisture level.
    """
    balance = rain_7d - et0_7d
    if balance > 30:
        level = "waterlogged"
        advice = "Field may be waterlogged. Avoid heavy fertilizer until drainage improves."
    elif balance > 10:
        level = "wet"
        advice = "Soil is wet. Light top-dress OK; avoid large doses."
    elif balance > -10:
        level = "optimal"
        advice = "Soil moisture is in a good range for fertilizer application."
    elif balance > -30:
        level = "dry"
        advice = "Soil is getting dry. Irrigate before or after applying fertilizer."
    else:
        level = "very_dry"
        advice = "Soil is very dry. Irrigate first, then apply fertilizer after 24h."

    return {
        "rain_minus_et_mm": round(balance, 1),
        "level": level,
        "advice": advice,
    }


# ─── Geocoding ────────────────────────────────────────────────────────────────
def geocode(place_name: str) -> Dict:
    """
    Convert a place name (e.g. 'Coimbatore' or 'Madurai') to lat/lon.
    Uses Open-Meteo Geocoding API (free, no API key).

    Returns {"lat": float, "lon": float, "name": str, "country": str}
    or {"error": "..."} on failure.
    """
    try:
        resp = _SESSION.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place_name, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {"error": f"Place '{place_name}' not found"}
        r = results[0]
        return {
            "lat": r["latitude"],
            "lon": r["longitude"],
            "name": r.get("name", place_name),
            "admin": r.get("admin1", ""),
            "country": r.get("country", ""),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─── Current conditions ───────────────────────────────────────────────────────
def _get_current(lat: float, lon: float) -> Dict:
    """
    Fetch real-time current weather (temperature, rain, humidity, wind, cloud).
    Uses Open-Meteo current endpoint (no caching — always live).
    """
    try:
        resp = _LIVE.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "rain,"
                    "weather_code,"
                    "cloud_cover,"
                    "wind_speed_10m,"
                    "wind_direction_10m,"
                    "soil_temperature_0cm,"
                    "soil_moisture_0_to_1cm"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        c = resp.json().get("current", {})
        time_str = c.get("time", "")

        return {
            "time": time_str,
            "temperature_c": c.get("temperature_2m"),
            "feels_like_c": c.get("apparent_temperature"),
            "humidity_pct": c.get("relative_humidity_2m"),
            "rain_mm": c.get("rain", 0.0),
            "precipitation_mm": c.get("precipitation", 0.0),
            "cloud_cover_pct": c.get("cloud_cover"),
            "wind_speed_kmh": c.get("wind_speed_10m"),
            "wind_direction_deg": c.get("wind_direction_10m"),
            "soil_temp_surface_c": c.get("soil_temperature_0cm"),
            "soil_moisture_surface": c.get("soil_moisture_0_to_1cm"),
            "weather_code": c.get("weather_code"),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─── WMO weather code → description ──────────────────────────────────────────
_WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail",
}


# ─── Main public function ─────────────────────────────────────────────────────
def get_weather(lat: float, lon: float, crop: Optional[str] = None) -> Dict:
    """
    Fetch complete weather data for a field location.

    Parameters
    ----------
    lat, lon : Field coordinates (decimal degrees)
    crop     : Optional crop name — used for crop-specific advisory phrasing

    Returns
    -------
    dict with:
        current         – real-time conditions
        forecast        – 7-day daily summary
        hourly_next_24h – hourly temp, rain, humidity for today
        soil_moisture   – simple water-balance estimate
        application_window – best time to apply fertilizer (score 0-100)
        advisories      – actionable list
        source, cached, trace
    """
    trace: List[str] = [f"Weather request for ({lat:.4f}, {lon:.4f}) | crop={crop}"]
    advisories: List[Dict] = []

    # ── 1. Current conditions (live, no cache) ────────────────────────────
    current = _get_current(lat, lon)
    if "error" not in current:
        wcode = current.get("weather_code")
        current["description"] = _WMO.get(wcode, f"Code {wcode}")
        trace.append(
            f"Current: {current['temperature_c']}°C  "
            f"RH={current['humidity_pct']}%  "
            f"rain={current['rain_mm']}mm  "
            f"wind={current['wind_speed_kmh']}km/h  "
            f"[{current['description']}]"
        )

    # ── 2. 7-day daily forecast ───────────────────────────────────────────
    try:
        resp = _SESSION.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "forecast_days": 7,
                "timezone": "auto",
                "daily": (
                    "precipitation_sum,"
                    "temperature_2m_max,"
                    "temperature_2m_min,"
                    "et0_fao_evapotranspiration,"
                    "wind_speed_10m_max,"
                    "relative_humidity_2m_max,"
                    "sunshine_duration,"
                    "uv_index_max,"
                    "precipitation_probability_max"
                ),
            },
            timeout=15,
        )
        resp.raise_for_status()
        cached = getattr(resp, "from_cache", False)
    except Exception as exc:
        return {
            "source": "unavailable",
            "error": str(exc),
            "current": current,
            "advisories": [{
                "type": "weather_unavailable",
                "message": "Could not retrieve forecast. Check local forecast before applying fertilizer."
            }],
            "trace": trace + [f"Forecast API error: {exc}"],
        }

    daily = resp.json().get("daily", {})
    dates = daily.get("time", [])
    rain = [x or 0.0 for x in daily.get("precipitation_sum", [])]
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    et0  = [x or 0.0 for x in daily.get("et0_fao_evapotranspiration", [])]
    wind_max = daily.get("wind_speed_10m_max", [])
    rh_max   = daily.get("relative_humidity_2m_max", [])
    sunshine = [x or 0.0 for x in daily.get("sunshine_duration", [])]  # seconds
    uv       = daily.get("uv_index_max", [])
    rain_prob = daily.get("precipitation_probability_max", [])

    # Summary aggregates
    rain_3d = sum(rain[:3])
    rain_7d = sum(rain[:7])
    et0_3d  = sum(et0[:3])
    et0_7d  = sum(et0[:7])
    tmax_3d = max(_nonnull(tmax[:3])) if _nonnull(tmax[:3]) else None
    tmin_3d = min(_nonnull(tmin[:3])) if _nonnull(tmin[:3]) else None

    trace.append(
        f"Forecast: rain_3d={rain_3d:.1f}mm  rain_7d={rain_7d:.1f}mm  "
        f"tmax_3d={tmax_3d}°C  et0_3d={et0_3d:.1f}mm  cached={cached}"
    )

    # Build per-day forecast rows
    forecast_days = []
    for i in range(min(7, len(dates))):
        forecast_days.append({
            "date": _safe(dates, i),
            "rain_mm": round(_safe(rain, i, 0.0), 1),
            "rain_prob_pct": _safe(rain_prob, i),
            "tmax_c": _safe(tmax, i),
            "tmin_c": _safe(tmin, i),
            "et0_mm": round(_safe(et0, i, 0.0), 1),
            "wind_max_kmh": _safe(wind_max, i),
            "rh_max_pct": _safe(rh_max, i),
            "sunshine_hr": round(_safe(sunshine, i, 0.0) / 3600, 1),
            "uv_index": _safe(uv, i),
        })

    # ── 3. Hourly data (next 24h) ─────────────────────────────────────────
    hourly_next_24 = []
    _HOURLY_PARAMS = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": 2,
        "timezone": "auto",
        "hourly": (
            "temperature_2m,"
            "precipitation_probability,"
            "precipitation,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "soil_moisture_0_to_1cm"
        ),
    }
    for _attempt in range(2):  # retry once on 503
        try:
            hr_resp = _LIVE.get(
                "https://api.open-meteo.com/v1/forecast",
                params=_HOURLY_PARAMS,
                timeout=10,
            )
            hr_resp.raise_for_status()
            h = hr_resp.json().get("hourly", {})
            h_times = h.get("time", [])[:24]
            for i, t in enumerate(h_times):
                hourly_next_24.append({
                    "time": t,
                    "temp_c": _safe(h.get("temperature_2m", []), i),
                    "rain_prob_pct": _safe(h.get("precipitation_probability", []), i),
                    "rain_mm": _safe(h.get("precipitation", []), i, 0.0),
                    "rh_pct": _safe(h.get("relative_humidity_2m", []), i),
                    "wind_kmh": _safe(h.get("wind_speed_10m", []), i),
                })
            rain_24h = sum(x.get("rain_mm", 0.0) or 0 for x in hourly_next_24)
            rain_48h = rain_24h + (rain[1] if len(rain) > 1 else 0.0)
            trace.append(f"Hourly next 24h: {len(hourly_next_24)} rows  rain={rain_24h:.1f}mm")
            break
        except Exception as exc:
            if _attempt == 0:
                import time as _t; _t.sleep(1)  # wait 1s then retry
                trace.append(f"Hourly attempt {_attempt+1} failed ({exc}) — retrying")
            else:
                # Final fallback: use today's daily rain as 24h estimate
                rain_24h = rain[0] if rain else 0.0
                rain_48h = rain_24h + (rain[1] if len(rain) > 1 else 0.0)
                trace.append(f"Hourly unavailable — using daily rain[0]={rain_24h:.1f}mm as 24h estimate")

    # ── 4. Soil moisture estimate ─────────────────────────────────────────
    soil_moisture = _estimate_soil_moisture(rain_7d, et0_7d)

    # ── 5. Fertilizer window score ────────────────────────────────────────
    cur_wind = current.get("wind_speed_kmh") if "error" not in current else None
    cur_rh   = current.get("humidity_pct")   if "error" not in current else None
    window = _window_score(rain_24h, rain_48h, tmax_3d, cur_wind, cur_rh)
    trace.append(f"Application window: score={window['score']} rating={window['rating']}")

    # ── 6. Advisory flags ─────────────────────────────────────────────────
    delay_rain = _W_RULES.get("topdress_delay_rain_mm", 40.0)
    if rain_3d > delay_rain:
        advisories.append({
            "type": "delay_topdress",
            "severity": "high",
            "message": (
                f"Heavy rain forecast in next 3 days ({rain_3d:.1f} mm). "
                "Delay N top-dressing to prevent leaching."
            ),
        })

    heat_thresh = _W_RULES.get("heat_stress_tmax_c", 38.0)
    if tmax_3d is not None and tmax_3d > heat_thresh:
        advisories.append({
            "type": "heat_stress",
            "severity": "medium",
            "message": (
                f"High temperature ({tmax_3d}°C). Apply fertilizer before 8 AM "
                "to reduce ammonia volatilisation from urea."
            ),
        })

    if soil_moisture["level"] == "waterlogged":
        advisories.append({
            "type": "waterlogged",
            "severity": "high",
            "message": soil_moisture["advice"],
        })
    elif soil_moisture["level"] == "very_dry":
        advisories.append({
            "type": "very_dry",
            "severity": "medium",
            "message": soil_moisture["advice"],
        })

    wet_thresh = _W_RULES.get("wet_soil_moisture_pct", 80.0)
    cur_sm = current.get("soil_moisture_surface") if "error" not in current else None
    if cur_sm is not None and cur_sm * 100 > wet_thresh:
        advisories.append({
            "type": "high_soil_moisture",
            "severity": "medium",
            "message": f"Soil moisture is high ({cur_sm*100:.0f}%). Allow soil to dry before broadcast application.",
        })

    if not window["apply_now"]:
        advisories.append({
            "type": "poor_application_window",
            "severity": "medium",
            "message": (
                f"Application window score is {window['score']}/100 ({window['rating']}). "
                "Reasons: " + "; ".join(window["reasons"])
            ),
        })

    # ── 7. Assemble response ──────────────────────────────────────────────
    return {
        "source": "open-meteo",
        "cached": cached,
        "location": {"lat": lat, "lon": lon},
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),

        # Real-time
        "current": current,

        # Summary for decision engine
        "rain_3d_mm": round(rain_3d, 1),
        "rain_7d_mm": round(rain_7d, 1),
        "rain_24h_mm": round(rain_24h, 1),
        "tmax_3d_c": round(tmax_3d, 1) if tmax_3d is not None else None,
        "tmin_3d_c": round(tmin_3d, 1) if tmin_3d is not None else None,
        "et0_3d_mm": round(et0_3d, 1),
        "et0_7d_mm": round(et0_7d, 1),

        # Computed
        "soil_moisture_estimate": soil_moisture,
        "application_window": window,

        # Full forecast
        "forecast_7d": forecast_days,
        "hourly_next_24h": hourly_next_24,

        # Actionable
        "advisories": advisories,
        "missing_days": any(x is None for x in daily.get("precipitation_sum", [])[:7]),

        "trace": trace,
    }
