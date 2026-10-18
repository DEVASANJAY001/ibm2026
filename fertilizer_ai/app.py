"""
app.py — FastAPI entry point
────────────────────────────
Run:  uvicorn app:app --reload
Docs: http://localhost:8000/docs
"""

import sys
from pathlib import Path

# Ensure project root is on path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from recommendation.decision_engine import recommend
from llm.advisory_engine import generate_advisory

app = FastAPI(
    title="Precision Fertilizer Recommendation API",
    description=(
        "Per-crop nutrient-balance recommendations for 8 crops. "
        "Nutrient doses come from agronomic formulas (ICAR/TNAU). "
        "ML is a consistency cross-check only."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    # --- Required ---
    crop: str = Field(..., example="Rice", description="Crop name (exactly as configured)")
    growth_stage: str = Field(..., example="Vegetative")
    soil_type: str = Field(..., example="Clay_Loam")
    irrigation_type: str = Field(..., example="Flood")
    area_ha: float = Field(..., gt=0, example=1.5)
    target_yield: float = Field(..., gt=0, example=6.0, description="t/ha")
    soil_ph: float = Field(..., ge=3.0, le=10.0, example=6.5)

    # Soil NPK – None triggers dataset-median fallback (low confidence)
    soil_n: Optional[float] = Field(None, ge=0, example=80.0, description="mg/kg")
    soil_p: Optional[float] = Field(None, ge=0, example=45.0, description="mg/kg")
    soil_k: Optional[float] = Field(None, ge=0, example=120.0, description="mg/kg")

    # --- Optional: improve confidence ---
    previous_yield: Optional[float] = Field(None, ge=0, example=5.0)
    organic_matter: Optional[float] = Field(None, ge=0, example=1.2)
    ec: Optional[float] = Field(None, ge=0, example=0.8)
    crop_age_days: Optional[float] = Field(None, ge=0, example=45)
    air_temp: Optional[float] = Field(None, example=30.0)
    humidity: Optional[float] = Field(None, ge=0, le=100, example=70.0)
    rainfall: Optional[float] = Field(None, ge=0, example=50.0)

    # --- Optional: weather (lat/lon → auto-fetch) ---
    lat: Optional[float] = Field(None, ge=-90, le=90, example=11.0168)
    lon: Optional[float] = Field(None, ge=-180, le=180, example=76.9558)

    # --- Optional: health proxy ---
    chlorophyll: Optional[float] = Field(None, example=45.0)
    stress: Optional[float] = Field(None, example=0.3)
    pest: Optional[float] = Field(None, example=0.2)
    disease: Optional[float] = Field(None, example=0.1)
    weed: Optional[float] = Field(None, example=0.2)

    # --- Optional: product preference ---
    p_source: str = Field("dap", example="dap", description='"dap" or "ssp"')

    # --- Optional: vision engine result (from /analyse-leaf) ---
    vision_label: Optional[str] = Field(None, example="Nitrogen_Deficiency")
    vision_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.87)

    # --- Optional: include LLM advisory ---
    include_advisory: bool = Field(True)


# Mount static assets
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["UI"])
def root_ui():
    """Serves the farmer-friendly web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "ok",
        "service": "Precision Fertilizer Recommendation API v1.0",
        "endpoints": ["/recommend", "/analyse-leaf", "/weather", "/geocode", "/crops"],
    }


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "Precision Fertilizer Recommendation API v1.0",
        "endpoints": ["/recommend", "/analyse-leaf", "/weather", "/geocode", "/crops"],
    }


@app.get("/crops", tags=["Reference"])
def list_crops():
    """Return the list of supported crops."""
    from engines.nutrient_engine import supported_crops
    return {"crops": supported_crops()}


@app.get("/crops/{crop_name}", tags=["Reference"])
def crop_config(crop_name: str):
    """Return raw formula config for a crop (for transparency)."""
    from engines.nutrient_engine import get_crop_config
    cfg = get_crop_config(crop_name)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Crop '{crop_name}' not found")
    return {"crop": crop_name, "config": cfg}


@app.post("/recommend", tags=["Core"])
def recommend_endpoint(req: RecommendRequest) -> Dict[str, Any]:
    """
    Main recommendation endpoint.

    Returns fertilizer doses (kg/ha), commercial products (bags + cost),
    application schedule, health flags, weather advisories, and a
    plain-language LLM advisory.
    """
    result = recommend(
        crop=req.crop,
        growth_stage=req.growth_stage,
        soil_type=req.soil_type,
        irrigation_type=req.irrigation_type,
        area_ha=req.area_ha,
        target_yield=req.target_yield,
        soil_n=req.soil_n,
        soil_p=req.soil_p,
        soil_k=req.soil_k,
        soil_ph=req.soil_ph,
        lat=req.lat,
        lon=req.lon,
        previous_yield=req.previous_yield,
        organic_matter=req.organic_matter,
        ec=req.ec,
        crop_age_days=req.crop_age_days,
        air_temp=req.air_temp,
        humidity=req.humidity,
        rainfall=req.rainfall,
        chlorophyll=req.chlorophyll,
        stress=req.stress,
        pest=req.pest,
        disease=req.disease,
        weed=req.weed,
        p_source=req.p_source,
        vision_label=req.vision_label,
        vision_confidence=req.vision_confidence,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result)

    # Attach vision result if provided
    if req.vision_label:
        result["vision_input"] = {
            "label": req.vision_label,
            "confidence": req.vision_confidence,
            "note": "Provided from /analyse-leaf scan.",
        }
        # Lower confidence if vision disagrees with formula dominant nutrient
        vision_nutrient_map = {
            "Nitrogen_Deficiency": "N",
            "Phosphorus_Deficiency": "P2O5",
            "Potassium_Deficiency": "K2O",
        }
        vision_nutrient = vision_nutrient_map.get(req.vision_label)
        if vision_nutrient:
            dominant = max(
                result["nutrients_kg_ha"],
                key=lambda x: result["nutrients_kg_ha"][x]
            )
            if vision_nutrient != dominant:
                result["warnings"].append(
                    f"Vision scan detected {req.vision_label} but formula dominant nutrient is {dominant}. "
                    "Confirm with soil test."
                )

    if req.include_advisory:
        result["advisory"] = generate_advisory(result)

    return result


@app.post("/analyse-leaf", tags=["Vision"])
async def analyse_leaf(
    file: UploadFile = File(..., description="Leaf photo (JPG/PNG)"),
    crop: Optional[str] = Query(None, description="Crop name e.g. Rice"),
) -> Dict[str, Any]:
    """
    Upload a leaf photo to detect NPK deficiency or disease.

    Returns predicted class (Nitrogen_Deficiency / Phosphorus_Deficiency /
    Potassium_Deficiency / Disease_Stress / Healthy), confidence score,
    severity, and next-step advice.

    Use the returned `prediction` and `confidence` as `vision_label` and
    `vision_confidence` inputs to /recommend for an integrated recommendation.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {file.content_type}. Use JPEG or PNG."
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB cap
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    from vision.deficiency_engine import predict_from_upload
    result = predict_from_upload(image_bytes, crop=crop)
    return result


@app.get("/weather", tags=["Weather"])
def weather_endpoint(
    lat: Optional[float] = Query(None, description="Latitude (decimal degrees)"),
    lon: Optional[float] = Query(None, description="Longitude (decimal degrees)"),
    place: Optional[str] = Query(None, description="Place name e.g. Coimbatore"),
    crop: Optional[str] = Query(None, description="Crop name e.g. Rice"),
) -> Dict[str, Any]:
    """
    Get current weather + 7-day forecast + fertilizer application window score.

    Provide either lat/lon OR a place name. Place name uses Open-Meteo geocoding.
    """
    from engines.weather_engine import get_weather, geocode

    if place and (lat is None or lon is None):
        loc = geocode(place)
        if "error" in loc:
            raise HTTPException(status_code=404, detail=loc["error"])
        lat, lon = loc["lat"], loc["lon"]

    if lat is None or lon is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either lat+lon or place name."
        )

    return get_weather(lat, lon, crop=crop)


@app.get("/geocode", tags=["Weather"])
def geocode_endpoint(
    place: str = Query(..., description="Place name e.g. Madurai")
) -> Dict[str, Any]:
    """Convert a place name to lat/lon using Open-Meteo Geocoding API."""
    from engines.weather_engine import geocode
    result = geocode(place)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
