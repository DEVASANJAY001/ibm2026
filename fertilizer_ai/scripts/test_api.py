"""Quick test of all API endpoints"""
import json
import requests

BASE = "http://localhost:8000"

def test(name, method, url, **kwargs):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        resp = getattr(requests, method)(url, timeout=30, **kwargs)
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(json.dumps(data, indent=2)[:2000])  # first 2000 chars
        return data
    except Exception as e:
        print(f"ERROR: {e}")
        return None

# 1. Health check
test("GET /", "get", f"{BASE}/")

# 2. List crops
test("GET /crops", "get", f"{BASE}/crops")

# 3. Geocode
test("GET /geocode?place=Coimbatore", "get", f"{BASE}/geocode", params={"place": "Coimbatore"})

# 4. Weather by place name
test("GET /weather?place=Madurai", "get", f"{BASE}/weather", params={"place": "Madurai", "crop": "Rice"})

# 5. Weather by lat/lon
test("GET /weather?lat=11.01&lon=76.96", "get", f"{BASE}/weather", params={"lat": 11.01, "lon": 76.96})

# 6. Full recommendation
test("POST /recommend (Rice)", "post", f"{BASE}/recommend", json={
    "crop": "Rice",
    "growth_stage": "Vegetative",
    "soil_type": "Clay_Loam",
    "irrigation_type": "Flood",
    "area_ha": 2.5,
    "target_yield": 6.0,
    "soil_n": 80.0,
    "soil_p": 45.0,
    "soil_k": 120.0,
    "soil_ph": 6.5,
    "lat": 11.0168,
    "lon": 76.9558,
    "previous_yield": 5.0,
    "include_advisory": False   # skip LLM for speed
})

# 7. Crop config
test("GET /crops/Tomato", "get", f"{BASE}/crops/Tomato")

# 8. LLM Status check
test("GET /api/llm-status", "get", f"{BASE}/api/llm-status")

# 9. Recommendation with Qwen Advisory (English)
test("POST /recommend (with Qwen English Advisory)", "post", f"{BASE}/recommend", json={
    "crop": "Rice",
    "growth_stage": "Vegetative",
    "soil_type": "Clay_Loam",
    "irrigation_type": "Flood",
    "area_ha": 1.5,
    "target_yield": 6.0,
    "soil_n": 70.0,
    "soil_p": 35.0,
    "soil_k": 110.0,
    "soil_ph": 6.5,
    "lat": 11.0168,
    "lon": 76.9558,
    "include_advisory": True,
    "advisory_lang": "en",
    "llm_provider": "ollama",
    "ollama_model": "qwen2.5:1.5b"
})

print("\n\n✅ All endpoint tests complete")

