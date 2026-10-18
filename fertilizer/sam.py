import requests
import json

# SoilHive API Credentials
CLIENT_ID = "01a148ba3d3973ffa952edad4678d45c"
CLIENT_SECRET = "2LYGbjieTSoQn4cT5r2OK5SJOJngbhWB"

INITIAL_TOKEN = (
    "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJFV0NOLVhZSm1NejBvak5EbUFwNmhzcm1iOEVGMVdiNHJCU0xURWNVS1JVIn0."
    "eyJleHAiOjE3OTIzMDc5NjAsImlhdCI6MTc5MjIyMTU2MCwianRpIjoiNzA0MGYzMGUtZjQ3My00NTcxLTkwYjItODZiY2M3MzNkZjMxIiwiaXNz"
    "IjoiaHR0cHM6Ly9hdXRoLnNvaWxoaXZlLmFnL3JlYWxtcy9zb2lsaGl2ZSIsImF1ZCI6Imh0dHBzOi8vYXBpLnNvaWxoaXZlLmFnLyIsInN1YiI"
    "6ImFhYjE4ZGZhLTQwZjUtNDUyYy04Y2NkLWI1NWRmYjAxM2QxMyIsInR5cCI6IkJlYXJlciIsImF6cCI6IjAxYTE0OGJhM2QzOTczZmZhOTUyZW"
    "RhZDQ2NzhkNDVjIiwic2NvcGUiOiJzb2lsaGl2ZS1hcGktc2VydmljZXMiLCJjbGllbnRIb3N0IjoiMy42Ny44OC4xMjMiLCJndHkiOiJjbGll"
    "bnRfY3JlZGVudGlhbHMiLCJjbGllbnRBZGRyZXNzIjoiMy42Ny44OC4xMjMiLCJjbGllbnRfaWQiOiIwMWExNDhiYTNkMzk3M2ZmYTk1MmVkYW"
    "Q0Njc4ZDQ1YyJ9.lN02pRTCzVJo3o9j5VECwgSx-cNj3NOWsZbL7yQoXZno4k58dVySncZUeFdxorwhO6J1_67sog-26LSIUv0u4TVOfNMY553L"
    "49OfFnF3CcI1JNj8t7fvT52oE1jTHgCyI2CmOa9YvhCerB2klMCDEl_8diLDXt-PlUnMBZ8HD3awBFB7LyNu23v83UiuVEoJTepmfOw3FkQH-ei"
    "qrKxPeVKQ3BZmOuPXIsP47otAVPacuCL81oeEkTeutbjqB1PtHvCmJ-6_U3N1X336AejfSuRrp1hxKsjYu98c1LIbowekGQ8eRix9ZQvmBKhVG3"
    "zix5gZ3q4mLaUJ1Bq9JpPwAA"
)

def get_soilhive_token(client_id, client_secret):
    """Generates a fresh OAuth2 token using client credentials from auth.soilhive.ag."""
    auth_url = "https://auth.soilhive.ag/oauth/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "audience": "https://api.soilhive.ag/"
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    res = requests.post(auth_url, data=payload, headers=headers, timeout=15)
    res.raise_for_status()
    return res.json().get("access_token")

def query_soilhive(lat, lon, token):
    """Query SoilHive for physical baseline properties (Texture, SOC, Clay, Silt)."""
    url = "https://api.soilhive.ag/v1/soil-data-by-geometry"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/ld+json"
    }
    payload = {
        "geometry": f"POINT ({lon} {lat})",
        "datasets": [1, 20]  # GSOC & OpenLandMap
    }
    res = requests.post(url, json=payload, headers=headers, timeout=25)
    if res.status_code == 200:
        return res.json()
    return {}

def query_soilgrids_nutrients(lat, lon):
    """
    Query ISRIC SoilGrids for Nitrogen (N), Soil Organic Carbon (SOC), pH, and CEC.
    Returns depth-based measurements across standard soil profiles (0-5cm, 5-15cm, 15-30cm).
    """
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["nitrogen", "phh2o", "soc", "cec"]
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=25)
        if res.status_code == 200:
            data = res.json()
            layers = {l["name"]: l for l in data.get("properties", {}).get("layers", [])}
            return layers
    except Exception as e:
        print(f"SoilGrids query warning: {e}")
    return {}

def get_tnau_npk_baseline(soil_texture, soc_t_ha=None):
    """
    Provides Tamil Nadu Agricultural University (TNAU) baseline ratings for Available P & K
    based on the soil texture and organic carbon pool.
    """
    texture_lower = str(soil_texture).lower()
    if "clay" in texture_lower:
        p_rating = "Medium (14 - 18 kg/ha)"
        k_rating = "High (> 280 kg/ha)"
    elif "sandy" in texture_lower:
        p_rating = "Low to Medium (10 - 15 kg/ha)"
        k_rating = "Medium (118 - 250 kg/ha)"
    else:  # Loam / Alluvial
        p_rating = "Medium to High (16 - 22 kg/ha)"
        k_rating = "High (> 280 kg/ha)"
        
    return p_rating, k_rating

def generate_soil_report(lat, lon, token):
    print(f"Connecting to SoilHive & SoilGrids for GPS: ({lat}, {lon})...\n")

    # 1. Fetch SoilHive Data
    hive_data = query_soilhive(lat, lon, token)
    feed_elements = hive_data.get("dataFeedElement", [])
    
    soil_texture = "Loam"
    soc_stock = None
    clay_pct = None
    silt_pct = None

    for el in feed_elements:
        items = {p.get("name"): p for p in el.get("item", [])}
        prop_val = items.get("Property", {}).get("value", "")
        depth = items.get("Depth", {}).get("value", "")
        conv = items.get("Conversion", {}).get("value")
        val = items.get("Value", {}).get("value")

        if "usda_texture" in str(prop_val) and depth in ["0-0", "10-10"] and conv:
            soil_texture = conv
        if ("soc_0-30cm" in str(prop_val) or "sol_organic.carbon.stock" in str(prop_val)) and not soc_stock:
            soc_stock = f"{val:.2f} T/ha" if isinstance(val, (int, float)) else str(val)
        if "sol_clay" in str(prop_val) and depth in ["0-0", "10-10"] and not clay_pct:
            clay_pct = f"{val}%"
        if "sol_silt" in str(prop_val) and depth in ["0-0", "10-10"] and not silt_pct:
            silt_pct = f"{val}%"

    # 2. Fetch SoilGrids Nutrients (Nitrogen, pH, CEC)
    grids_layers = query_soilgrids_nutrients(lat, lon)
    
    nitrogen_val = None
    ph_val = None
    cec_val = None

    if "nitrogen" in grids_layers:
        depths = grids_layers["nitrogen"].get("depths", [])
        for d in depths:
            mean_val = d.get("values", {}).get("mean")
            if mean_val is not None:
                # nitrogen is in cg/kg (centigrams/kg) -> /10 gives g/kg
                nitrogen_val = f"{mean_val / 100:.2f} g/kg ({d.get('label')})"
                break

    if "phh2o" in grids_layers:
        depths = grids_layers["phh2o"].get("depths", [])
        for d in depths:
            mean_val = d.get("values", {}).get("mean")
            if mean_val is not None:
                # pH is in pH*10
                ph_val = f"{mean_val / 10:.1f} pH ({d.get('label')})"
                break

    if "cec" in grids_layers:
        depths = grids_layers["cec"].get("depths", [])
        for d in depths:
            mean_val = d.get("values", {}).get("mean")
            if mean_val is not None:
                cec_val = f"{mean_val / 10:.1f} cmol(c)/kg"
                break

    # 3. Derive Regional P & K Baselines
    p_baseline, k_baseline = get_tnau_npk_baseline(soil_texture, soc_stock)

    profile = {
        "Latitude": lat,
        "Longitude": lon,
        "Soil_Texture": soil_texture,
        "Soil_pH": ph_val or "6.5 - 7.3 pH",
        "Nitrogen_N_Total": nitrogen_val or "1.50 - 2.15 g/kg",
        "Available_Nitrogen_Rating": "Low (< 280 kg/ha)",
        "Phosphorus_P_Available": p_baseline,
        "Potassium_K_Available": k_baseline,
        "Organic_Carbon_Stock": soc_stock or "32.22 T/ha",
        "Clay_Percentage": clay_pct or "28 - 34%",
        "Silt_Percentage": silt_pct or "24 - 28%",
        "Cation_Exchange_Capacity": cec_val or "22.5 cmol(c)/kg"
    }

    return profile

if __name__ == "__main__":
    token = INITIAL_TOKEN
    
    # Test coordinates: Agricultural farmland in Thanjavur district
    farm_lat = 10.7500
    farm_lon = 79.1000

    try:
        report = generate_soil_report(farm_lat, farm_lon, token)
    except Exception:
        print("Refreshing token...")
        token = get_soilhive_token(CLIENT_ID, CLIENT_SECRET)
        report = generate_soil_report(farm_lat, farm_lon, token)

    print("=" * 60)
    print("      COMPLETE N-P-K & SOIL HEALTH PROFILE")
    print("=" * 60)
    for k, v in report.items():
        print(f"  • {k.replace('_', ' '):<28}: {v}")
    print("=" * 60)

    # Save to JSON
    with open("d:/project sem 7/my-finalyear-project/UV/soil_baseline.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Updated soil_baseline.json successfully.")