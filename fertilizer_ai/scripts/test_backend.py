"""Quick diagnostic test for the backend."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from recommendation.decision_engine import recommend
from llm.advisory_engine import check_llm_status

# 1. LLM status
status = check_llm_status()
print("=== LLM STATUS ===")
for k, v in status.items():
    print(f"  {k}: {v}")

# 2. Recommendation
print("\n=== RECOMMENDATION TEST ===")
r = recommend(
    crop='Rice', growth_stage='Vegetative', soil_type='Clay_Loam',
    irrigation_type='Flood', area_ha=1.5, target_yield=6.0, soil_ph=6.5,
    soil_n=80, soil_p=45, soil_k=120, lat=11.01, lon=76.96
)
print(f"STATUS: {r.get('status')}")
print(f"CONFIDENCE: {r.get('confidence')}")
print(f"TOTAL COST RANGE: {r.get('approx_total_cost_range_inr')}")
print(f"ADVISORY provider: {r.get('advisory', {}).get('provider', 'N/A')}")
print(f"WEATHER source: {r.get('weather', {}).get('source')}")
print(f"WARNINGS ({len(r.get('warnings',[]))}): {r.get('warnings', [])[:3]}")

prods = r.get('products', {})
print(f"\n=== PRODUCTS ({len(prods)}) ===")
for name, data in prods.items():
    print(f"  {name}: {data['bags']} bags ({data['total_kg']} kg) => {data.get('approx_cost_range_inr')}")

# Check for any 'ollama' or 'qwen' in advisory text
adv = r.get('advisory', {})
adv_text = adv.get('text', '')
for bad in ['ollama', 'qwen', 'Ollama', 'Qwen']:
    if bad in adv_text:
        print(f"\n⚠️  WARNING: '{bad}' found in advisory text!")
    if bad in str(adv.get('provider', '')):
        print(f"\n⚠️  WARNING: '{bad}' found in advisory provider!")

print("\n=== SCHEDULE ===")
for stage in r.get('schedule', []):
    print(f"  {stage['stage']}: ", end="")
    for nut, info in stage['nutrients'].items():
        if info['kg_field'] > 0:
            print(f"{nut}={info['kg_field']}kg  ", end="")
    print()

print("\nDiagnostic complete.")
