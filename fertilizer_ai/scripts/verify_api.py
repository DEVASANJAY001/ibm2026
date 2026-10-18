"""Full API endpoint verification test."""
import sys, io, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

payload = {
    'crop': 'Rice', 'growth_stage': 'Vegetative', 'soil_type': 'Clay_Loam',
    'irrigation_type': 'Flood', 'area_ha': 2.02, 'target_yield': 6.0,
    'soil_ph': 6.8, 'soil_n': 180, 'soil_p': 16, 'soil_k': 122,
    'lat': 11.01, 'lon': 76.96, 'include_advisory': True, 'advisory_lang': 'en',
    'llm_provider': 'rule_based'
}

r = requests.post('http://127.0.0.1:8000/recommend', json=payload, timeout=30)
d = r.json()

print('STATUS:', d.get('status'))
print('CONFIDENCE:', d.get('confidence'))
print('NPK kg/ha:', d.get('nutrients_kg_ha'))
print('TOTAL COST RANGE:', d.get('approx_total_cost_range_inr'))
print('ADVISORY PROVIDER:', d.get('advisory', {}).get('provider'))
print('WEATHER SCORE:', d.get('weather', {}).get('application_window', {}).get('score'))
print('WEATHER rain_3d_mm:', d.get('weather', {}).get('rain_3d_mm'))
print()

prods = d.get('products', {})
print('=== PRODUCTS ===')
for k, v in prods.items():
    print(f'  {k}: bags={v["bags"]}, range={v.get("approx_cost_range_inr")}')

# Check for bad vendor names
adv_text = d.get('advisory', {}).get('text', '')
adv_provider = str(d.get('advisory', {}).get('provider', ''))
all_clean = True
for bad in ['ollama', 'qwen', 'Qwen', 'Ollama']:
    if bad in adv_text:
        print(f'WARNING: found "{bad}" in advisory text!')
        all_clean = False
    if bad in adv_provider:
        print(f'WARNING: found "{bad}" in provider name: {adv_provider}')
        all_clean = False

if all_clean:
    print('\n✅ All checks passed — no vendor names in output!')
else:
    print('\n❌ Vendor name leak detected!')

# Check LLM status endpoint
r2 = requests.get('http://127.0.0.1:8000/api/llm-status', timeout=5)
llm_status = r2.json()
print('\n=== LLM STATUS ===')
for k, v in llm_status.items():
    print(f'  {k}: {v}')
    if any(bad in str(v).lower() for bad in ['ollama', 'qwen']):
        print(f'  ⚠️  WARNING: vendor name in {k}!')
