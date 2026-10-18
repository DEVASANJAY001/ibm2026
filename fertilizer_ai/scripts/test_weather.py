import sys, json
sys.path.insert(0, '.')
from engines.weather_engine import get_weather, geocode

loc = geocode('Coimbatore')
print('=== GEOCODE ===')
print(json.dumps(loc, indent=2))

w = get_weather(loc['lat'], loc['lon'], crop='Rice')
print()
print('=== CURRENT CONDITIONS ===')
print(json.dumps(w['current'], indent=2))
print()
print('=== APPLICATION WINDOW ===')
print(json.dumps(w['application_window'], indent=2))
print()
print('=== SOIL MOISTURE ESTIMATE ===')
print(json.dumps(w['soil_moisture_estimate'], indent=2))
print()
print('=== 7-DAY FORECAST (first 3 days) ===')
print(json.dumps(w['forecast_7d'][:3], indent=2))
print()
print('=== ADVISORIES ===')
print(json.dumps(w['advisories'], indent=2))
print()
print('=== KEY NUMBERS ===')
keys = ['rain_24h_mm','rain_3d_mm','rain_7d_mm','et0_7d_mm','tmax_3d_c']
for k in keys:
    print(f'  {k}: {w.get(k)}')
print(f'  hourly rows: {len(w["hourly_next_24h"])}')
print()
print('=== TRACE ===')
for t in w['trace']:
    print(' ', t)
