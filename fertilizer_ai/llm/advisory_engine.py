"""
llm/advisory_engine.py
───────────────────────
LLM Advisory Layer — the final decision translator.

Takes the fully structured JSON output from the Decision Engine and produces:
  • A clear, concise farmer-readable advisory in plain English
  • Optionally in Tamil (set ADVISORY_LANG=ta in environment)

Model selection (in priority order):
  1. Ollama local model (Llama-3.1-8B or Mistral-7B) — zero cost, offline
  2. Groq API (llama-3.1-8b-instant) — ultra fast, free tier available
  3. OpenAI-compatible API (configurable)
  4. Rule-based fallback — always works, no LLM required

Configuration (environment variables):
  LLM_PROVIDER      = "ollama" | "groq" | "openai" | "rule_based"
  OLLAMA_MODEL      = "llama3.1:8b"  (or "mistral:7b-instruct")
  OLLAMA_BASE_URL   = "http://localhost:11434"
  GROQ_API_KEY      = "gsk_..."
  GROQ_MODEL        = "llama-3.1-8b-instant"
  OPENAI_API_KEY    = "sk-..."
  OPENAI_MODEL      = "gpt-4o-mini"
  ADVISORY_LANG     = "en" | "ta"
"""

import os
import json
from typing import Dict, Any

_LANG = os.getenv("ADVISORY_LANG", "en")
_PROVIDER = os.getenv("LLM_PROVIDER", "rule_based").lower()

# ─── System Prompt (the "grounding" — never let LLM hallucinate doses) ────────
_SYSTEM_PROMPT = """You are an expert agricultural advisor for Tamil Nadu, India.
You will receive a structured JSON object containing:
- Crop details, growth stage, area
- Calculated N, P2O5, K2O fertilizer doses (kg/ha) — these numbers are FIXED, do not change them
- Commercial fertilizer products with exact quantities and bags
- Application schedule by stage
- Health flags (if any)
- Weather advisories (if any)
- Warnings

Your task:
1. Write a clear, practical advisory for the farmer in {lang}.
2. Use simple language — the farmer may not be literate in agronomy jargon.
3. Always mention: what fertilizer to buy, how much (bags), when to apply (stages).
4. If there are warnings, explain them in simple terms.
5. Keep to 200-300 words. Use bullet points.
6. NEVER invent doses or quantities. Use ONLY the numbers given in the JSON.
7. End with: "These recommendations are estimates. Consult your local agriculture officer for field verification."
""".format(lang="English" if _LANG == "en" else "Tamil")

_USER_TEMPLATE = """Here is the fertilizer recommendation data:

{json_data}

Write the farmer advisory now."""


# ─── Rule-based fallback ──────────────────────────────────────────────────────

def _rule_based_advisory(data: Dict[str, Any]) -> str:
    """Generate a structured text advisory without any LLM."""
    crop = data.get("crop", "your crop")
    stage = data.get("growth_stage", "")
    area = data.get("area_ha", 1.0)
    n = data.get("nutrients_kg_ha", {}).get("N", 0)
    p = data.get("nutrients_kg_ha", {}).get("P2O5", 0)
    k = data.get("nutrients_kg_ha", {}).get("K2O", 0)
    ratio = data.get("npk_ratio", "")
    confidence = data.get("confidence", "medium")
    products = data.get("products", {})
    warnings = data.get("warnings", [])
    schedule = data.get("schedule", [])

    leaf_diag = data.get("leaf_diagnosis") or {}
    vis_adj = data.get("vision_adjustment") or {}

    lines = [
        f"📋 FERTILIZER ADVISORY — {crop.upper()} ({stage})",
        f"   Field: {area} ha | Confidence: {confidence.upper()}",
        "",
    ]

    # Leaf Diagnosis Section
    diag_name = leaf_diag.get("prediction") or vis_adj.get("diagnosis")
    if diag_name and diag_name not in ("Not scanned", "No leaf scan provided", "No leaf provided"):
        conf_str = leaf_diag.get("confidence_pct") or (f"{vis_adj['confidence']}%" if vis_adj.get("confidence") else "Verified")
        status_str = leaf_diag.get("status") or "Foliar Screening Complete"
        impact_str = vis_adj.get("detail") or leaf_diag.get("impact") or "Standard baseline calculation maintained"
        lines.extend([
            "🍃 LEAF DIAGNOSIS & HEALTH SCAN:",
            f"   • Detected Condition: {diag_name}",
            f"   • Model Confidence:   {conf_str}",
            f"   • Health Assessment:  {status_str}",
            f"   • Agronomic Impact:   {impact_str}",
            "",
        ])

    lines.extend([
        "🧪 RECOMMENDED DOSES (per hectare):",
        f"   • Nitrogen (N):      {n} kg/ha",
        f"   • Phosphorus (P2O5): {p} kg/ha",
        f"   • Potassium (K2O):   {k} kg/ha",
        f"   • NPK Ratio:         {ratio}",
        "",
        "🛒 WHAT TO BUY:",
    ])

    for prod_name, prod_data in products.items():
        bags = prod_data.get("bags", 0)
        total_kg = prod_data.get("total_kg", 0)
        cost = prod_data.get("cost_inr", 0)
        lines.append(
            f"   • {prod_name.upper()}: {bags} bag(s) "
            f"({total_kg:.1f} kg total) — ₹{cost}"
        )

    total_cost = data.get("total_cost_inr", 0)
    lines += ["", f"   💰 Estimated total cost: ₹{total_cost:.0f}", ""]

    if schedule:
        lines.append("📅 APPLICATION SCHEDULE:")
        for s in schedule:
            stage_name = s.get("stage", "")
            parts = []
            for nut, nd in s.get("nutrients", {}).items():
                if nd.get("kg_field", 0) > 0:
                    parts.append(f"{nut}: {nd['kg_field']} kg")
            if parts:
                lines.append(f"   • {stage_name}: {', '.join(parts)}")
        lines.append("")

    if warnings:
        lines.append("⚠️  IMPORTANT NOTES:")
        for w in warnings:
            lines.append(f"   • {w}")
        lines.append("")

    lines.append(
        "ℹ️  These are estimates based on soil test values and crop requirements. "
        "Consult your local agriculture officer for field verification."
    )

    return "\n".join(lines)


# ─── Ollama ───────────────────────────────────────────────────────────────────

def _ollama_advisory(data: Dict[str, Any]) -> str:
    try:
        import requests
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    json_data=json.dumps(data, indent=2)
                )},
            ],
            "stream": False,
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as exc:
        return f"[Ollama unavailable: {exc}]\n\n" + _rule_based_advisory(data)


# ─── Groq ─────────────────────────────────────────────────────────────────────

def _groq_advisory(data: Dict[str, Any]) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    json_data=json.dumps(data, indent=2)
                )},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"[Groq unavailable: {exc}]\n\n" + _rule_based_advisory(data)


# ─── OpenAI ───────────────────────────────────────────────────────────────────

def _openai_advisory(data: Dict[str, Any]) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    json_data=json.dumps(data, indent=2)
                )},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"[OpenAI unavailable: {exc}]\n\n" + _rule_based_advisory(data)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_advisory(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate farmer-readable advisory from structured recommendation data.

    Returns:
        {"text": "...", "provider": "ollama|groq|openai|rule_based", "language": "en|ta"}
    """
    provider = _PROVIDER

    if provider == "ollama":
        text = _ollama_advisory(data)
    elif provider == "groq":
        text = _groq_advisory(data)
    elif provider == "openai":
        text = _openai_advisory(data)
    else:
        text = _rule_based_advisory(data)
        provider = "rule_based"

    return {
        "text": text,
        "provider": provider,
        "language": _LANG,
    }
