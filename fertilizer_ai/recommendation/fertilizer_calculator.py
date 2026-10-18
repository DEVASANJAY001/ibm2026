"""
recommendation/fertilizer_calculator.py
────────────────────────────────────────
Converts N / P2O5 / K2O kg/ha into commercial fertilizer product quantities.

Logic:
  1. Use DAP as the default P source.
  2. If DAP supplies MORE N than the total N requirement → switch to SSP.
  3. Remaining N gap is supplied by Urea.
  4. K is supplied by MOP.
  5. Bags are always rounded UP (ceiling).
  6. Returns per-ha and total-field quantities plus cost estimate.

All product grades and bag sizes come from config/fertilizer_products.json.
"""

import json
import math
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).parent.parent
_PROD_PATH = _HERE / "config" / "fertilizer_products.json"

with open(_PROD_PATH, "r", encoding="utf-8") as _f:
    _PRODUCTS: Dict = json.load(_f)


def _bags(total_kg: float, bag_kg: int) -> int:
    """Ceiling division → always round up to the nearest full bag."""
    return math.ceil(total_kg / bag_kg) if total_kg > 0 else 0


def calc_products(
    n_kg_ha: float,
    p2o5_kg_ha: float,
    k2o_kg_ha: float,
    area_ha: float = 1.0,
    p_source: str = "dap",
) -> Dict:
    """
    Parameters
    ----------
    n_kg_ha, p2o5_kg_ha, k2o_kg_ha : Nutrient requirements per hectare
    area_ha    : Field area
    p_source   : Preferred P source ("dap" or "ssp")

    Returns
    -------
    dict with:
        products : {product_name: {kg_per_ha, total_kg, bags, cost_inr}}
        total_cost_inr
        notes
        trace
    """
    notes: List[str] = []
    trace: List[str] = []

    # ── Step 1: Resolve P source ──────────────────────────────────────────
    p_prod = _PRODUCTS[p_source]
    p_rate_pct = p_prod["P2O5_pct"] / 100.0

    p_product_kg_ha = (p2o5_kg_ha / p_rate_pct) if p2o5_kg_ha > 0 else 0.0
    n_from_p_product = p_product_kg_ha * (p_prod["N_pct"] / 100.0)

    trace.append(
        f"P source: {p_source}  P2O5 needed={p2o5_kg_ha:.1f} kg/ha  "
        f"→ {p_product_kg_ha:.1f} kg/ha {p_source}  (supplies {n_from_p_product:.1f} kg/ha N)"
    )

    # ── Step 2: DAP excess N → switch to SSP ──────────────────────────────
    if p_source == "dap" and n_kg_ha > 0 and n_from_p_product > n_kg_ha:
        notes.append(
            f"DAP would supply {n_from_p_product:.1f} kg/ha N but only {n_kg_ha:.1f} kg/ha N "
            "is required. Switched to SSP to avoid N over-application."
        )
        trace.append("Switching P source from DAP to SSP due to N excess.")
        p_source = "ssp"
        p_prod = _PRODUCTS["ssp"]
        p_rate_pct = p_prod["P2O5_pct"] / 100.0
        p_product_kg_ha = (p2o5_kg_ha / p_rate_pct) if p2o5_kg_ha > 0 else 0.0
        n_from_p_product = p_product_kg_ha * (p_prod["N_pct"] / 100.0)

    # ── Step 3: Urea for remaining N ──────────────────────────────────────
    n_remaining = max(0.0, n_kg_ha - n_from_p_product)
    urea_kg_ha = n_remaining / (_PRODUCTS["urea"]["N_pct"] / 100.0) if n_remaining > 0 else 0.0

    trace.append(
        f"N from {p_source}={n_from_p_product:.1f}  remaining N={n_remaining:.1f}  "
        f"→ urea={urea_kg_ha:.1f} kg/ha"
    )

    # ── Step 4: MOP for K ─────────────────────────────────────────────────
    mop_kg_ha = k2o_kg_ha / (_PRODUCTS["mop"]["K2O_pct"] / 100.0) if k2o_kg_ha > 0 else 0.0
    trace.append(f"K2O needed={k2o_kg_ha:.1f}  → mop={mop_kg_ha:.1f} kg/ha")

    # ── Step 5: Build output per product ─────────────────────────────────
    product_map = {
        p_source: p_product_kg_ha,
        "urea": urea_kg_ha,
        "mop": mop_kg_ha,
    }

    products: Dict[str, Dict] = {}
    total_cost_mid = 0.0
    total_cost_min = 0.0
    total_cost_max = 0.0

    for name, per_ha in product_map.items():
        if per_ha <= 0:
            continue
        prod = _PRODUCTS[name]
        total_kg = per_ha * area_ha
        bag_size = prod["bag_kg"]
        n_bags = _bags(total_kg, bag_size)

        price_min = prod.get("price_min_inr", prod.get("price_per_bag_inr", 0))
        price_max = prod.get("price_max_inr", prod.get("price_per_bag_inr", 0))
        price_mid = prod.get("price_per_bag_inr", (price_min + price_max) // 2)

        cost_min = n_bags * price_min
        cost_max = n_bags * price_max
        cost_mid = n_bags * price_mid

        total_cost_min += cost_min
        total_cost_max += cost_max
        total_cost_mid += cost_mid

        price_range_str = f"₹{price_min:,} – ₹{price_max:,}/bag" if price_min != price_max else f"₹{price_min:,}/bag"
        cost_range_str = f"₹{cost_min:,} – ₹{cost_max:,}" if cost_min != cost_max else f"₹{cost_min:,}"

        products[name] = {
            "kg_per_ha": round(per_ha, 1),
            "total_kg": round(total_kg, 1),
            "bags": n_bags,
            "bag_kg": bag_size,
            "cost_inr": cost_mid,
            "cost_min_inr": cost_min,
            "cost_max_inr": cost_max,
            "price_per_bag_inr": price_mid,
            "price_per_bag_range_inr": price_range_str,
            "approx_cost_range_inr": cost_range_str,
        }
        trace.append(
            f"{name}: {per_ha:.1f} kg/ha × {area_ha} ha = {total_kg:.1f} kg → "
            f"{n_bags} bags ({bag_size} kg) @ {price_range_str} = approx. {cost_range_str}"
        )

    approx_total_str = f"₹{int(total_cost_min):,} – ₹{int(total_cost_max):,} INR" if total_cost_min != total_cost_max else f"₹{int(total_cost_mid):,} INR"

    return {
        "products": products,
        "total_cost_inr": round(total_cost_mid, 2),
        "total_cost_min_inr": round(total_cost_min, 2),
        "total_cost_max_inr": round(total_cost_max, 2),
        "approx_total_cost_range_inr": approx_total_str,
        "notes": notes,
        "trace": trace,
    }

