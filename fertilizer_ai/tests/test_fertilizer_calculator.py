"""
tests/test_fertilizer_calculator.py
─────────────────────────────────────
Unit tests for the fertilizer product calculator.
"""

import sys
from pathlib import Path
import math
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from recommendation.fertilizer_calculator import calc_products


def test_basic_output_structure():
    result = calc_products(120.0, 60.0, 80.0, area_ha=2.0)
    assert "products" in result
    assert "total_cost_inr" in result
    assert "notes" in result
    assert "trace" in result


def test_bags_rounded_up():
    """Bags must always be ceiling division."""
    result = calc_products(100.0, 50.0, 60.0, area_ha=1.0)
    for prod, data in result["products"].items():
        total_kg = data["total_kg"]
        bag_kg = data["bag_kg"]
        expected_bags = math.ceil(total_kg / bag_kg)
        assert data["bags"] == expected_bags, \
            f"{prod}: expected {expected_bags} bags, got {data['bags']}"


def test_dap_excess_n_switches_to_ssp():
    """If DAP supplies more N than needed, must switch to SSP."""
    # Low N requirement but high P → DAP would over-supply N
    result = calc_products(n_kg_ha=10.0, p2o5_kg_ha=60.0, k2o_kg_ha=0.0, area_ha=1.0, p_source="dap")
    # Should have switched to ssp
    assert any("SSP" in note or "ssp" in note.lower() for note in result["notes"]), \
        "Expected DAP→SSP switch note"
    assert "ssp" in result["products"]
    assert "dap" not in result["products"]


def test_zero_k_no_mop():
    """No MOP should appear when K2O requirement is 0."""
    result = calc_products(80.0, 40.0, 0.0, area_ha=1.0)
    assert "mop" not in result["products"]


def test_zero_n_no_urea():
    """No Urea when N requirement is 0."""
    result = calc_products(0.0, 40.0, 60.0, area_ha=1.0)
    assert "urea" not in result["products"]


def test_field_total_scales_with_area():
    """total_kg should scale linearly with area_ha."""
    r1 = calc_products(100.0, 50.0, 60.0, area_ha=1.0)
    r2 = calc_products(100.0, 50.0, 60.0, area_ha=2.0)
    for prod in r1["products"]:
        if prod in r2["products"]:
            assert abs(r2["products"][prod]["total_kg"] - 2 * r1["products"][prod]["total_kg"]) < 0.2


def test_cost_is_non_negative():
    result = calc_products(80.0, 40.0, 60.0, area_ha=1.5)
    assert result["total_cost_inr"] >= 0
