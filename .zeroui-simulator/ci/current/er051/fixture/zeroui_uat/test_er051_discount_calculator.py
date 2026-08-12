"""ER-051 discount calculator tests (high coverage for pass)."""

import pytest
from zeroui_uat.er051_discount_calculator import (
    calculate_customer_discount,
    calculate_shipping,
)


def test_gold_customer_discount():
    assert calculate_customer_discount("gold", 200.0) == 20.0


def test_silver_customer_discount():
    assert calculate_customer_discount("silver", 100.0) == 5.0


def test_bronze_customer_discount():
    assert calculate_customer_discount("bronze", 100.0) == 2.0


def test_unknown_tier_discount():
    assert calculate_customer_discount("other", 100.0) == 0.0


def test_negative_subtotal_rejected():
    with pytest.raises(ValueError):
        calculate_customer_discount("gold", -1.0)


def test_standard_shipping_under_free_threshold():
    assert calculate_shipping(50.0, "standard") == 9.99


def test_express_shipping():
    assert calculate_shipping(50.0, "express") == 19.99


def test_other_region_shipping():
    assert calculate_shipping(50.0, "other") == 14.99


def test_free_shipping_over_threshold():
    assert calculate_shipping(150.0, "standard") == 0.0
