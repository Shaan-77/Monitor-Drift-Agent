"""ER-051 discount calculator tests (intentionally partial coverage)."""

from zeroui_uat.er051_discount_calculator import (
    calculate_customer_discount,
    calculate_shipping,
)


def test_gold_customer_discount_only():
    assert calculate_customer_discount("gold", 200.0) == 20.0


def test_standard_shipping_under_free_threshold():
    assert calculate_shipping(50.0, "standard") == 9.99
