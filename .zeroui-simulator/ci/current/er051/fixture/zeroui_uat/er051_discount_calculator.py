"""ER-051 discount calculator fixture."""

def _validate_subtotal(subtotal: float) -> float:
    if subtotal < 0:
        raise ValueError("subtotal must be non-negative")
    return float(subtotal)

def calculate_customer_discount(customer_tier: str, subtotal: float) -> float:
    amount = _validate_subtotal(subtotal)
    if customer_tier == "gold":
        return round(amount * 0.10, 2)
    if customer_tier == "silver":
        return round(amount * 0.05, 2)
    if customer_tier == "bronze":
        return round(amount * 0.02, 2)
    return 0.0

def calculate_shipping(subtotal: float, region: str) -> float:
    amount = _validate_subtotal(subtotal)
    free_threshold = 100.0
    if amount >= free_threshold:
        return 0.0
    if region == "standard":
        return 9.99
    if region == "express":
        return 19.99
    return 14.99
