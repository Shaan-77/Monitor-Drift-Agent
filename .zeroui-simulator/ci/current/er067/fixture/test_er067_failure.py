"""ER-067 controlled pytest failure fixture."""


def test_er067_controlled_assertion_failure():
    actual_total = 2 + 2
    expected_total = 5
    assert actual_total == expected_total
