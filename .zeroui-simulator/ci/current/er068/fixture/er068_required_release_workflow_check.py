"""ER-068 required release workflow check fixture."""


def main() -> int:
    print("Running controlled ER-068 required release workflow check.")
    print("Controlled blocker scenario: required release workflow failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
