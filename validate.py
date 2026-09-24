"""Standalone artifact validation command."""

from pathlib import Path

from validation import validate_artifacts


def main() -> int:
    root = Path(__file__).resolve().parent
    errors = validate_artifacts(root)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
