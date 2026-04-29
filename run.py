"""Thin CLI shim. All logic lives in kg_construction.pipeline."""
from kg_construction.pipeline import extract_and_validate, main  # noqa: F401

if __name__ == '__main__':
    main()
