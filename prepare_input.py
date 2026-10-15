#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from resilience.analysis import load_config
from resilience.provenance import write_reconstructed_input


ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = load_config(ROOT / "config.json")
    summary = write_reconstructed_input(ROOT, config)
    print(
        "Reconstructed analysis input: "
        f"{summary['generated_rows']} samples, "
        f"{summary['generated_columns']} columns, "
        f"maximum numeric difference {summary['maximum_numeric_absolute_difference']:.3g}."
    )


if __name__ == "__main__":
    main()
