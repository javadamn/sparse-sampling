from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _compare_frames(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = generated.set_index(key).sort_index()
    right = reference.set_index(key).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Generated and reference sample identifiers do not match")

    rows: list[dict[str, Any]] = []
    for column in left.columns:
        if column not in right.columns:
            rows.append(
                {
                    "column": column,
                    "comparison_type": "missing_reference_column",
                    "same_missingness": False,
                    "maximum_absolute_difference": np.nan,
                    "equal_values": False,
                    "status": "FAIL",
                }
            )
            continue
        x = left[column]
        y = right[column]
        same_missingness = bool(x.isna().equals(y.isna()))
        if "date" in column.lower():
            maximum = np.nan
            equal = bool(
                same_missingness
                and pd.to_datetime(x, errors="coerce").equals(
                    pd.to_datetime(y, errors="coerce")
                )
            )
            comparison_type = "datetime"
        elif pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv = x.to_numpy(float)
            yv = y.to_numpy(float)
            valid = np.isfinite(xv) & np.isfinite(yv)
            maximum = float(np.max(np.abs(xv[valid] - yv[valid]))) if valid.any() else 0.0
            equal = bool(same_missingness and maximum <= 1e-12)
            comparison_type = "numeric"
        else:
            maximum = np.nan
            equal = bool(
                same_missingness
                and x.fillna("<missing>").astype(str).equals(
                    y.fillna("<missing>").astype(str)
                )
            )
            comparison_type = "text"
        rows.append(
            {
                "column": column,
                "comparison_type": comparison_type,
                "same_missingness": same_missingness,
                "maximum_absolute_difference": maximum,
                "equal_values": equal,
                "status": "PASS" if equal else "FAIL",
            }
        )

    comparison = pd.DataFrame(rows)
    numeric_differences = comparison["maximum_absolute_difference"].dropna()
    summary = {
        "generated_rows": int(len(generated)),
        "generated_columns": int(len(generated.columns)),
        "reference_rows": int(len(reference)),
        "reference_columns": int(len(reference.columns)),
        "same_sample_keys": True,
        "compared_columns": int(len(comparison)),
        "passing_columns": int((comparison["status"] == "PASS").sum()),
        "maximum_numeric_absolute_difference": (
            float(numeric_differences.max()) if len(numeric_differences) else 0.0
        ),
        "all_compared_columns_match": bool((comparison["status"] == "PASS").all()),
    }
    return comparison, summary


def reconstruct_analysis_input(
    root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct the canonical analysis table from the two source CSVs.

    The raw metadata/SCFA table is joined to exact metabolite rows from the
    HMP2 metabolomics intensity matrix by sample identifier. No normalization,
    imputation, PCA, or substring feature matching is used.
    """
    source_files = config["source_files"]
    raw_path = root / source_files["raw_metadata_scfa"]
    matrix_path = root / source_files["metabolomics_matrix"]
    reference_path = root / config["reference_processed_file"]

    raw = pd.read_csv(raw_path)
    matrix = pd.read_csv(matrix_path)
    sample_column = config["sample_column"]
    subject_column = config["subject_column"]

    if sample_column not in raw.columns or subject_column not in raw.columns:
        raise ValueError("Raw table is missing its sample or subject identifier")
    if raw[sample_column].isna().any() or not raw[sample_column].is_unique:
        raise ValueError("Raw sample identifiers must be complete and unique")
    if matrix.columns[0] != "Metabolite":
        raise ValueError("Metabolomics matrix must have 'Metabolite' as its first column")
    if matrix["Metabolite"].isna().any() or not matrix["Metabolite"].is_unique:
        raise ValueError("Metabolite names must be complete and unique")

    raw_ids = set(raw[sample_column].astype(str))
    matrix_ids = set(map(str, matrix.columns[1:]))
    if raw_ids != matrix_ids:
        raise ValueError("Raw and metabolomics-matrix sample sets do not match")

    output_to_source = {
        name: ("propionate" if name == "propionate_y" else name)
        for name in config["metabolites"]
    }
    missing_metabolites = sorted(
        set(output_to_source.values()).difference(matrix["Metabolite"])
    )
    if missing_metabolites:
        raise ValueError(f"Metabolomics matrix is missing: {missing_metabolites}")

    selected = matrix.set_index("Metabolite").loc[list(output_to_source.values())]
    wide = selected.T
    wide.index.name = sample_column
    wide = wide.reset_index()
    wide = wide.rename(columns={source: output for output, source in output_to_source.items()})
    for column in output_to_source:
        wide[column] = pd.to_numeric(wide[column], errors="coerce")

    base = raw.rename(columns={"propionate": "propionate_x"}).copy()
    generated = base.merge(wide, on=sample_column, how="left", validate="one_to_one")
    if len(generated) != len(raw):
        raise ValueError("The source join changed the number of samples")

    reference = pd.read_csv(reference_path)
    comparison, comparison_summary = _compare_frames(
        generated, reference, sample_column
    )
    subject_mapping_matches = bool(
        generated[[sample_column, subject_column]]
        .sort_values(sample_column)
        .reset_index(drop=True)
        .equals(
            reference[[sample_column, subject_column]]
            .sort_values(sample_column)
            .reset_index(drop=True)
        )
    )
    summary = {
        **comparison_summary,
        "raw_source": str(raw_path.relative_to(root)),
        "metabolomics_source": str(matrix_path.relative_to(root)),
        "processed_reference": str(reference_path.relative_to(root)),
        "source_samples": int(len(raw)),
        "source_subjects": int(raw[subject_column].nunique()),
        "matrix_metabolites": int(len(matrix)),
        "matrix_sample_set_matches_raw": True,
        "subject_mapping_matches_reference": subject_mapping_matches,
        "legacy_derived_columns_excluded": [
            "F_met",
            "A_met",
            "aux_propionate_z",
            "aux_succinate_z",
            "aux_lactate_z",
        ],
    }
    if not summary["all_compared_columns_match"] or not subject_mapping_matches:
        raise ValueError("Reconstructed canonical input does not match the supplied reference")
    return generated, comparison, summary


def write_reconstructed_input(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    generated, comparison, summary = reconstruct_analysis_input(root, config)
    output_path = root / config["input_file"]
    comparison_path = root / "provenance" / "raw_to_analysis_comparison.csv"
    summary_path = root / "provenance" / "raw_to_analysis_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
