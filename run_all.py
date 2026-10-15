#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from resilience import __version__
from resilience.analysis import (
    estimate_curvature,
    load_config,
    prepare_data,
    run_toy_models,
    run_toy_sensitivity,
    screen_metabolites,
    sha256_file,
    software_versions,
    test_multistability,
)
from resilience.figures import (
    plot_curvature,
    plot_empirical_landscapes,
    plot_math_snapshot,
    plot_toy_models,
    plot_toy_sensitivity,
)
from resilience.provenance import write_reconstructed_input
from resilience.reviewer_analyses import (
    arx_robustness,
    build_rule_based_transitions,
    elapsed_time_sensitivity,
    exposure_sensitivity,
    feature_provenance_table,
    plot_arx_robustness,
    plot_random_walk_bias,
    plot_sampling_intervals,
    random_walk_bias_analysis,
    return_model_goodness_of_fit,
    threshold_sensitivity,
)


ROOT = Path(__file__).resolve().parent


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def write_reproduction_check(
    path: Path,
    raw: pd.DataFrame,
    transitions: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    curvature: pd.DataFrame,
    topology: pd.DataFrame,
    screen: pd.DataFrame,
    toy: pd.DataFrame,
    toy_sensitivity: pd.DataFrame,
    input_provenance: dict,
    config: dict,
) -> None:
    summary = cohort_summary.iloc[0]
    toy_lookup = {(row.system, row.sampling): int(row.inferred_minima) for row in toy.itertuples(index=False)}
    reported_toy = toy_sensitivity.loc[
        np.isclose(toy_sensitivity["sigma"], float(config["toy"]["sigma"]))
        & (toy_sensitivity["integration_steps_retained"] == int(config["toy"]["sparse_every"]))
    ].set_index("system")
    ou_grid = toy_sensitivity.loc[toy_sensitivity["system"] == "ou"]
    ou_runs = int(ou_grid["runs_valid"].sum())
    ou_detections = int(
        np.rint(
            (
                ou_grid["multistable_classification_fraction"]
                * ou_grid["runs_valid"]
            ).sum()
        )
    )
    retained_gaps = transitions["elapsed_days"].to_numpy(float)
    lines = [
        "# Reproduction check for the revised manuscript",
        "",
        "This report records the validated outputs used in the revised manuscript.",
        "",
        "| Revised statement | Rerun result |",
        "|---|---:|",
        f"| Joined HMP2 source cohort | {raw[config['subject_column']].nunique()} subjects, {len(raw)} samples |",
        f"| Primary cohort | {int(summary['subjects'])} subjects |",
        f"| Samples and transitions per primary subject | median {_fmt(float(summary['median_samples_per_subject']))} and {_fmt(float(summary['median_transitions_per_subject']))} |",
        f"| Retained observation gaps | {int(np.min(retained_gaps))}–{int(np.max(retained_gaps))} days, pooled median {_fmt(float(np.median(retained_gaps)))} |",
        f"| Negative apparent return coefficients | {int((curvature['b'] < 0).sum())}/{len(curvature)} |",
        f"| Sign-stable paired-bootstrap ranges | {int(curvature['ci_reportable'].sum())}/{len(curvature)} |",
        f"| Robust empirical multistability | {int(topology['robust_multistability'].sum())}/{len(topology)} subjects |",
        f"| Selected-seed toy minima | {toy_lookup} |",
        f"| Double-well detection at sigma={config['toy']['sigma']}, stride={config['toy']['sparse_every']} | {float(reported_toy.loc['double_well', 'multistable_classification_fraction']):.3f} across {int(reported_toy.loc['double_well', 'runs_requested'])} runs |",
        f"| OU spurious second minima across the full grid | {ou_detections}/{ou_runs} |",
        f"| ARX candidates screened | {len(screen)} |",
        "",
        "## Provenance checks",
        "",
        f"- Raw-to-analysis reconstruction: {input_provenance['passing_columns']}/{input_provenance['compared_columns']} compared columns match.",
        f"- Maximum numeric reconstruction difference: {input_provenance['maximum_numeric_absolute_difference']:.3g}.",
        f"- Primary analysis IDs: {', '.join(config['analysis_subjects'])}; the code requires exact agreement with the seven-transition rule.",
        f"- Bootstrap replicates requested per subject: {config['bootstrap_replicates']}.",
        f"- Toy sensitivity trajectories per cell: {config['toy']['sensitivity']['seeds']}.",
        "",
        "## Interpretation",
        "",
        "The validated package supports claims about detection limits and study design. It does not support biological resilience, subject ranking, or metabolite-modulator claims in the primary empirical cohort.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config_path = ROOT / "config.json"
    config = load_config(config_path)
    input_provenance = write_reconstructed_input(ROOT, config)
    input_path = ROOT / config["input_file"]
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw, observations, transitions, cohort = prepare_data(input_path, config)
    curvature, bootstrap_draws = estimate_curvature(observations, transitions, config)
    topology, landscapes = test_multistability(transitions, config)
    screen = screen_metabolites(transitions, config)
    toy_summary, toy_details = run_toy_models(config)
    toy_sensitivity_runs, toy_sensitivity_summary = run_toy_sensitivity(config)
    arx_primary, arx_influence, arx_maxstat_null = arx_robustness(transitions, config)
    broader_transitions, broader_cohort = build_rule_based_transitions(
        raw, config, config["minimum_transitions_documented"]
    )
    arx_broader = screen_metabolites(broader_transitions, config)
    random_walk_summary, random_walk_subjects = random_walk_bias_analysis(
        transitions, config
    )
    topology_sensitivity = threshold_sensitivity(transitions, config)
    elapsed_time = elapsed_time_sensitivity(transitions)
    return_gof = return_model_goodness_of_fit(transitions)
    exposure_observations, exposure_return = exposure_sensitivity(raw, config)
    feature_provenance = feature_provenance_table(config)

    observations_to_save = observations[
        [
            "subject",
            config["sample_column"],
            "_time",
            "_butyrate",
            "z",
            "observation_index",
            *config["metabolites"],
        ]
    ].rename(
        columns={
            config["sample_column"]: "sample_id",
            "_time": "date",
            "_butyrate": "butyrate",
        }
    )
    observations_to_save.to_csv(results_dir / "analysis_observations.csv", index=False)
    transitions.to_csv(results_dir / "analysis_transitions.csv", index=False)
    cohort.to_csv(results_dir / "cohort_subjects.csv", index=False)
    curvature.to_csv(results_dir / "subject_curvature.csv", index=False)
    topology.to_csv(results_dir / "multistability.csv", index=False)
    screen.to_csv(results_dir / "metabolite_screen.csv", index=False)
    toy_summary.to_csv(results_dir / "toy_model_summary.csv", index=False)
    toy_sensitivity_runs.to_csv(results_dir / "toy_sensitivity_runs.csv", index=False)
    toy_sensitivity_summary.to_csv(results_dir / "toy_sensitivity_summary.csv", index=False)
    arx_primary.to_csv(results_dir / "metabolite_screen_robustness.csv", index=False)
    arx_influence.to_csv(results_dir / "metabolite_screen_leave_one_subject_out.csv", index=False)
    arx_maxstat_null.to_csv(results_dir / "metabolite_screen_maxstat_null.csv", index=False)
    broader_transitions.to_csv(results_dir / "six_transition_cohort_transitions.csv", index=False)
    broader_cohort.to_csv(results_dir / "six_transition_cohort_subjects.csv", index=False)
    arx_broader.to_csv(results_dir / "metabolite_screen_six_transition_cohort.csv", index=False)
    random_walk_summary.to_csv(results_dir / "random_walk_bias_summary.csv", index=False)
    random_walk_subjects.to_csv(results_dir / "random_walk_subject_probabilities.csv", index=False)
    topology_sensitivity.to_csv(results_dir / "multistability_threshold_sensitivity.csv", index=False)
    elapsed_time.to_csv(results_dir / "elapsed_time_sensitivity.csv", index=False)
    return_gof.to_csv(results_dir / "return_model_goodness_of_fit.csv", index=False)
    exposure_observations.to_csv(results_dir / "exposure_observations.csv", index=False)
    exposure_return.to_csv(results_dir / "antibiotic_adjacent_transition_sensitivity.csv", index=False)
    feature_provenance.to_csv(results_dir / "metabolite_feature_provenance.csv", index=False)

    draw_table = pd.concat(
        [frame.assign(subject=sid, replicate=np.arange(len(frame))) for sid, frame in bootstrap_draws.items()],
        ignore_index=True,
    )
    draw_table.to_csv(results_dir / "curvature_bootstrap_draws.csv", index=False)

    usable = raw.assign(
        _time_parsed=pd.to_datetime(raw[config["time_column"]], errors="coerce"),
        _butyrate_numeric=pd.to_numeric(raw[config["butyrate_column"]], errors="coerce"),
    )
    usable = usable.loc[usable["_time_parsed"].notna() & usable["_butyrate_numeric"].ge(0)]
    usable_counts = usable.groupby(config["subject_column"]).size()
    source_transition_counts = usable_counts - 1
    cohort_flow = pd.DataFrame(
        [
            {"stage": "supplied broad source table", "subjects": int(raw[config["subject_column"]].nunique()), "provenance": "observed"},
            {"stage": "at least 6 usable transitions in broad table", "subjects": int((source_transition_counts >= 6).sum()), "provenance": "computed diagnostic"},
            {"stage": "at least 7 usable transitions in broad table", "subjects": int((source_transition_counts >= 7).sum()), "provenance": "computed diagnostic"},
            {"stage": "canonical cohort selected by at least 7 usable transitions", "subjects": int(len(config["analysis_subjects"])), "provenance": "computed; configured IDs must match rule exactly"},
        ]
    )
    cohort_flow.to_csv(results_dir / "cohort_flow.csv", index=False)

    cohort_summary = pd.DataFrame(
        [
            {
                "subjects": int(len(curvature)),
                "median_samples_per_subject": float(curvature["samples"].median()),
                "median_transitions_per_subject": float(curvature["transitions"].median()),
                "median_return_coefficient": float(curvature["b"].median()),
                "median_kappa_eq": float(curvature["kappa_eq"].median()),
                "minimum_retained_sample_gap_days": float(transitions["elapsed_days"].min()),
                "median_retained_sample_gap_days": float(transitions["elapsed_days"].median()),
                "maximum_retained_sample_gap_days": float(transitions["elapsed_days"].max()),
                "robust_multistable_subjects": int(topology["robust_multistability"].sum()),
                "ci_reportable_subjects": int(curvature["ci_reportable"].sum()),
            }
        ]
    )
    cohort_summary.to_csv(results_dir / "cohort_summary.csv", index=False)

    plot_math_snapshot(figures_dir)
    plot_toy_models(toy_details, config, figures_dir)
    plot_toy_sensitivity(toy_sensitivity_summary, figures_dir)
    exemplars = plot_empirical_landscapes(landscapes, topology, figures_dir)
    plot_curvature(curvature, figures_dir)
    plot_arx_robustness(arx_primary, figures_dir)
    plot_random_walk_bias(random_walk_summary, curvature, figures_dir)
    plot_sampling_intervals(cohort, figures_dir)
    (results_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "figure_0_math_snapshot": "illustrative observation-to-drift-to-potential workflow with validation gate",
                "figure_toy_sampling": "seeded double-well and OU simulations",
                "figure_toy_sensitivity": "single-panel double-well two-minimum recovery grid; OU control retained in numerical tables",
                "figure_1_empirical_landscapes": exemplars,
                "figure_2_curvature": "apparent stepwise return coefficient; raw paired-bootstrap ranges shown descriptively",
                "figure_4_arx_robustness": "in-sample versus leave-one-transition-out loss and max-statistic adjusted permutation p-values",
                "figure_s1_random_walk_bias": "50,000-simulation random-walk null across measurement-error ratios",
                "figure_s2_sampling_intervals": "subject-level minimum, median, and maximum elapsed sampling intervals",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_reproduction_check(
        results_dir / "reproduction_check.md",
        raw,
        transitions,
        cohort_summary,
        curvature,
        topology,
        screen,
        toy_summary,
        toy_sensitivity_summary,
        input_provenance,
        config,
    )

    hash_targets = [
        path
        for path in sorted(results_dir.glob("*")) + sorted(figures_dir.glob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    ]
    manifest = {
        "package_version": __version__,
        "input": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256_file(input_path),
        "source_inputs": {
            name: {
                "path": relative,
                "sha256": sha256_file(ROOT / relative),
            }
            for name, relative in config["source_files"].items()
        },
        "processed_reference": {
            "path": config["reference_processed_file"],
            "sha256": sha256_file(ROOT / config["reference_processed_file"]),
        },
        "config_sha256": sha256_file(config_path),
        "seeds": {
            "empirical": config["seed"],
            "toy": config["toy"]["seed"],
            "toy_sensitivity": list(range(config["toy"]["sensitivity"]["seeds"])),
        },
        "software": software_versions(),
        "outputs": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in hash_targets
        },
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Reconstruction complete.")
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()

