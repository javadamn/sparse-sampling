from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or window <= 1:
        return values.copy()
    window = min(int(window), len(values) if len(values) % 2 else len(values) - 1)
    window = max(1, window)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def cumulative_potential(z_grid: np.ndarray, drift: np.ndarray) -> np.ndarray:
    z_grid = np.asarray(z_grid, dtype=float)
    drift = np.asarray(drift, dtype=float)
    potential = np.zeros_like(z_grid)
    if len(z_grid) > 1:
        widths = np.diff(z_grid)
        potential[1:] = np.cumsum(-0.5 * (drift[1:] + drift[:-1]) * widths)
    potential -= np.nanmin(potential)
    return potential


def prepare_data(
    input_path: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the supplied table and construct canonical step-based transitions."""
    raw = pd.read_csv(input_path)
    required = {
        config["subject_column"],
        config["time_column"],
        config["sample_column"],
        config["butyrate_column"],
        *config["metabolites"],
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    subject_col = config["subject_column"]
    time_col = config["time_column"]
    butyrate_col = config["butyrate_column"]
    raw[subject_col] = raw[subject_col].astype(str).str.strip()
    raw["_time"] = pd.to_datetime(raw[time_col], errors="coerce")
    raw["_butyrate"] = pd.to_numeric(raw[butyrate_col], errors="coerce")

    all_counts = (
        raw.assign(_usable=raw["_time"].notna() & raw["_butyrate"].ge(0))
        .loc[lambda x: x["_usable"]]
        .groupby(subject_col, as_index=False)
        .size()
        .rename(columns={"size": "usable_samples"})
    )
    all_counts["usable_transitions"] = (all_counts["usable_samples"] - 1).clip(lower=0)
    all_counts["passes_documented_six_transition_rule"] = (
        all_counts["usable_transitions"] >= config["minimum_transitions_documented"]
    )
    all_counts["passes_canonical_seven_transition_rule"] = (
        all_counts["usable_transitions"] >= config["minimum_transitions_canonical"]
    )

    ids = [str(x) for x in config["analysis_subjects"]]
    rule_ids = sorted(
        all_counts.loc[
            all_counts["passes_canonical_seven_transition_rule"], subject_col
        ].astype(str)
    )
    if sorted(ids) != rule_ids:
        raise ValueError(
            "Configured analysis subjects do not exactly match the auditable "
            f"seven-transition rule: configured={sorted(ids)}, rule={rule_ids}"
        )
    selected = raw.loc[raw[subject_col].isin(ids)].copy()
    selected = selected.loc[selected["_time"].notna() & selected["_butyrate"].ge(0)].copy()
    selected = selected.sort_values([subject_col, "_time", config["sample_column"]])
    selected["z"] = np.log1p(selected["_butyrate"].astype(float))

    found = set(selected[subject_col].unique())
    if found != set(ids):
        raise ValueError(f"Canonical subjects not found exactly; found={sorted(found)}")

    observations: list[pd.DataFrame] = []
    transitions: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for subject_index, sid in enumerate(ids):
        group = selected.loc[selected[subject_col] == sid].copy()
        group = group.sort_values("_time").reset_index(drop=True)
        if group["_time"].duplicated().any() or not group["_time"].is_monotonic_increasing:
            raise ValueError(f"Non-unique or non-monotonic timestamps for {sid}")
        group["subject"] = sid
        group["observation_index"] = np.arange(len(group), dtype=int)
        observations.append(group)

        z = group["z"].to_numpy(float)
        dates = group["_time"].to_numpy(dtype="datetime64[ns]")
        gaps = np.diff(dates).astype("timedelta64[D]").astype(float)
        if len(z) < config["minimum_transitions_canonical"] + 1:
            raise ValueError(f"{sid} does not have the canonical minimum sample count")
        if not np.isfinite(z).all() or np.std(z, ddof=1) <= config["minimum_state_sd"]:
            raise ValueError(f"Degenerate transformed trajectory for {sid}")

        for i in range(len(group) - 1):
            row: dict[str, Any] = {
                "subject": sid,
                "subject_index": subject_index,
                "transition_index": i,
                "sample_from": group.loc[i, config["sample_column"]],
                "sample_to": group.loc[i + 1, config["sample_column"]],
                "date_from": group.loc[i, "_time"],
                "date_to": group.loc[i + 1, "_time"],
                "elapsed_days": float(gaps[i]),
                "z": float(z[i]),
                "z_next": float(z[i + 1]),
                "dz": float(z[i + 1] - z[i]),
            }
            for metabolite in config["metabolites"]:
                row[metabolite] = pd.to_numeric(
                    pd.Series([group.loc[i, metabolite]]), errors="coerce"
                ).iloc[0]
            transitions.append(row)

        cohort_rows.append(
            {
                "subject": sid,
                "samples": int(len(group)),
                "transitions": int(len(group) - 1),
                "state_sd": float(np.std(z, ddof=1)),
                "minimum_gap_days": float(np.min(gaps)),
                "median_gap_days": float(np.median(gaps)),
                "maximum_gap_days": float(np.max(gaps)),
            }
        )

    obs = pd.concat(observations, ignore_index=True)
    transition_frame = pd.DataFrame(transitions)
    cohort = pd.DataFrame(cohort_rows)
    return raw, obs, transition_frame, cohort.merge(
        all_counts.rename(columns={subject_col: "subject"}), on="subject", how="left"
    )


def fit_return(z: np.ndarray, dz: np.ndarray) -> tuple[float, float] | None:
    z = np.asarray(z, dtype=float)
    dz = np.asarray(dz, dtype=float)
    valid = np.isfinite(z) & np.isfinite(dz)
    z = z[valid]
    dz = dz[valid]
    if len(z) < 2:
        return None
    design = np.column_stack([np.ones(len(z)), z])
    if np.linalg.matrix_rank(design) < 2:
        return None
    coef, *_ = np.linalg.lstsq(design, dz, rcond=None)
    if not np.isfinite(coef).all():
        return None
    return float(coef[0]), float(coef[1])


def _percentile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return tuple(np.quantile(values, [alpha, 1.0 - alpha]).astype(float))


def estimate_curvature(
    observations: pd.DataFrame,
    transitions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records: list[dict[str, Any]] = []
    draws: dict[str, pd.DataFrame] = {}
    B = int(config["bootstrap_replicates"])
    confidence = float(config["confidence_level"])

    for subject_index, sid in enumerate(config["analysis_subjects"]):
        group = transitions.loc[transitions["subject"] == sid]
        z = group["z"].to_numpy(float)
        dz = group["dz"].to_numpy(float)
        state = observations.loc[observations["subject"] == sid, "z"].to_numpy(float)
        sigma_state = float(np.std(state, ddof=1))
        point = fit_return(z, dz)
        if point is None:
            raise ValueError(f"Point return model is rank deficient for {sid}")
        a, b = point
        z_star = -a / b if abs(b) > 1e-12 else np.nan
        kappa = -b
        kappa_eq = kappa / (1.0 + abs(z_star)) if np.isfinite(z_star) else np.nan
        kappa_fluc = kappa / sigma_state

        rng = np.random.default_rng(int(config["seed"]) + 1000 + subject_index)
        boot_rows: list[dict[str, float]] = []
        for _ in range(B):
            sample = rng.integers(0, len(z), size=len(z))
            fit = fit_return(z[sample], dz[sample])
            if fit is None or abs(fit[1]) <= 1e-12:
                continue
            a_i, b_i = fit
            z_star_i = -a_i / b_i
            if not np.isfinite(z_star_i):
                continue
            kappa_i = -b_i
            boot_rows.append(
                {
                    "a": a_i,
                    "b": b_i,
                    "z_star": z_star_i,
                    "kappa": kappa_i,
                    "kappa_eq": kappa_i / (1.0 + abs(z_star_i)),
                    "kappa_fluc": kappa_i / sigma_state,
                }
            )
        draw_frame = pd.DataFrame(boot_rows)
        draws[sid] = draw_frame
        valid_fraction = len(draw_frame) / B
        intervals: dict[str, tuple[float, float]] = {}
        for key in ["b", "z_star", "kappa", "kappa_eq", "kappa_fluc"]:
            intervals[key] = _percentile_interval(draw_frame[key].to_numpy(float), confidence)
        b_low, b_high = intervals["b"]
        sign_stable = bool(b_high < 0.0 or b_low > 0.0)
        reportable = bool(
            valid_fraction >= config["bootstrap_valid_fraction"] and sign_stable
        )

        record: dict[str, Any] = {
            "subject": sid,
            "samples": int(len(state)),
            "transitions": int(len(z)),
            "a": a,
            "b": b,
            "z_star": z_star,
            "kappa": kappa,
            "state_sd": sigma_state,
            "kappa_eq": kappa_eq,
            "kappa_fluc": kappa_fluc,
            "bootstrap_valid": int(len(draw_frame)),
            "bootstrap_requested": B,
            "bootstrap_valid_fraction": valid_fraction,
            "b_sign_stable": sign_stable,
            "ci_reportable": reportable,
        }
        for key, (low, high) in intervals.items():
            record[f"{key}_ci_raw_low"] = low
            record[f"{key}_ci_raw_high"] = high
            record[f"{key}_ci_low"] = low if reportable else np.nan
            record[f"{key}_ci_high"] = high if reportable else np.nan
        records.append(record)
    return pd.DataFrame(records), draws


@dataclass
class Landscape:
    grid: np.ndarray
    drift: np.ndarray
    potential: np.ndarray
    minima_indices: np.ndarray
    global_minimum_index: int
    qualifies_as_multiwell: bool
    maximum_relative_barrier: float
    maximum_separation_fraction: float
    occupied_bins: int


def _local_minima(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.array([], dtype=int)
    indices = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])) + 1
    return indices.astype(int)


def reconstruct_empirical_landscape(
    z: np.ndarray,
    dz: np.ndarray,
    settings: dict[str, Any],
    state_range: tuple[float, float] | None = None,
) -> Landscape | None:
    z = np.asarray(z, dtype=float)
    dz = np.asarray(dz, dtype=float)
    valid = np.isfinite(z) & np.isfinite(dz)
    z = z[valid]
    dz = dz[valid]
    if len(z) < 3:
        return None
    lo, hi = state_range if state_range is not None else (float(np.min(z)), float(np.max(z)))
    if not np.isfinite([lo, hi]).all() or hi - lo <= 1e-12:
        return None

    n_bins = int(settings["n_bins"])
    edges = np.linspace(lo, hi, n_bins + 1)
    membership = np.clip(np.digitize(z, edges, right=False) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for index in range(n_bins):
        selected = membership == index
        counts[index] = int(selected.sum())
        if counts[index] >= int(settings["min_bin_count"]):
            means[index] = float(np.mean(dz[selected]))
    occupied = np.flatnonzero(np.isfinite(means))
    if len(occupied) < 2:
        return None

    grid = np.linspace(lo, hi, int(settings["grid_points"]))
    drift = np.interp(grid, centers[occupied], means[occupied])
    drift = moving_average(drift, int(settings["drift_smooth_window"]))
    potential = cumulative_potential(grid, drift)
    potential = moving_average(potential, int(settings["potential_smooth_window"]))
    potential -= np.min(potential)
    minima = _local_minima(potential)
    global_minimum = int(np.argmin(potential))

    state_width = hi - lo
    potential_width = float(np.max(potential) - np.min(potential))
    max_barrier = 0.0
    max_separation = 0.0
    qualifies = False
    for left_pos in range(len(minima)):
        for right_pos in range(left_pos + 1, len(minima)):
            left = int(minima[left_pos])
            right = int(minima[right_pos])
            separation = float((grid[right] - grid[left]) / state_width)
            barrier_height = float(
                np.max(potential[left : right + 1]) - max(potential[left], potential[right])
            )
            relative_barrier = barrier_height / potential_width if potential_width > 0 else 0.0
            max_barrier = max(max_barrier, relative_barrier)
            max_separation = max(max_separation, separation)
            if (
                separation >= settings["minimum_separation_fraction"]
                and relative_barrier >= settings["minimum_barrier_fraction"]
            ):
                qualifies = True
    return Landscape(
        grid=grid,
        drift=drift,
        potential=potential,
        minima_indices=minima,
        global_minimum_index=global_minimum,
        qualifies_as_multiwell=qualifies,
        maximum_relative_barrier=max_barrier,
        maximum_separation_fraction=max_separation,
        occupied_bins=int(len(occupied)),
    )


def test_multistability(
    transitions: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Landscape]]:
    settings = config["landscape"]
    B = int(config["bootstrap_replicates"])
    records: list[dict[str, Any]] = []
    landscapes: dict[str, Landscape] = {}
    for subject_index, sid in enumerate(config["analysis_subjects"]):
        group = transitions.loc[transitions["subject"] == sid]
        z = group["z"].to_numpy(float)
        dz = group["dz"].to_numpy(float)
        fixed_range = (float(np.min(z)), float(np.max(z)))
        full = reconstruct_empirical_landscape(z, dz, settings, fixed_range)
        if full is None:
            raise ValueError(f"Could not reconstruct full-data landscape for {sid}")
        landscapes[sid] = full

        rng = np.random.default_rng(int(config["seed"]) + 2000 + subject_index)
        valid = 0
        multiwell = 0
        for _ in range(B):
            sample = rng.integers(0, len(z), size=len(z))
            landscape = reconstruct_empirical_landscape(
                z[sample], dz[sample], settings, fixed_range
            )
            if landscape is None:
                continue
            valid += 1
            multiwell += int(landscape.qualifies_as_multiwell)
        valid_fraction = valid / B
        reproducibility = multiwell / valid if valid else 0.0
        robust = bool(
            full.qualifies_as_multiwell
            and valid_fraction >= config["bootstrap_valid_fraction"]
            and reproducibility >= settings["minimum_reproducibility"]
        )
        grid_width = float(full.grid[-1] - full.grid[0])
        boundary_distance = float(
            min(
                full.grid[full.global_minimum_index] - full.grid[0],
                full.grid[-1] - full.grid[full.global_minimum_index],
            )
            / grid_width
        )
        records.append(
            {
                "subject": sid,
                "occupied_bins": full.occupied_bins,
                "full_data_local_minima": int(len(full.minima_indices)),
                "full_data_qualifies_multiwell": full.qualifies_as_multiwell,
                "maximum_separation_fraction": full.maximum_separation_fraction,
                "maximum_relative_barrier": full.maximum_relative_barrier,
                "global_minimum_boundary_distance_fraction": boundary_distance,
                "bootstrap_valid": valid,
                "bootstrap_requested": B,
                "bootstrap_valid_fraction": valid_fraction,
                "bootstrap_multiwell": multiwell,
                "multiwell_reproducibility": reproducibility,
                "robust_multistability": robust,
            }
        )
    return pd.DataFrame(records), landscapes


def huber_loss(residuals: np.ndarray, delta: float) -> float:
    residuals = np.asarray(residuals, dtype=float)
    magnitude = np.abs(residuals)
    quadratic = magnitude <= delta
    losses = np.where(
        quadratic,
        0.5 * residuals**2,
        delta * (magnitude - 0.5 * delta),
    )
    return float(np.sum(losses))


def _arx_design(frame: pd.DataFrame, include_x: bool) -> tuple[np.ndarray, list[str]]:
    subjects = sorted(frame["subject"].unique())
    one_hot = np.column_stack(
        [(frame["subject"].to_numpy() == sid).astype(float) for sid in subjects]
    )
    z = frame["z"].to_numpy(float)
    subject_slopes = one_hot * z[:, None]
    design = np.column_stack([one_hot, subject_slopes])
    names = [f"intercept[{sid}]" for sid in subjects] + [f"z[{sid}]" for sid in subjects]
    if include_x:
        design = np.column_stack([design, frame["X_centered"].to_numpy(float)])
        names.append("gamma")
    return design, names


def screen_metabolites(transitions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    for metabolite in config["metabolites"]:
        frame = transitions[["subject", "z", "dz", metabolite]].copy()
        frame[metabolite] = pd.to_numeric(frame[metabolite], errors="coerce")
        frame = frame.dropna()
        frame["X_centered"] = frame[metabolite] - frame.groupby("subject")[
            metabolite
        ].transform("mean")
        if len(frame) == 0 or np.std(frame["X_centered"].to_numpy(float), ddof=1) <= 1e-12:
            results.append(
                {
                    "metabolite": metabolite,
                    "n_transitions": int(len(frame)),
                    "n_subjects": int(frame["subject"].nunique()),
                    "loss_baseline": np.nan,
                    "loss_arx": np.nan,
                    "delta_huber_loss": np.nan,
                    "gamma": np.nan,
                    "gamma_per_centered_sd": np.nan,
                }
            )
            continue
        response = frame["dz"].to_numpy(float)
        baseline, _ = _arx_design(frame, include_x=False)
        augmented, names = _arx_design(frame, include_x=True)
        if np.linalg.matrix_rank(baseline) < baseline.shape[1]:
            raise ValueError(f"Rank-deficient baseline ARX design for {metabolite}")
        if np.linalg.matrix_rank(augmented) < augmented.shape[1]:
            raise ValueError(f"Rank-deficient augmented ARX design for {metabolite}")
        beta0, *_ = np.linalg.lstsq(baseline, response, rcond=None)
        beta1, *_ = np.linalg.lstsq(augmented, response, rcond=None)
        residual0 = response - baseline @ beta0
        residual1 = response - augmented @ beta1
        loss0 = huber_loss(residual0, float(config["huber_delta"]))
        loss1 = huber_loss(residual1, float(config["huber_delta"]))
        gamma = float(beta1[names.index("gamma")])
        centered_sd = float(np.std(frame["X_centered"].to_numpy(float), ddof=1))
        results.append(
            {
                "metabolite": metabolite,
                "n_transitions": int(len(frame)),
                "n_subjects": int(frame["subject"].nunique()),
                "loss_baseline": loss0,
                "loss_arx": loss1,
                "delta_huber_loss": loss0 - loss1,
                "gamma": gamma,
                "centered_x_sd": centered_sd,
                "gamma_per_centered_sd": gamma * centered_sd,
            }
        )
    return pd.DataFrame(results).sort_values("delta_huber_loss", ascending=False).reset_index(drop=True)


def simulate_euler_maruyama(
    drift_function,
    z0: float,
    dt: float,
    n_steps: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.zeros(n_steps + 1, dtype=float)
    values[0] = z0
    noise_scale = sigma * np.sqrt(dt)
    for index in range(n_steps):
        values[index + 1] = (
            values[index]
            + drift_function(values[index]) * dt
            + noise_scale * rng.normal()
        )
    return values


def reconstruct_toy_landscape(
    sampled: np.ndarray, dt_effective: float, n_bins: int, minimum_count: int
) -> Landscape | None:
    z = np.asarray(sampled[:-1], dtype=float)
    dz = np.asarray(np.diff(sampled), dtype=float)
    lo, hi = np.percentile(z, [1.0, 99.0])
    keep = (z >= lo) & (z <= hi)
    z = z[keep]
    dz = dz[keep]
    if len(z) < max(20, n_bins * 2):
        n_bins = max(8, int(np.sqrt(max(1, len(z)))))
    edges = np.linspace(float(np.min(z)), float(np.max(z)), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    membership = np.clip(np.digitize(z, edges) - 1, 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for index in range(n_bins):
        selected = membership == index
        counts[index] = int(selected.sum())
        if counts[index] >= minimum_count:
            means[index] = float(np.mean(dz[selected]) / dt_effective)
    occupied = np.flatnonzero(np.isfinite(means))
    if len(occupied) < 3:
        return None
    grid = centers[occupied]
    drift = means[occupied]
    potential = cumulative_potential(grid, drift)
    smoothed = moving_average(potential, 5)
    minima = _local_minima(smoothed)
    return Landscape(
        grid=grid,
        drift=drift,
        potential=smoothed - np.min(smoothed),
        minima_indices=minima,
        global_minimum_index=int(np.argmin(smoothed)),
        qualifies_as_multiwell=bool(len(minima) >= 2),
        maximum_relative_barrier=np.nan,
        maximum_separation_fraction=np.nan,
        occupied_bins=int(len(occupied)),
    )


def run_toy_models(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = config["toy"]
    dt = float(settings["dt"])
    n_steps = int(float(settings["T"]) / dt)
    rng = np.random.default_rng(int(settings["seed"]))
    tilt = float(settings["tilt"])
    theta = float(settings["ou_theta"])
    mu = float(settings["ou_mu"])
    sigma = float(settings["sigma"])
    double_well = simulate_euler_maruyama(
        lambda value: -(value**3) + value + tilt,
        float(settings["double_well_z0"]),
        dt,
        n_steps,
        sigma,
        rng,
    )
    ou = simulate_euler_maruyama(
        lambda value: -theta * (value - mu),
        float(settings["ou_z0"]),
        dt,
        n_steps,
        sigma,
        rng,
    )

    series = {"double_well": double_well, "ou": ou}
    details: dict[str, Any] = {"series": series, "landscapes": {}, "sample_indices": {}}
    rows: list[dict[str, Any]] = []
    for system, values in series.items():
        for regime in ["dense", "sparse"]:
            every = int(settings[f"{regime}_every"])
            indices = np.arange(0, len(values), every, dtype=int)
            sampled = values[indices]
            landscape = reconstruct_toy_landscape(
                sampled,
                dt * every,
                int(settings["n_bins"]),
                int(settings["minimum_bin_count"]),
            )
            if landscape is None:
                raise ValueError(f"Toy reconstruction failed for {system}/{regime}")
            key = f"{system}_{regime}"
            details["landscapes"][key] = landscape
            details["sample_indices"][key] = indices
            rows.append(
                {
                    "system": system,
                    "sampling": regime,
                    "integration_steps_retained": every,
                    "effective_dt": dt * every,
                    "observations": int(len(sampled)),
                    "occupied_bins": landscape.occupied_bins,
                    "inferred_minima": int(len(landscape.minima_indices)),
                }
            )
    return pd.DataFrame(rows), details


def run_toy_sensitivity(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat toy reconstructions across prespecified noise, sampling, and seeds.

    Trajectories are simulated for all seeds in parallel. Only time points used
    by at least one requested sampling stride are retained, which makes the
    publication-grade 200-seed grid reproducible without storing dense arrays.
    """
    settings = config["toy"]
    sensitivity = settings["sensitivity"]
    dt = float(settings["dt"])
    n_steps = int(float(settings["T"]) / dt)
    tilt = float(settings["tilt"])
    theta = float(settings["ou_theta"])
    mu = float(settings["ou_mu"])
    rows: list[dict[str, Any]] = []
    n_seeds = int(sensitivity["seeds"])
    sampling_every = [int(value) for value in sensitivity["sampling_every"]]
    requested_indices = sorted(
        set().union(*(set(range(0, n_steps + 1, every)) for every in sampling_every))
    )
    index_position = {step: position for position, step in enumerate(requested_indices)}

    for sigma_index, sigma in enumerate(sensitivity["sigmas"]):
        sigma = float(sigma)
        seed_sequence = np.random.SeedSequence(int(settings["seed"]) + 10000 * sigma_index)
        double_rng, ou_rng = [np.random.default_rng(child) for child in seed_sequence.spawn(2)]
        states = {
            "double_well": np.full(n_seeds, float(settings["double_well_z0"])),
            "ou": np.full(n_seeds, float(settings["ou_z0"])),
        }
        retained = {
            name: np.empty((n_seeds, len(requested_indices)), dtype=float)
            for name in states
        }
        retained["double_well"][:, 0] = states["double_well"]
        retained["ou"][:, 0] = states["ou"]
        next_requested = 1
        noise_scale = sigma * np.sqrt(dt)
        for step in range(1, n_steps + 1):
            dw = states["double_well"]
            states["double_well"] = dw + (-(dw**3) + dw + tilt) * dt + noise_scale * double_rng.normal(size=n_seeds)
            ou = states["ou"]
            states["ou"] = ou + (-theta * (ou - mu)) * dt + noise_scale * ou_rng.normal(size=n_seeds)
            if next_requested < len(requested_indices) and step == requested_indices[next_requested]:
                retained["double_well"][:, next_requested] = states["double_well"]
                retained["ou"][:, next_requested] = states["ou"]
                next_requested += 1

        for every in sampling_every:
            positions = [index_position[step] for step in range(0, n_steps + 1, every)]
            for system in ("double_well", "ou"):
                sampled_matrix = retained[system][:, positions]
                for seed in range(n_seeds):
                    sampled = sampled_matrix[seed]
                    landscape = reconstruct_toy_landscape(
                        sampled,
                        dt * every,
                        int(settings["n_bins"]),
                        int(settings["minimum_bin_count"]),
                    )
                    minima = int(len(landscape.minima_indices)) if landscape is not None else np.nan
                    rows.append(
                        {
                            "system": system,
                            "seed": seed,
                            "sigma": sigma,
                            "integration_steps_retained": every,
                            "effective_dt": dt * every,
                            "observations": int(len(sampled)),
                            "reconstruction_valid": landscape is not None,
                            "inferred_minima": minima,
                            "classified_multistable": bool(minima >= 2) if np.isfinite(minima) else False,
                        }
                    )

    runs = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    group_columns = ["system", "sigma", "integration_steps_retained", "effective_dt"]
    for keys, group in runs.groupby(group_columns, sort=True):
        valid = group.loc[group["reconstruction_valid"]]
        system, sigma, every, effective_dt = keys
        multistable_fraction = (
            float(valid["classified_multistable"].mean()) if len(valid) else np.nan
        )
        successes = int(valid["classified_multistable"].sum()) if len(valid) else 0
        if len(valid):
            p = successes / len(valid)
            z_value = 1.959963984540054
            denominator = 1.0 + z_value**2 / len(valid)
            center = (p + z_value**2 / (2.0 * len(valid))) / denominator
            radius = z_value * np.sqrt(
                p * (1.0 - p) / len(valid) + z_value**2 / (4.0 * len(valid) ** 2)
            ) / denominator
            ci_low, ci_high = center - radius, center + radius
        else:
            ci_low, ci_high = np.nan, np.nan
        expected_fraction = (
            multistable_fraction
            if system == "double_well"
            else (1.0 - multistable_fraction if np.isfinite(multistable_fraction) else np.nan)
        )
        aggregate_rows.append(
            {
                "system": system,
                "sigma": float(sigma),
                "integration_steps_retained": int(every),
                "effective_dt": float(effective_dt),
                "runs_requested": int(len(group)),
                "runs_valid": int(len(valid)),
                "valid_fraction": float(len(valid) / len(group)),
                "median_inferred_minima": (
                    float(valid["inferred_minima"].median()) if len(valid) else np.nan
                ),
                "multistable_classification_fraction": multistable_fraction,
                "multistable_fraction_ci_low": ci_low,
                "multistable_fraction_ci_high": ci_high,
                "expected_classification_fraction": expected_fraction,
            }
        )
    summary = pd.DataFrame(aggregate_rows)
    return runs, summary

