from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import Landscape


COLORS = {
    "navy": "#17324D",
    "blue": "#2F6B9A",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "gray": "#697386",
    "light": "#95E8F3",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        temporary = output_dir / f"{stem}.tmp.{extension}"
        final = output_dir / f"{stem}.{extension}"
        fig.savefig(
            temporary,
            format=extension,
            bbox_inches="tight",
            facecolor="white",
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise OSError(f"Figure export failed: {temporary}")
        temporary.replace(final)
    plt.close(fig)


def plot_math_snapshot(output_dir: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))

    # A deterministic illustrative trajectory is used only to explain the
    # workflow.  It is intentionally distinct from the empirical analysis.
    rng = np.random.default_rng(20261008)
    times = np.cumsum(rng.integers(10, 48, size=18)).astype(float)
    z_values = np.empty(len(times), dtype=float)
    z_values[0] = 2.2
    for index in range(len(z_values) - 1):
        z_values[index + 1] = 2.7 + 0.58 * (z_values[index] - 2.7) + rng.normal(0, 0.33)

    axes[0, 0].plot(times, z_values, marker="o", ms=3.5, lw=1.2, color=COLORS["blue"])
    axes[0, 0].set(xlabel="observation time", ylabel=r"$z_i=\log(1+B_i)$")
    # axes[0, 0].set_title("Illustrative longitudinal observations")
    # axes[0, 0].text(
    #     0.02,
    #     0.95,
    #     "irregular and sparse",
    #     transform=axes[0, 0].transAxes,
    #     va="top",
    #     color=COLORS["gray"],
    #     fontsize=8,
    # )

    current = z_values[:-1]
    change = np.diff(z_values)
    design = np.column_stack([np.ones(len(current)), current])
    intercept, slope = np.linalg.lstsq(design, change, rcond=None)[0]
    grid = np.linspace(float(current.min()), float(current.max()), 200)
    axes[0, 1].scatter(current, change, s=24, color=COLORS["teal"], alpha=0.85)
    axes[0, 1].plot(grid, intercept + slope * grid, color=COLORS["orange"], lw=2.0)
    axes[0, 1].axhline(0, color=COLORS["gray"], lw=0.8, ls="--")
    axes[0, 1].set(xlabel=r"current state, $z_i$", ylabel=r"change, $\Delta z_i$")
    # axes[0, 1].set_title("Estimate transition-scale drift")
    axes[0, 1].text(
        0.03,
        0.05,
        r"$\Delta z_i=a+bz_i+\varepsilon_i$",
        transform=axes[0, 1].transAxes,
        color=COLORS["navy"],
    )

    potential = -(intercept * grid + 0.5 * slope * grid**2)
    potential -= np.min(potential)
    axes[1, 0].plot(grid, potential, color=COLORS["navy"], lw=2.2)
    minimum = int(np.argmin(potential))
    axes[1, 0].scatter(
        [grid[minimum]], [potential[minimum]], s=35, color=COLORS["orange"], zorder=3
    )
    axes[1, 0].set(xlabel="Observed state, z", ylabel=r"relative $\hat V(z)$")
    axes[1, 0].set_title(r"Integrate $\hat V(z)=-\int \hat f(z)\,dz$")
    # axes[1, 0].text(
    #     0.03,
    #     0.95,
    #     "defined only over the visited range",
    #     transform=axes[1, 0].transAxes,
    #     va="top",
    #     color=COLORS["gray"],
    #     fontsize=8,
    # )

    # validation = axes[1, 1]
    # validation.axis("off")
    # validation.set_title("Validate before biological interpretation", pad=8)
    # boxes = [
    #     (0.08, 0.68, "Known-ground-truth\nsampling simulations", COLORS["light"]),
    #     (0.08, 0.39, "Random-walk +\nmeasurement-error null", "#D9F0EC"),
    #     (0.08, 0.10, "Resampling, multiplicity,\nand influence checks", "#FCE5DE"),
    # ]
    # for x, y, text, color in boxes:
    #     validation.add_patch(
    #         plt.Rectangle(
    #             (x, y), 0.84, 0.19, transform=validation.transAxes, fc=color, ec="none"
    #         )
    #     )
    #     validation.text(
    #         x + 0.42,
    #         y + 0.095,
    #         text,
    #         transform=validation.transAxes,
    #         ha="center",
    #         va="center",
    #         fontsize=8.5,
    #     )
    # validation.text(
    #     0.50,
    #     0.01,
    #     "Interpret topology or return only when operating characteristics support it",
    #     transform=validation.transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=7.6,
    #     color=COLORS["gray"],
    # )

    for label, axis in zip(["A", "B", "C", "D"], axes.ravel()):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, weight="bold", fontsize=12)
    # fig.suptitle("Effective-landscape inference at the observation timescale", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_0_math_snapshot")


def plot_toy_models(details: dict[str, Any], config: dict[str, Any], output_dir: Path) -> None:
    configure_style()
    settings = config["toy"]
    dt = float(settings["dt"])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for row, regime in enumerate(["dense", "sparse"]):
        dw_key = f"double_well_{regime}"
        ou_key = f"ou_{regime}"
        indices = details["sample_indices"][dw_key]
        values = details["series"]["double_well"][indices]
        if regime == "dense":
            keep = min(len(indices), 4000)
            axes[row, 0].plot(indices[:keep] * dt, values[:keep], lw=0.8, color=COLORS["blue"])
        else:
            axes[row, 0].plot(
                indices * dt,
                values,
                lw=0.8,
                marker="o",
                ms=2.0,
                color=COLORS["orange"],
            )
        axes[row, 0].set_ylabel("state, z")
        axes[row, 0].set_xlabel("time")
        axes[row, 0].set_title(f"Double well: {regime} observations")

        for col, key, title in [
            (1, dw_key, "Double-well reconstruction"),
            (2, ou_key, "OU-control reconstruction"),
        ]:
            landscape: Landscape = details["landscapes"][key]
            axes[row, col].plot(
                landscape.grid, landscape.potential, lw=2.0, color=COLORS["navy"]
            )
            if len(landscape.minima_indices):
                axes[row, col].scatter(
                    landscape.grid[landscape.minima_indices],
                    landscape.potential[landscape.minima_indices],
                    s=28,
                    color=COLORS["teal"],
                    zorder=3,
                )
            axes[row, col].set_xlabel("state, z")
            axes[row, col].set_ylabel(r"$\hat V(z)$")
            count = len(landscape.minima_indices)
            noun = "minimum" if count == 1 else "minima"
            axes[row, col].set_title(
                f"{title}\n{regime}: {count} inferred {noun}"
            )
    for label, axis in zip(["A", "B", "C", "D", "E", "F"], axes.ravel()):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, weight="bold", fontsize=11)
    # fig.suptitle("Sparse sampling can obscure known bistability", y=1.01, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_toy_sampling")


def plot_toy_sensitivity(summary: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    fig, axis = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    frame = summary.loc[summary["system"] == "double_well"]
    pivot = frame.pivot(
        index="sigma",
        columns="integration_steps_retained",
        values="multistable_classification_fraction",
    ).sort_index(ascending=False)
    image = axis.imshow(
        pivot.to_numpy(float),
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns.astype(str))
    axis.set_yticks(np.arange(len(pivot.index)), [f"{value:.1f}" for value in pivot.index])
    axis.set_xlabel("Sampling stride (integration steps)")
    axis.set_ylabel(r"Noise amplitude, $\sigma$")
    # axis.set_title("Recovery of known double-well topology")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = float(pivot.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    runs = int(summary["runs_requested"].max())
    colorbar.set_label(f"two-minimum recovery fraction across {runs} trajectories")
    save_figure(fig, output_dir, "figure_toy_sensitivity")


def plot_empirical_landscapes(
    landscapes: dict[str, Landscape],
    topology: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    configure_style()
    ordered = topology.sort_values("global_minimum_boundary_distance_fraction")
    boundary_subject = str(ordered.iloc[0]["subject"])
    interior_subject = str(ordered.iloc[-1]["subject"])
    if interior_subject == boundary_subject and len(ordered) > 1:
        interior_subject = str(ordered.iloc[1]["subject"])

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))
    for axis, (sid, descriptor) in zip(
        axes,
        [(boundary_subject, "boundary-supported"), (interior_subject, "most interior")],
    ):
        landscape = landscapes[sid]
        axis.plot(landscape.grid, landscape.potential, color=COLORS["navy"], lw=2.2)
        minimum = landscape.global_minimum_index
        axis.scatter(
            [landscape.grid[minimum]],
            [landscape.potential[minimum]],
            color=COLORS["orange"],
            s=35,
            zorder=3,
        )
        axis.fill_between(
            landscape.grid,
            landscape.potential,
            np.max(landscape.potential),
            color=COLORS["light"],
            alpha=0.35,
            zorder=-1,
        )
        axis.set_xlabel("log(1 + butyrate), z")
        axis.set_ylabel(r"relative $\hat V(z)$")
        # axis.set_title(f"{sid}: {descriptor} minimum", pad=12)
    for label, axis in zip(["A", "B"], axes):
        axis.text(-0.17, 1.10, label, transform=axis.transAxes, weight="bold", fontsize=11)
    # fig.suptitle("Observed-range empirical landscape reconstructions", y=1.04, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_1_empirical_landscapes")
    return {"boundary_exemplar": boundary_subject, "interior_exemplar": interior_subject}


def plot_curvature(curvature: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    frame = curvature.sort_values("b").reset_index(drop=True)
    y = np.arange(len(frame))
    fig, axis = plt.subplots(figsize=(6.4, 3.7))
    axis.axvline(0, color=COLORS["gray"], lw=0.9, ls="--")
    for index, row in frame.iterrows():
        value = float(row["b"])
        low = float(row["b_ci_raw_low"])
        high = float(row["b_ci_raw_high"])
        axis.errorbar(
            value,
            index,
            xerr=np.array([[value - low], [high - value]]),
            fmt="o",
            ms=6,
            color=COLORS["orange"] if not bool(row["ci_reportable"]) else COLORS["blue"],
            ecolor=COLORS["gray"],
            capsize=3,
        )
    axis.set_yticks(y, frame["subject"].tolist())
    axis.set_xlabel(r"Apparent stepwise return coefficient, $b$")
    axis.set_ylabel("Subject")
    # axis.set_title("Descriptive return estimates with paired-bootstrap ranges")
    axis.grid(axis="x", color=COLORS["light"], lw=0.8)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_2_curvature")


def plot_modulators(screen: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    frame = screen.sort_values("delta_huber_loss", ascending=True).reset_index(drop=True)
    y = np.arange(len(frame))
    labels = frame["metabolite"].str.replace("_", " ").tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    improvement_colors = [
        COLORS["teal"] if value > 0 else COLORS["gray"]
        for value in frame["delta_huber_loss"].to_numpy(float)
    ]
    axes[0].barh(y, frame["delta_huber_loss"], color=improvement_colors)
    axes[0].axvline(0, color=COLORS["gray"], lw=0.8)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Huber-loss improvement\n(baseline − ARX)")
    axes[0].set_title("Candidate ranking")

    gamma_colors = [
        COLORS["orange"] if value < 0 else COLORS["blue"]
        for value in frame["gamma"].to_numpy(float)
    ]
    axes[1].barh(y, frame["gamma"], color=gamma_colors)
    axes[1].axvline(0, color=COLORS["gray"], lw=0.8)
    axes[1].set_xlabel(r"raw coupling estimate, $\gamma$")
    axes[1].set_title("ARX effect direction")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    for label, axis in zip(["A", "B"], axes):
        axis.text(-0.12, 1.03, label, transform=axis.transAxes, weight="bold", fontsize=11)
        axis.grid(axis="x", color=COLORS["light"], lw=0.8)
    fig.suptitle("Metabolite-informed return-dynamics screen", y=1.01, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_3_modulators")

