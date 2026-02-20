"""Visualization utilities for benchmark profiles."""

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_plot_style() -> None:
    """Apply specific matplotlib rcParams for scientific style."""
    plt.rcParams.update({
        # fonts: prefer clean sans-serif
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": "medium",
        "axes.labelsize": "medium",

        # spines & ticks: show all four sides
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # grid: subtle and in the background
        "axes.grid": False,
        "grid.color": "#cbcbcb",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,

        # colors
        "axes.prop_cycle": plt.cycler(color=[
            "#348ABD", 
            "#E24A33", 
            "#988ED5", 
            "#777777",
            "#FBC15E", 
            "#8EBA42", 
            "#FFB5B8"
        ]),
    })


# consistent grey color for axis labels, ticks, and secondary text
AXIS_GREY = "#555555"


def _clean_axes(ax: plt.Axes) -> None:
    """Helper to ensure all four axis spines are visible and apply grey styling."""
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # apply grey color to axis labels and tick labels
    ax.xaxis.label.set_color(AXIS_GREY)
    ax.yaxis.label.set_color(AXIS_GREY)
    ax.tick_params(axis="both", colors=AXIS_GREY)


def plot_metric_by_config(
    summary: pd.DataFrame,
    metric: str,
    x_col: str = "config_id",
    baseline_value: float | None = None,
    baseline_std: float | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    **errorbar_kwargs: Any,
) -> plt.Axes:
    """Plot a metric with error bars across configurations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    _clean_axes(ax)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # lighter error bars to emphasize the mean point
    defaults = {
        "fmt": "o-",
        "capsize": 0,    # remove caps for cleaner look
        "elinewidth": 1,
        "markersize": 6,
        "markeredgewidth": 1,
        "markeredgecolor": "white", # separates marker from line
        "zorder": 3
    }
    defaults.update(errorbar_kwargs)

    ax.errorbar(
        summary[x_col],
        summary[mean_col],
        yerr=summary[std_col],
        **defaults,
    )

    if baseline_value is not None:
        # use a muted color for baseline reference
        ax.axhline(baseline_value, color="#444444", linestyle="--", linewidth=1, label="Baseline", zorder=1)
        if baseline_std is not None:
            ax.axhspan(
                baseline_value - baseline_std,
                baseline_value + baseline_std,
                color="#999999",
                alpha=0.1,
                edgecolor="none",
                zorder=0
            )

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or metric)

    # left-aligned title
    if title:
        ax.set_title(title, loc="left", fontweight="medium", fontsize=10)

    # grid behind data
    ax.grid(True, axis="y", zorder=-1)

    # minimal legend
    ax.legend(frameon=False, loc="best")

    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_tradeoff_scatter(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_col: str | None = None,
    label_col: str | None = None,
    baseline_row: pd.Series | None = None,
    ax: plt.Axes | None = None,
    title: str = "metric tradeoff",
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    save_path: str | Path | None = None,
    **scatter_kwargs: Any,
) -> plt.Axes:
    """Plot a scatter of two metrics showing their tradeoff."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    _clean_axes(ax)

    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"
    x_std = f"{x_metric}_std"
    y_std = f"{y_metric}_std"

    # clear distinction between data and annotations
    defaults = {
        "s": 100,
        "edgecolors": "white",
        "linewidth": 0.5,
        "alpha": 0.9,
        "zorder": 3
    }
    defaults.update(scatter_kwargs)

    # plot error bars first (behind points)
    # using 'zorder' to ensure points sit on top of error bars
    for _, row in summary.iterrows():
        ax.errorbar(
            row[x_mean],
            row[y_mean],
            xerr=row.get(x_std, 0),
            yerr=row.get(y_std, 0),
            fmt="none",
            color="#bbbbbb", # muted grey for errors
            alpha=0.6,
            capsize=0,
            elinewidth=1,
            zorder=2
        )

    if color_col is not None and color_col in summary.columns:
        defaults["c"] = summary[color_col]
        defaults["cmap"] = cmap
        scatter = ax.scatter(summary[x_mean], summary[y_mean], **defaults)
        # cleaner colorbar
        cbar = plt.colorbar(scatter, ax=ax, label=color_col)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0, colors=AXIS_GREY)
        cbar.ax.yaxis.label.set_color(AXIS_GREY)
    else:
        ax.scatter(summary[x_mean], summary[y_mean], **defaults)

    # labels with slightly less visual weight
    if label_col is not None and label_col in summary.columns:
        for _, row in summary.iterrows():
            ax.annotate(
                str(row[label_col]),
                (row[x_mean], row[y_mean]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color="#333333",
                alpha=0.8
            )

    # baseline marker
    if baseline_row is not None:
        ax.scatter(
            baseline_row[x_mean],
            baseline_row[y_mean],
            marker="X",
            s=120,
            c="#E24A33", # specific alert color
            edgecolors="white",
            linewidth=1,
            zorder=4,
            label="Baseline",
        )

    ax.set_xlabel(xlabel or x_metric)
    ax.set_ylabel(ylabel or y_metric)
    ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, zorder=-1)

    if baseline_row is not None:
        ax.legend(frameon=False)

    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_metric_heatmap(
    pivot_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    square: bool = False,
    col_label_decimals: int | None = 2,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a heatmap from a pivoted DataFrame.

    Args:
        pivot_df: Pivoted DataFrame with values to plot.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        title: Title for the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations.
        cmap: Colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        cbar_label: Label for colorbar.
        square: Whether to enforce square cells.
        col_label_decimals: Number of decimal places to round column labels to.
            Set to None to disable rounding.
        save_path: Optional path to save the figure. If provided, saves at 150 dpi.

    Returns:
        The matplotlib axes with the heatmap.
    """
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError("seaborn is required for heatmap plots") from e

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # round column labels if requested
    plot_df = pivot_df.copy()
    if col_label_decimals is not None:
        plot_df.columns = [round(c, col_label_decimals) if isinstance(c, float) else c for c in plot_df.columns]

    cbar_kws = {"label": cbar_label} if cbar_label else {}

    # seaborn integration
    sns.heatmap(
        plot_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_kws=cbar_kws,
        square=square,
        linewidths=0.5, # grid for heatmap
        linecolor='white'
    )

    if title:
        ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=AXIS_GREY)
    if ylabel:
        ax.set_ylabel(ylabel, color=AXIS_GREY)

    # apply grey to tick labels
    ax.tick_params(axis="both", colors=AXIS_GREY)

    # clean up the colorbar axes if accessible
    if ax.collections:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(size=0, colors=AXIS_GREY)
        cbar.ax.yaxis.label.set_color(AXIS_GREY)

    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_comparison_bars(
    comparison_df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_col: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str = "Value",
    colors: Sequence[str] | None = None,
    bar_width: float = 0.35,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot grouped bar chart comparing metrics across groups."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    _clean_axes(ax)

    n_groups = len(comparison_df)
    n_metrics = len(metric_cols)
    x = np.arange(n_groups)

    if colors is None:
        # use a qualitative cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    total_width = bar_width * n_metrics
    offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, n_metrics)

    for i, (col, offset) in enumerate(zip(metric_cols, offsets)):
        ax.bar(
            x + offset,
            comparison_df[col],
            bar_width,
            label=col,
            color=colors[i % len(colors)],
            edgecolor="none", # no borders on bars
            zorder=3
        )

    # strong zero line
    ax.axhline(0, color="#333333", linewidth=1, zorder=4)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df[group_col], rotation=0, ha="center")

    if title:
        ax.set_title(title, loc="left", fontweight="medium", fontsize=10)

    ax.grid(True, axis="y", zorder=0)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.1), ncol=n_metrics)

    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


_FIXED_PIPELINE_STYLES = [
    {"color": "#555555", "linestyle": "--"},   # baseline grey
    {"color": "#E24A33", "linestyle": ":"},    # red
    {"color": "#348ABD", "linestyle": "-."},   # blue
    {"color": "#988ED5", "linestyle": ":"},    # purple
    {"color": "#8EBA42", "linestyle": "-."},   # green
    {"color": "#FBC15E", "linestyle": ":"},    # amber
]


def plot_sensitivity(
    swept: pd.DataFrame,
    metric: str,
    sweep_col: str,
    baseline: pd.DataFrame | None = None,
    compare_to_pipelines: list[tuple[str, pd.DataFrame]] | None = None,
    per_trial_data: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    metric_label: str | None = None,
    sweep_label: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a single metric's sensitivity to a swept parameter.

    Displays the swept configurations as connected points with error bars and
    optional per-trial scatter. Fixed (non-swept) pipelines are overlaid as
    horizontal reference lines with ±1 std shaded bands.

    Args:
        swept: DataFrame of swept configurations with {metric}_mean and {metric}_std columns.
        metric: Name of the metric (used to find {metric}_mean and {metric}_std columns).
        sweep_col: Column name for the swept parameter (x-axis).
        baseline: Optional DataFrame with baseline row(s) for reference line. Deprecated in
            favor of ``compare_to_pipelines``; if both are provided, baseline is prepended to
            the list with label ``"baseline"``.
        compare_to_pipelines: Optional list of ``(label, summary_df)`` tuples for fixed
            (non-swept) pipelines to overlay as horizontal reference lines. Each ``summary_df``
            should contain ``{metric}_mean`` and ``{metric}_std`` columns and must represent a
            single pipeline configuration (not a ControlSpec sweep with multiple configs).
        per_trial_data: Optional DataFrame with per-trial values for scatter overlay. Should have
            columns for sweep_col and metric (the raw metric name, not _mean/_std).
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        metric_label: Label for the y-axis. Defaults to metric name.
        sweep_label: Label for the x-axis. Defaults to sweep_col.
        title: Plot title. Defaults to "{metric_label} sensitivity".
        xlim: Optional tuple of (min, max) for x-axis limits.
        ylim: Optional tuple of (min, max) for y-axis limits.
        save_path: Optional path to save the figure. If provided, saves at 150 dpi.

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If a ``compare_to_pipelines`` entry contains multiple configurations
            (i.e. results from a ControlSpec sweep rather than a fixed pipeline).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    _clean_axes(ax)

    metric_label = metric_label or metric
    sweep_label = sweep_label or sweep_col
    title = title or f"{metric_label} sensitivity"

    x_vals = swept[sweep_col].values
    y_vals = swept[f"{metric}_mean"].values
    y_err = swept[f"{metric}_std"].values

    # thin black dashed line connecting points
    ax.plot(
        x_vals,
        y_vals,
        linestyle="--",
        linewidth=0.5,
        color="black",
        zorder=2,
    )

    # individual sample points as small black dots (no jitter, 50% opacity for overlap visibility)
    if per_trial_data is not None and metric in per_trial_data.columns:
        for x_val in x_vals:
            trial_vals = per_trial_data.loc[per_trial_data[sweep_col] == x_val, metric].values
            ax.scatter(
                np.full(len(trial_vals), x_val),
                trial_vals,
                s=12,
                color="black",
                zorder=2,
                alpha=0.5,
            )

    # thin black error bars with small tails
    ax.errorbar(
        x_vals,
        y_vals,
        yerr=y_err,
        fmt="none",
        ecolor="black",
        elinewidth=0.5,
        capsize=2,
        capthick=0.5,
        zorder=3,
    )

    # medium-large white circle markers with double black ring
    ax.scatter(
        x_vals,
        y_vals,
        s=120,
        facecolor="white",
        edgecolor="black",
        linewidth=1,
        zorder=4,
    )
    # inner ring for double-ring effect
    ax.scatter(
        x_vals,
        y_vals,
        s=60,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )

    # build merged list of fixed reference pipelines (baseline first for backward compat)
    all_refs: list[tuple[str, pd.DataFrame]] = []
    if baseline is not None and not baseline.empty:
        all_refs.append(("baseline", baseline))
    if compare_to_pipelines:
        for label, ref_df in compare_to_pipelines:
            if ref_df is not None and not ref_df.empty and "config_id" in ref_df.columns:
                n_configs = ref_df["config_id"].nunique()
                if n_configs > 1:
                    raise ValueError(
                        f"compare_to_pipelines entry '{label}' contains {n_configs} configurations. "
                        f"Only fixed (non-swept) pipelines are supported — ControlSpec sweeps with "
                        f"multiple configurations should be plotted as a separate swept series."
                    )
            all_refs.append((label, ref_df))

    for i, (label, ref_df) in enumerate(all_refs):
        if ref_df is None or ref_df.empty:
            continue
        style = _FIXED_PIPELINE_STYLES[i % len(_FIXED_PIPELINE_STYLES)]
        ref_val = ref_df[f"{metric}_mean"].iloc[0]
        ref_std = ref_df[f"{metric}_std"].iloc[0]
        ax.axhline(ref_val, linewidth=1.5, label=label, **style)
        ax.axhspan(ref_val - ref_std, ref_val + ref_std, color=style["color"], alpha=0.1)

    ax.set_xlabel(sweep_label)
    ax.set_ylabel(metric_label)
    ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    ax.grid(True, axis="both", zorder=-1)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if all_refs:
        ax.legend(frameon=False, loc="best", fontsize=8)

    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


_FIXED_PIPELINE_MARKERS = [
    {"marker": "X",  "color": "black"},    # baseline
    {"marker": "s",  "color": "#E24A33"},   # red square
    {"marker": "D",  "color": "#348ABD"},   # blue diamond
    {"marker": "^",  "color": "#988ED5"},   # purple triangle
    {"marker": "P",  "color": "#8EBA42"},   # green plus
    {"marker": "v",  "color": "#FBC15E"},   # amber down-triangle
]


def plot_tradeoff(
    swept: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    sweep_col: str,
    baseline: pd.DataFrame | None = None,
    compare_to_pipelines: list[tuple[str, pd.DataFrame]] | None = None,
    per_trial_data: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    sweep_label: str | None = None,
    title: str = "tradeoff",
    cmap: str = "magma",
    show_pareto: bool = True,
    maximize_x: bool = True,
    maximize_y: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Plot a tradeoff scatter with optional Pareto frontier overlay.

    Displays swept configurations as color-coded scatter points with error bars.
    Fixed (non-swept) pipelines are overlaid as distinct markers with error bars
    and included in the Pareto frontier computation.

    Args:
        swept: DataFrame of swept configurations with metric columns.
        x_metric: Metric for x-axis (uses {x_metric}_mean and {x_metric}_std).
        y_metric: Metric for y-axis (uses {y_metric}_mean and {y_metric}_std).
        sweep_col: Column for color-coding points.
        baseline: Optional DataFrame with baseline row(s) for reference marker.
            Deprecated in favor of ``compare_to_pipelines``; if both are provided,
            baseline is prepended to the list with label ``"baseline"``.
        compare_to_pipelines: Optional list of ``(label, summary_df)`` tuples for fixed
            (non-swept) pipelines to overlay as distinct markers. Each ``summary_df``
            should contain metric mean/std columns and must represent a single pipeline
            configuration (not a ControlSpec sweep with multiple configs).
        per_trial_data: Optional DataFrame with per-trial values for scatter overlay. Should have
            columns for sweep_col, x_metric, and y_metric (raw metric names, not _mean/_std).
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        x_label: Label for x-axis. Defaults to x_metric.
        y_label: Label for y-axis. Defaults to y_metric.
        sweep_label: Label for colorbar. Defaults to sweep_col.
        title: Plot title.
        cmap: Colormap for scatter points.
        show_pareto: Whether to overlay the Pareto frontier.
        maximize_x: Whether higher x values are better (for Pareto).
        maximize_y: Whether higher y values are better (for Pareto).
        xlim: Optional tuple of (min, max) for x-axis limits.
        ylim: Optional tuple of (min, max) for y-axis limits.
        save_path: Optional path to save the figure. If provided, saves at 150 dpi.

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If a ``compare_to_pipelines`` entry contains multiple configurations
            (i.e. results from a ControlSpec sweep rather than a fixed pipeline).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    _clean_axes(ax)

    x_label = x_label or x_metric
    y_label = y_label or y_metric
    sweep_label = sweep_label or sweep_col

    x_vals = swept[f"{x_metric}_mean"].values
    y_vals = swept[f"{y_metric}_mean"].values
    c_vals = swept[sweep_col].values

    # thin black error bars with small tails (behind everything)
    for _, row in swept.iterrows():
        ax.errorbar(
            row[f"{x_metric}_mean"],
            row[f"{y_metric}_mean"],
            xerr=row[f"{x_metric}_std"],
            yerr=row[f"{y_metric}_std"],
            fmt="none",
            ecolor="black",
            elinewidth=0.5,
            capsize=2,
            capthick=0.5,
            zorder=2,
        )

    # per-trial scatter overlay (50% opacity for overlap visibility)
    if per_trial_data is not None and x_metric in per_trial_data.columns and y_metric in per_trial_data.columns:
        # get colormap for mapping sweep_col values to colors
        cmap_obj = plt.get_cmap(cmap)
        c_min, c_max = c_vals.min(), c_vals.max()
        norm = plt.Normalize(vmin=c_min, vmax=c_max)

        for sweep_val in c_vals:
            trial_mask = per_trial_data[sweep_col] == sweep_val
            trial_x = per_trial_data.loc[trial_mask, x_metric].values
            trial_y = per_trial_data.loc[trial_mask, y_metric].values
            color = cmap_obj(norm(sweep_val))
            ax.scatter(
                trial_x,
                trial_y,
                s=12,
                color=color,
                zorder=2,
                alpha=0.5,
            )

    # outer ring (larger black circle)
    ax.scatter(
        x_vals,
        y_vals,
        s=120,
        facecolor="none",
        edgecolor="black",
        linewidth=1,
        zorder=3,
    )

    # inner ring (smaller black circle)
    ax.scatter(
        x_vals,
        y_vals,
        s=60,
        facecolor="none",
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
    )

    # color-filled center
    scatter = ax.scatter(
        x_vals,
        y_vals,
        c=c_vals,
        cmap=cmap,
        s=50,
        edgecolors="none",
        zorder=5,
    )

    # build merged list of fixed reference pipelines (baseline first for backward compat)
    all_refs: list[tuple[str, pd.DataFrame]] = []
    if baseline is not None and not baseline.empty:
        all_refs.append(("baseline", baseline))
    if compare_to_pipelines:
        for label, ref_df in compare_to_pipelines:
            if ref_df is not None and not ref_df.empty and "config_id" in ref_df.columns:
                n_configs = ref_df["config_id"].nunique()
                if n_configs > 1:
                    raise ValueError(
                        f"compare_to_pipelines entry '{label}' contains {n_configs} configurations. "
                        f"Only fixed (non-swept) pipelines are supported — ControlSpec sweeps with "
                        f"multiple configurations should be plotted as a separate swept series."
                    )
            all_refs.append((label, ref_df))

    for i, (label, ref_df) in enumerate(all_refs):
        if ref_df is None or ref_df.empty:
            continue
        style = _FIXED_PIPELINE_MARKERS[i % len(_FIXED_PIPELINE_MARKERS)]
        brow = ref_df.iloc[0]
        bx, by = brow[f"{x_metric}_mean"], brow[f"{y_metric}_mean"]
        bx_err, by_err = brow[f"{x_metric}_std"], brow[f"{y_metric}_std"]
        ax.errorbar(
            bx, by,
            xerr=bx_err, yerr=by_err,
            fmt="none",
            ecolor="black",
            elinewidth=0.5,
            capsize=2,
            capthick=0.5,
            zorder=6,
        )
        ax.scatter(
            bx, by,
            marker=style["marker"],
            s=100,
            c=style["color"],
            linewidths=1.0,
            edgecolors="white",
            zorder=7,
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontweight="medium", fontsize=10)
    ax.grid(True, zorder=-1)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # colorbar - use discrete ticks if sweep values are all integers
    cbar = plt.colorbar(scatter, ax=ax, label=sweep_label)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0, colors=AXIS_GREY)
    cbar.ax.yaxis.label.set_color(AXIS_GREY)

    # check if sweep values are discrete (all integers or integer-like floats)
    unique_vals = np.unique(c_vals)
    is_discrete = all(float(v).is_integer() for v in unique_vals)
    if is_discrete and len(unique_vals) <= 10:
        cbar.set_ticks(unique_vals)
        cbar.set_ticklabels([str(int(v)) for v in unique_vals])

    # Pareto frontier (include fixed pipelines)
    if show_pareto:
        pareto_parts = [swept]
        for _, ref_df in all_refs:
            if ref_df is not None and not ref_df.empty:
                pareto_parts.append(ref_df)
        pareto_data = pd.concat(pareto_parts, ignore_index=True)
        _overlay_pareto_frontier(
            ax,
            pareto_data,
            x_metric,
            y_metric,
            maximize_x=maximize_x,
            maximize_y=maximize_y,
        )

    if all_refs:
        ax.legend(frameon=False, loc="best", fontsize=8)

    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


# alias for backward compatibility
plot_tradeoff_with_pareto = plot_tradeoff


def create_tradeoff_figure(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    sweep_col: str,
    baseline_pipeline: str = "baseline",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    sweep_label: str | None = None,
    figsize: tuple[float, float] = (12, 4),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a multi-panel tradeoff figure with sensitivity and tradeoff plots.

    Creates a figure with three panels:
        1. Left: x_metric sensitivity to sweep_col
        2. Center: y_metric sensitivity to sweep_col
        3. Right: x_metric vs y_metric tradeoff scatter

    Args:
        summary: DataFrame with metric columns ({metric}_mean, {metric}_std) and sweep_col.
        x_metric: Metric for x-axis of tradeoff plot and first sensitivity panel.
        y_metric: Metric for y-axis of tradeoff plot and second sensitivity panel.
        sweep_col: Column name for the swept parameter.
        baseline_pipeline: Name of the baseline pipeline in the "pipeline" column.
        title: Optional overall figure title.
        x_label: Label for x_metric. Defaults to x_metric.
        y_label: Label for y_metric. Defaults to y_metric.
        sweep_label: Label for sweep_col. Defaults to sweep_col.
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure. If provided, saves at 150 dpi.

    Returns:
        The matplotlib Figure with three panels.
    """
    x_label = x_label or x_metric
    y_label = y_label or y_metric
    sweep_label = sweep_label or sweep_col

    # separate baseline from swept configurations
    baseline = summary[summary["pipeline"] == baseline_pipeline] if "pipeline" in summary.columns else pd.DataFrame()
    swept = summary[summary["pipeline"] != baseline_pipeline] if "pipeline" in summary.columns else summary

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # panel 1: x_metric sensitivity
    plot_sensitivity(
        swept,
        metric=x_metric,
        sweep_col=sweep_col,
        baseline=baseline,
        ax=axes[0],
        metric_label=x_label,
        sweep_label=sweep_label,
        title=f"{x_label} sensitivity",
    )

    # panel 2: y_metric sensitivity
    plot_sensitivity(
        swept,
        metric=y_metric,
        sweep_col=sweep_col,
        baseline=baseline,
        ax=axes[1],
        metric_label=y_label,
        sweep_label=sweep_label,
        title=f"{y_label} sensitivity",
    )

    # panel 3: tradeoff scatter
    plot_tradeoff(
        swept,
        x_metric=x_metric,
        y_metric=y_metric,
        sweep_col=sweep_col,
        baseline=baseline,
        ax=axes[2],
        x_label=x_label,
        y_label=y_label,
        sweep_label=sweep_label,
        title="tradeoff",
    )

    if title:
        fig.suptitle(title, fontweight="medium", fontsize=12)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def _compute_pareto_points(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> list[tuple[float, float]]:
    """Compute Pareto-optimal points from summary data.

    A point is Pareto-optimal if no other point dominates it (i.e., no other point
    is at least as good in both dimensions and strictly better in at least one).

    Args:
        summary: DataFrame with {x_metric}_mean and {y_metric}_mean columns.
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        maximize_x: Whether higher x values are better.
        maximize_y: Whether higher y values are better.

    Returns:
        List of (x, y) tuples representing the Pareto frontier points, sorted by x.
    """
    x_mean = f"{x_metric}_mean"
    y_mean = f"{y_metric}_mean"

    points = [(row[x_mean], row[y_mean]) for _, row in summary.iterrows()]

    pareto_points = []
    for px, py in points:
        dominated = False
        for qx, qy in points:
            # check if (qx, qy) dominates (px, py)
            qx_better = (qx > px) if maximize_x else (qx < px)
            qy_better = (qy > py) if maximize_y else (qy < py)
            qx_equal_or_better = (qx >= px) if maximize_x else (qx <= px)
            qy_equal_or_better = (qy >= py) if maximize_y else (qy <= py)

            if qx_equal_or_better and qy_equal_or_better and (qx_better or qy_better):
                dominated = True
                break

        if not dominated:
            pareto_points.append((px, py))

    # sort by x for plotting
    pareto_points.sort(key=lambda p: p[0])
    return pareto_points


def _overlay_pareto_frontier(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> list[tuple[float, float]]:
    """Compute and draw the Pareto frontier on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        summary: DataFrame with {x_metric}_mean and {y_metric}_mean columns.
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        maximize_x: Whether higher x values are better.
        maximize_y: Whether higher y values are better.

    Returns:
        List of (x, y) tuples representing the Pareto frontier points.
    """
    pareto_points = _compute_pareto_points(summary, x_metric, y_metric, maximize_x, maximize_y)

    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(
            pareto_x,
            pareto_y,
            color="black",
            linestyle="-",
            linewidth=3,
            alpha=0.3,
            zorder=2,
        )
        # add "frontier" label near the midpoint of the line
        mid_idx = len(pareto_x) // 2
        ax.annotate(
            "frontier",
            (pareto_x[mid_idx], pareto_y[mid_idx]),
            xytext=(8, -8),
            textcoords="offset points",
            fontsize=8,
            color=AXIS_GREY,
            alpha=0.8,
        )

    return pareto_points


def plot_pareto_frontier(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    ax: plt.Axes | None = None,
    maximize_x: bool = True,
    maximize_y: bool = True,
    frontier_style: dict[str, Any] | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Axes, list[tuple[float, float]]]:
    """Overlay Pareto frontier on an existing or new scatter plot.

    Args:
        summary: DataFrame with {x_metric}_mean and {y_metric}_mean columns.
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        maximize_x: Whether higher x values are better.
        maximize_y: Whether higher y values are better.
        frontier_style: Dict of style kwargs for the frontier line.
        save_path: Optional path to save the figure. If provided, saves at 150 dpi.

    Returns:
        Tuple of (axes, list of Pareto frontier points as (x, y) tuples).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        _clean_axes(ax)

    if frontier_style is None:
        frontier_style = {
            "color": "black",
            "linestyle": "-",
            "linewidth": 3,
            "alpha": 0.3,
            "zorder": 2,
        }

    pareto_points = _compute_pareto_points(summary, x_metric, y_metric, maximize_x, maximize_y)

    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, label="pareto frontier", **frontier_style)
        ax.legend(frameon=False, loc="best")

    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax, pareto_points
