"""Utilities for benchmark evaluation."""

from aisteer360.evaluation.utils.data_utils import (
    build_per_example_df,
    extract_metric,
    extract_param,
    flatten_profiles,
    get_param_values,
    summarize_by_config,
    to_jsonable,
)

__all__ = [
    "build_per_example_df",
    "extract_metric",
    "extract_param",
    "flatten_profiles",
    "get_param_values",
    "summarize_by_config",
    "to_jsonable",
]

# viz utils are optional (require matplotlib)
try:
    from aisteer360.evaluation.utils.viz_utils import (
        plot_comparison_bars,
        plot_metric_by_config,
        plot_metric_heatmap,
        plot_pareto_frontier,
        plot_sensitivity,
        plot_tradeoff,
        plot_tradeoff_scatter,
        plot_tradeoff_with_pareto,
    )

    __all__.extend([
        "plot_comparison_bars",
        "plot_metric_by_config",
        "plot_metric_heatmap",
        "plot_pareto_frontier",
        "plot_sensitivity",
        "plot_tradeoff",
        "plot_tradeoff_scatter",
        "plot_tradeoff_with_pareto",
    ])
except ImportError:
    pass  # matplotlib not installed
