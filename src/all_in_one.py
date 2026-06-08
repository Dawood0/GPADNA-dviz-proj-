"""Central registry for all project visualizations."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from vizs_src.my_visualization import create_visual as create_my_visualization


VisualFactory = Callable[..., tuple[object, str, str]]

VISUALIZATIONS: dict[str, VisualFactory] = {
    "my_visualization": create_my_visualization,
}


def create_visual(
    df: pd.DataFrame,
    visual_name: str = "my_visualization",
    **kwargs,
) -> tuple[object, str, str]:
    """Create a registered visualization by name."""
    try:
        factory = VISUALIZATIONS[visual_name]
    except KeyError as exc:
        available = ", ".join(sorted(VISUALIZATIONS))
        raise ValueError(f"Unknown visualization '{visual_name}'. Available: {available}") from exc
    return factory(df, **kwargs)
