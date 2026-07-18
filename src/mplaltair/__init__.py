"""mpl-altair: render Altair/Vega-Lite charts as native matplotlib figures."""
from __future__ import annotations

import contextlib
import os
import warnings

from ._compile import compile_chart
from ._guides import apply_axes, apply_legends
from ._layout import make_figure
from ._marks import draw_marks
from ._scales import apply_position_scale, build_scales

__all__ = ["convert", "enable"]

_STYLE_DIR = os.path.join(os.path.dirname(__file__), "_style")
_DEFAULT_STYLE_PATH = os.path.join(_STYLE_DIR, "vega-lite.mplstyle")


def _style_context(style):
    """Resolve the `style=` kwarg to a `plt.style.context(...)` (or a no-op).

    'vega-lite' (default) -> our own sheet. None/'none' -> no context at all,
    so the caller's currently-active mpl style/rcParams govern (this must be
    a true no-op, not `plt.style.context('default')`, which would clobber
    whatever the caller already has active). Anything else is passed through
    to `plt.style.context` unchanged, so built-in style names (or the
    caller's own sheet) work too.
    """
    import matplotlib.pyplot as plt

    if style is None or (isinstance(style, str) and style.lower() == "none"):
        return contextlib.nullcontext()
    if style == "vega-lite":
        return plt.style.context(_DEFAULT_STYLE_PATH)
    return plt.style.context(style)


def convert(chart_or_vl_dict, ax=None, style: str | None = "vega-lite"):
    """Render an Altair chart (or raw Vega-Lite dict) as a matplotlib Figure.

    Accepts anything with a `.to_dict()` method (an Altair Chart) or a plain
    Vega-Lite spec dict. Returns the Figure; draws into `ax` if given.

    `style` controls the mpl style scoped around figure creation + drawing:
      - 'vega-lite' (default): our bundled VL-ish style sheet.
      - None or 'none': no style context -- use whatever mpl style/rcParams
        are currently active in the caller's session.
      - anything else: passed through to `plt.style.context(...)` (a
        built-in mpl style name, or a path to the caller's own sheet).
    """
    if hasattr(chart_or_vl_dict, "to_dict"):
        vl_spec = chart_or_vl_dict.to_dict()
    else:
        vl_spec = chart_or_vl_dict

    cspec, vf_warnings = compile_chart(vl_spec)
    for w in vf_warnings:
        warnings.warn(f"vegafusion: {w}")

    with _style_context(style):
        scales = build_scales(cspec)
        fig, ax = make_figure(cspec, scales=scales, ax=ax)

        if "x" in scales:
            apply_position_scale(ax, scales["x"], "x")
        if "y" in scales:
            apply_position_scale(ax, scales["y"], "y")

        registry = draw_marks(ax, cspec, scales)
        apply_axes(ax, cspec)
        apply_legends(fig, ax, cspec, scales, registry)

    return fig


def enable():
    """Register mplaltair as the active Altair renderer (`alt.renderers.enable('mplaltair')`)."""
    import altair as alt

    alt.renderers.enable("mplaltair")
