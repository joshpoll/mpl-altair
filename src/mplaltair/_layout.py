"""Figure/Axes setup from a CompiledSpec."""
from __future__ import annotations

_DPI = 100
_FALLBACK_W, _FALLBACK_H = 400, 300
_MIN_INCH = 2.2  # floor for the inner-plot rect; below this, labels/legends clip


def _band_size_px(cspec, scale, axis: str) -> float | None:
    """Approximate pixel extent for a band/point-scale-driven axis.

    Vega derives this from a `<axis>_step` signal + `bandspace()`; we use the
    simpler `n_cats * step / band_frac` approximation noted in the plan.
    """
    if scale is None or scale.vtype not in ("band", "point") or not scale.categories:
        return None
    n = len(scale.categories)
    step = cspec.signals.get(f"{axis}_step", 20)
    band_frac = scale.band_frac or 1.0
    return n * step / band_frac


_MIN_AXES_PX = 160  # floor applied to the TARGET axes-box px, not the figure


def target_axes_px(cspec, scales: dict | None = None) -> tuple[float, float]:
    """The target axes-box size (px): Vega's width/height, or a band-count
    estimate when the spec has no numeric dims, floored to `_MIN_AXES_PX`."""
    scales = scales or {}
    w = cspec.width
    if w is None:
        w = _band_size_px(cspec, scales.get("x"), "x") or _FALLBACK_W
    h = cspec.height
    if h is None:
        h = _band_size_px(cspec, scales.get("y"), "y") or _FALLBACK_H
    return max(w, _MIN_AXES_PX), max(h, _MIN_AXES_PX)


def make_figure(cspec, scales: dict | None = None, ax=None):
    """Create (or reuse) a Figure/Axes sized from the spec's plot dims.

    Vega width/height describe the INNER plot rect (axis labels/legends live
    outside it); the actual figure sizing that targets this rect happens
    later in `finalize_figure_size` once marks/guides are drawn (constrained
    layout needs to run at least once to know how much chrome it carves out).
    Here we just pick a starting figsize equal to the target px (a reasonable
    first guess) so `ax=None` callers get a sane initial canvas.
    """
    import matplotlib.pyplot as plt

    w, h = target_axes_px(cspec, scales)
    fig_w, fig_h = w / _DPI, h / _DPI
    fig_w = max(fig_w, _MIN_INCH)
    fig_h = max(fig_h, _MIN_INCH)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(fig_w, fig_h), dpi=_DPI, layout="constrained"
        )
    else:
        fig = ax.figure
    return fig, ax


def finalize_figure_size(fig, ax, target_w_px: float, target_h_px: float, iterations: int = 2) -> None:
    """Resize `fig` so the AXES box (not the whole figure) matches the Vega
    inner-plot-rect target size, in px.

    Vega's `width`/`height` describe the inner plot rectangle; axis
    labels/legends/colorbars live outside it. Simply sizing the whole figure
    to width/height (the old behavior) leaves the axes box far smaller once
    constrained layout carves out chrome for labels/legends -- markers then
    look inflated relative to the plot and every chart is squished.

    Two-pass (by default) fixed-point iteration: draw once to let the layout
    engine place the chrome, measure the resulting axes box in px, compute
    the shortfall, and grow the figure by exactly that shortfall. A second
    pass corrects for any small shift in chrome size caused by the first
    resize (e.g. tick label widths changing). Skipped entirely when the
    caller supplied their own `ax=` -- they own layout in that path.
    """
    target_w_px = max(target_w_px, _MIN_AXES_PX)
    target_h_px = max(target_h_px, _MIN_AXES_PX)

    for _ in range(iterations):
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        dw = target_w_px - bbox.width
        dh = target_h_px - bbox.height
        if abs(dw) <= 0.02 * target_w_px and abs(dh) <= 0.02 * target_h_px:
            break
        fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi
        new_w_px = max(fig_w_px + dw, 1.0)
        new_h_px = max(fig_h_px + dh, 1.0)
        fig.set_size_inches(new_w_px / fig.dpi, new_h_px / fig.dpi)

    fig.canvas.draw()
