"""Figure/Axes setup from a CompiledSpec."""
from __future__ import annotations

_DPI = 100
_FALLBACK_W, _FALLBACK_H = 400, 300
_MIN_INCH = 2.2  # floor for the inner-plot rect; below this, labels/legends clip


def _axis_step_px(cspec, scales: dict, scale, axis: str) -> float:
    """Resolve the `<axis>_step` Vega signal to a px-per-category step.

    Usually a literal (`{"name": "x_step", "value": 20}`, Vega's own default).
    For a grouped/faceted chart (an `xOffset`/`yOffset` sub-scale, e.g.
    `xOffset="grp:N"`), Vega instead emits a computed `update` signal --
    `x_step = offsetStep * bandspace(offsetCount, 0, 0) / (1 - x.paddingInner)`,
    which simplifies to `offsetStep * offsetCount / (1 - paddingInner)` since
    `bandspace(n, 0, 0) == n` -- and we only see it as an unresolved `update`
    (no literal `value`) in `cspec.signals`. Re-derive it from the offset
    scale's own (already-parsed) step/category-count in that case.
    """
    step = cspec.signals.get(f"{axis}_step")
    if step is not None:
        return step
    offset_name = f"{axis}Offset"
    offset_scale = scales.get(offset_name)
    offset_spec = cspec.scales.get(offset_name)
    if offset_scale and offset_spec and scale is not None and scale.band_frac:
        offset_step = (offset_spec.get("range") or {}).get("step")
        n_offset = len(offset_scale.categories or [])
        if isinstance(offset_step, (int, float)) and n_offset:
            return offset_step * n_offset / scale.band_frac
    return 20  # Vega's own default step


def _band_size_px(cspec, scales: dict, axis: str) -> float | None:
    """Exact pixel extent for a band/point-scale-driven axis.

    Vega derives this from a `<axis>_step` signal and
    `bandspace(count, paddingInner, paddingOuter) = count - paddingInner + 2 * paddingOuter`
    (see vega-scale's `bandSpace`), then `width/height = bandspace(...) * step`.
    Reuses the scale's already-resolved paddingInner/paddingOuter (`_scales._band_paddings`)
    instead of re-deriving them.
    """
    scale = scales.get(axis)
    if scale is None or scale.vtype not in ("band", "point") or not scale.categories:
        return None
    n = len(scale.categories)
    step = _axis_step_px(cspec, scales, scale, axis)
    bandspace = n - scale.paddingInner + 2 * scale.paddingOuter
    return bandspace * step


def target_axes_px(cspec, scales: dict | None = None) -> tuple[float, float]:
    """The target axes-box size (px): Vega's width/height, or an exact
    band-scale size when the spec has no numeric dims. Both are authoritative
    (Vega/the band math intentionally sized the axes) and are never floored;
    when neither is available, `_FALLBACK_W`/`_FALLBACK_H` are generous
    enough on their own that labels/legends don't clip, so no extra floor
    is applied."""
    scales = scales or {}
    w = cspec.width
    if w is None:
        w = _band_size_px(cspec, scales, "x")
        if w is None:
            w = _FALLBACK_W
    h = cspec.height
    if h is None:
        h = _band_size_px(cspec, scales, "y")
        if h is None:
            h = _FALLBACK_H
    return w, h


def _px_to_figsize(w_px: float, h_px: float) -> tuple[float, float]:
    """Convert a target px size to a `figsize=` (inches), floored at
    `_MIN_INCH` so labels/legends have room instead of clipping. Shared by
    `make_figure` and `make_facet_figure` -- both just pick a starting
    figsize equal to (a multiple of) the target px as a reasonable initial
    guess; the real convergence to the target happens later in
    `finalize_figure_size`, once marks/guides are drawn and constrained
    layout knows how much chrome it needs."""
    return max(w_px / _DPI, _MIN_INCH), max(h_px / _DPI, _MIN_INCH)


def make_figure(cspec, scales: dict | None = None, ax=None, axes_px: tuple[float, float] | None = None):
    """Create (or reuse) a Figure/Axes sized from the spec's plot dims.

    Vega width/height describe the INNER plot rect (axis labels/legends live
    outside it); the actual figure sizing that targets this rect happens
    later in `finalize_figure_size` once marks/guides are drawn (constrained
    layout needs to run at least once to know how much chrome it carves out).
    Here we just pick a starting figsize equal to the target px (a reasonable
    first guess) so `ax=None` callers get a sane initial canvas.

    `axes_px`, if given, is a precomputed `target_axes_px(cspec, scales)` --
    lets a caller that needs the same value elsewhere (e.g. for `draw_marks`
    and `finalize_figure_size`) compute it once and share it instead of
    recomputing it here.
    """
    import matplotlib.pyplot as plt

    w, h = axes_px if axes_px is not None else target_axes_px(cspec, scales)
    figsize = _px_to_figsize(w, h)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=_DPI, layout="constrained")
    else:
        fig = ax.figure
    return fig, ax


def make_facet_figure(nrows: int, ncols: int, axes_px: tuple[float, float]):
    """Create a `nrows` x `ncols` grid of shared-scale Axes sized from the
    per-panel target px (the same `target_axes_px` single-view figures use --
    shared scales mean every panel targets the same size). Returns (fig,
    axes) with `axes` a 2D array (`squeeze=False`), even for a 1xN/Nx1 grid.

    Mirrors `make_figure`: this picks a reasonable starting figsize (an
    initial guess), and `finalize_figure_size` (called once on a
    representative panel Axes) does the real convergence once the grid's
    chrome -- headers, a shared legend -- has been drawn.
    """
    import matplotlib.pyplot as plt

    w, h = axes_px
    figsize = _px_to_figsize(ncols * w, nrows * h)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=_DPI,
        layout="constrained", sharex=True, sharey=True, squeeze=False,
    )
    return fig, axes


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
    caller supplied their own `ax=` -- they own layout in that path. The
    caller (`target_axes_px`) is responsible for flooring unknown/degenerate
    targets; a real Vega/band-derived size is never clamped here.
    """
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
