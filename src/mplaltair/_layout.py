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


def make_figure(cspec, scales: dict | None = None, ax=None):
    """Create (or reuse) a Figure/Axes sized from the spec's plot dims.

    Vega width/height describe the inner plot rect; we map them to figsize at
    a fixed dpi. Band-driven charts without numeric dims fall back to a
    band-count-derived estimate, else a default size.
    """
    import matplotlib.pyplot as plt

    scales = scales or {}
    w = cspec.width
    if w is None:
        w = _band_size_px(cspec, scales.get("x"), "x") or _FALLBACK_W
    h = cspec.height
    if h is None:
        h = _band_size_px(cspec, scales.get("y"), "y") or _FALLBACK_H

    fig_w, fig_h = w / _DPI, h / _DPI
    # Minimum inner-plot floor: small band charts (e.g. 3 categories -> ~66px
    # wide) render too small for axis labels/legends to fit. Width and height
    # come from independent scales (category count vs. plot-height signal),
    # so clamp each dimension to the floor independently rather than scaling
    # both together preserving aspect -- a uniform scale-up would balloon the
    # unrelated dimension too (e.g. a 3-category bar chart's already-correct
    # 300px height would blow out to ~990px just because its width is tiny).
    # This is a deliberate divergence from the plan's literal "preserve
    # aspect" wording -- see M5 report.
    fig_w = max(fig_w, _MIN_INCH)
    fig_h = max(fig_h, _MIN_INCH)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(fig_w, fig_h), dpi=_DPI, layout="constrained"
        )
    else:
        fig = ax.figure
    return fig, ax
