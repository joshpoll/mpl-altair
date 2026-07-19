"""mpl-altair: render Altair/Vega-Lite charts as native matplotlib figures."""
from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import warnings

from ._compile import compile_chart
from ._guides import (
    apply_axes,
    apply_axis_titles,
    apply_facet_headers,
    apply_legends,
    apply_legends_faceted,
    unclip_gridlines,
)
from ._layout import finalize_figure_size, make_facet_figure, make_figure, target_axes_px
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

    if cspec.facet is not None and ax is not None:
        raise ValueError(
            "convert(..., ax=...) is not supported for a faceted chart: a "
            "facet needs a whole multi-Axes figure of its own, which a "
            "single caller-supplied Axes can't provide."
        )

    with _style_context(style):
        if cspec.facet is not None:
            resolve_scale = (vl_spec.get("resolve") or {}).get("scale") or {}
            if any(v == "independent" for v in resolve_scale.values()):
                warnings.warn("independent scale resolution not supported; using shared scales")
            fig = _convert_faceted(cspec)
        else:
            fig = _convert_single(cspec, ax)

    return fig


def _convert_single(cspec, ax):
    """Single-view (non-faceted) render path -- unchanged from before facets
    existed. Draws into `ax` if given, else creates its own Figure/Axes."""
    caller_supplied_ax = ax is not None

    scales = build_scales(cspec)
    # The authoritative target axes-box px size (Vega's own width/height,
    # or the exact band-derived size); a pure function of cspec/scales,
    # so it's computed once here and shared by figure sizing (initial
    # guess + final resize) and mark drawing (px->data conversion for
    # compiled bin-spacing offsets) rather than recomputed at each site.
    axes_px = target_axes_px(cspec, scales)
    fig, ax = make_figure(cspec, scales=scales, ax=ax, axes_px=axes_px)

    if "x" in scales:
        apply_position_scale(ax, scales["x"], "x")
    if "y" in scales:
        apply_position_scale(ax, scales["y"], "y")

    registry, symbol_style = draw_marks(ax, cspec, scales, axes_px)
    apply_axes(ax, cspec, scales)
    apply_axis_titles(ax, cspec)
    apply_legends(fig, ax, cspec, scales, registry, symbol_style)

    if not caller_supplied_ax:
        # Vega width/height are the INNER plot rect; grow the figure so
        # the axes box itself (not the whole figure) matches that target
        # -- see `finalize_figure_size`. When the caller supplied `ax`,
        # they own layout and we leave sizing alone.
        target_w_px, target_h_px = axes_px
        finalize_figure_size(fig, ax, target_w_px, target_h_px)

    # Must run AFTER any figure resize: mpl's auto locators recompute
    # ticks for the new physical size, and freshly created tick objects
    # come back with clipping re-enabled.
    unclip_gridlines(ax)

    return fig


def _facet_grid_shape(facet) -> tuple[int, int]:
    """(nrows, ncols) for a `FacetInfo`."""
    if facet.kind == "grid":
        return len(facet.row_values), len(facet.col_values)
    if facet.kind == "row":
        return len(facet.row_values), 1
    if facet.kind == "column":
        return 1, len(facet.col_values)
    # wrap
    ncols = facet.wrap_columns or 1
    nrows = math.ceil(len(facet.col_values) / ncols) if facet.col_values else 1
    return nrows, ncols


def _facet_panel_keys(facet, nrows: int, ncols: int) -> list[list[dict | None]]:
    """Grid (row-major, matching Vega's own panel order) of
    `{groupby_field: value}` per cell, or None for an unused trailing slot in
    a ragged wrapped grid."""
    grid: list[list[dict | None]] = [[None] * ncols for _ in range(nrows)]
    if facet.kind == "grid":
        for i, rv in enumerate(facet.row_values):
            for j, cv in enumerate(facet.col_values):
                grid[i][j] = {facet.row_field: rv, facet.col_field: cv}
    elif facet.kind == "row":
        for i, rv in enumerate(facet.row_values):
            grid[i][0] = {facet.row_field: rv}
    elif facet.kind == "column":
        for j, cv in enumerate(facet.col_values):
            grid[0][j] = {facet.col_field: cv}
    else:  # wrap
        for idx, cv in enumerate(facet.col_values):
            i, j = divmod(idx, ncols)
            grid[i][j] = {facet.col_field: cv}
    return grid


def _facet_key_fields(facet) -> list[str]:
    """The groupby field(s), in the fixed order every panel key dict/tuple
    uses: row field before column field. Both `_facet_panel_keys` (builds
    each panel's `{field: value}` key) and `_partition_facet_rows` (groups
    rows by the matching value tuple) must agree on this order."""
    return [f for f in (facet.row_field, facet.col_field) if f is not None]


def _partition_facet_rows(rows: list[dict], key_fields: list[str]) -> dict[tuple, list[dict]]:
    """Group `rows` by their `key_fields` value tuple in one O(n) pass, so
    each panel's rows are an O(1) dict lookup instead of an O(n) rescan of
    the whole facet dataset per panel."""
    partitioned: dict[tuple, list[dict]] = {}
    for r in rows:
        partitioned.setdefault(tuple(r.get(f) for f in key_fields), []).append(r)
    return partitioned


def _merge_registry(into: dict[str, list], panel_registry: dict[str, list]) -> None:
    """Fold one panel's legend-handle registry into the figure-wide one,
    deduplicating by label (every panel shares the same scale domain/colors,
    so the first panel's handle for a given category is as good as any)."""
    for name, entries in panel_registry.items():
        bucket = into.setdefault(name, [])
        seen = {label for _, label in bucket}
        for handle, label in entries:
            if label not in seen:
                bucket.append((handle, label))
                seen.add(label)


def _convert_faceted(cspec):
    """Faceted-chart render path: a grid of shared-scale Axes, one per facet
    value (or row x column combination). See `_compile.FacetInfo` for the
    recovered structure this reads.

    Only shared-scale facets are supported (`resolve: {scale: independent}`
    is warned-and-skipped by the caller, `convert()`, before this runs) --
    every panel reuses the same top-level x/y scales, so this is a
    straightforward "repeat the single-view per-Axes sequence once per
    panel" loop, plus figure-level chrome (headers, one shared legend) that
    a single Axes doesn't need.
    """
    facet = cspec.facet
    scales = build_scales(cspec)
    axes_px = target_axes_px(cspec, scales)
    nrows, ncols = _facet_grid_shape(facet)
    fig, axes_grid = make_facet_figure(nrows, ncols, axes_px)

    keys_grid = _facet_panel_keys(facet, nrows, ncols)
    key_fields = _facet_key_fields(facet)
    partitioned_rows = _partition_facet_rows(cspec.datasets.get(facet.dataset, []), key_fields)

    registry: dict[str, list] = {}
    symbol_style = None
    used_axes = []

    for i in range(nrows):
        for j in range(ncols):
            panel_ax = axes_grid[i][j]
            key = keys_grid[i][j]
            if key is None:
                # Ragged trailing slot in a wrapped grid -- no panel here.
                panel_ax.set_visible(False)
                continue
            used_axes.append(panel_ax)

            panel_rows = partitioned_rows.get(tuple(key[f] for f in key_fields), [])
            # `cell_marks` read their data via `from.data == "facet"` (Vega's
            # own name for the facet's per-group data alias) -- inject this
            # panel's filtered rows under that key so mark drawing (which
            # looks the dataset up by name off the mark's `from`) needs no
            # facet-specific code path at all.
            panel_cspec = dataclasses.replace(
                cspec, marks=facet.cell_marks,
                datasets={**cspec.datasets, "facet": panel_rows},
            )

            if "x" in scales:
                apply_position_scale(panel_ax, scales["x"], "x")
            if "y" in scales:
                apply_position_scale(panel_ax, scales["y"], "y")

            panel_registry, panel_symbol_style = draw_marks(panel_ax, panel_cspec, scales, axes_px)
            _merge_registry(registry, panel_registry)
            if panel_symbol_style is not None:
                symbol_style = panel_symbol_style

            apply_axes(panel_ax, cspec, scales)

    # Per-panel chrome (label_outer, column/row headers, the ragged-grid
    # tick-label fix) -- see `apply_facet_headers` for why it's one pass
    # over the whole grid rather than folded into the per-panel loop above.
    apply_facet_headers(axes_grid, keys_grid, facet)

    # Shared axis titles, once for the whole grid rather than per panel.
    x_title = next((a.get("title") for a in cspec.axes if a.get("orient") in ("bottom", "top")), None)
    y_title = next((a.get("title") for a in cspec.axes if a.get("orient") in ("left", "right")), None)
    if x_title:
        fig.supxlabel(x_title)
    if y_title:
        fig.supylabel(y_title)

    apply_legends_faceted(fig, used_axes, cspec, scales, registry, symbol_style)

    # Same fixed-point convergence as the single-view path, just measured off
    # one representative panel -- every panel targets the same per-panel px
    # size (shared scales), so growing the whole figure by that one panel's
    # shortfall brings every panel to size together. A facet grid's chrome
    # (per-panel headers, one shared figure-level legend) shifts by more
    # between passes than a single Axes' chrome does, so this needs a couple
    # more iterations to converge to the same tolerance.
    finalize_figure_size(fig, used_axes[0], *axes_px, iterations=4)

    for panel_ax in used_axes:
        unclip_gridlines(panel_ax)

    return fig


def enable():
    """Register mplaltair as the active Altair renderer (`alt.renderers.enable('mplaltair')`)."""
    import altair as alt

    alt.renderers.enable("mplaltair")
