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
from ._layout import (
    child_axes_px,
    compute_concat_sizes,
    finalize_figure_size,
    make_concat_root_figure,
    make_facet_figure,
    make_figure,
    render_concat_tree,
    target_axes_px,
)
from ._marks import draw_marks
from ._scales import apply_position_scale, build_panel_scale, build_scales

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

    if (cspec.facet is not None or cspec.concat is not None) and ax is not None:
        kind = "faceted" if cspec.facet is not None else "concatenated"
        raise ValueError(
            f"convert(..., ax=...) is not supported for a {kind} chart: it "
            "needs a whole multi-Axes figure of its own, which a single "
            "caller-supplied Axes can't provide."
        )

    with _style_context(style):
        if cspec.facet is not None:
            fig = _convert_faceted(cspec)
        elif cspec.concat is not None:
            fig = _convert_concat(cspec)
        else:
            fig = _convert_single(cspec, ax)

    return fig


def _render_view(ax, cspec, scales: dict, axes_px: tuple[float, float], *,
                  x_scale: str = "x", y_scale: str = "y",
                  titles: bool = True, legends: bool = True, legend_container=None):
    """Draw one chart view into `ax`: apply x/y position scales (looked up
    in `scales` under `x_scale`/`y_scale` -- plain "x"/"y" for a single-view
    chart or a facet panel, a concat leaf's own namespaced scale name
    otherwise), draw its marks, apply axes chrome, and -- unless the caller
    is handling it separately -- the axis titles and legend/colorbar.

    Shared by `_convert_single`, `_convert_faceted`'s per-panel loop, and a
    concat leaf's render callback: all three draw exactly this sequence,
    differing only in which scale names to use and whether titles/legends
    are handled per-view or once for the whole figure. A facet's panels
    share one title (`fig.supxlabel`/`supylabel`) and one deduplicated
    legend (`apply_legends_faceted`) instead of repeating them per panel --
    `titles=False, legends=False` there, with the caller doing its own
    registry-merging using the `(registry, symbol_style)` this always
    returns.

    `legend_container` is the `fig.legend`/`fig.colorbar` target `apply_legends`
    needs (a concat leaf's own SubFigure, rather than the assumed
    `ax.figure`); defaults to `ax.figure` for single-view.
    """
    if x_scale in scales:
        apply_position_scale(ax, scales[x_scale], "x")
    if y_scale in scales:
        apply_position_scale(ax, scales[y_scale], "y")

    registry, symbol_style = draw_marks(ax, cspec, scales, axes_px)
    apply_axes(ax, cspec, scales)
    if titles:
        apply_axis_titles(ax, cspec)
    if legends:
        apply_legends(legend_container if legend_container is not None else ax.figure,
                       ax, cspec, scales, registry, symbol_style)
    return registry, symbol_style


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

    _render_view(ax, cspec, scales, axes_px)

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
    """Faceted-chart render path: a grid of Axes, one per facet value (or
    row x column combination). See `_compile.FacetInfo` for the recovered
    structure this reads.

    Every panel reuses the same top-level x/y scales EXCEPT an axis
    `resolve: {scale: independent}` names (`facet.independent`) -- that
    axis gets its own per-panel scale (own domain, own tick labels, not
    shared across the grid) instead; see `build_panel_scale`. Otherwise
    this is a straightforward "repeat the single-view per-Axes sequence
    once per panel" loop, plus figure-level chrome (headers, one shared
    legend) that a single Axes doesn't need.
    """
    facet = cspec.facet
    scales = build_scales(cspec)
    axes_px = target_axes_px(cspec, scales)
    nrows, ncols = _facet_grid_shape(facet)
    share_x = "x" not in facet.independent
    share_y = "y" not in facet.independent
    fig, axes_grid = make_facet_figure(nrows, ncols, axes_px, sharex=share_x, sharey=share_y)

    keys_grid = _facet_panel_keys(facet, nrows, ncols)
    key_fields = _facet_key_fields(facet)
    partitioned_rows = _partition_facet_rows(cspec.datasets.get(facet.dataset, []), key_fields)

    # Scale NAME to look up for x/y position -- the shared top-level "x"/"y"
    # normally, or an independent axis's own local scale name (e.g.
    # "child_y"), which gets a freshly-built per-panel entry injected into
    # `panel_scales` below rather than sharing one `scales["y"]` MplScale
    # across every panel.
    x_scale_name = facet.independent["x"].local_name if "x" in facet.independent else "x"
    y_scale_name = facet.independent["y"].local_name if "y" in facet.independent else "y"

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

            panel_scales = scales
            if facet.independent:
                panel_scales = dict(scales)
                for info in facet.independent.values():
                    panel_scales[info.local_name] = build_panel_scale(
                        info.local_name, info.scale_spec, panel_rows, info.field,
                    )

            panel_registry, panel_symbol_style = _render_view(
                panel_ax, panel_cspec, panel_scales, axes_px,
                x_scale=x_scale_name, y_scale=y_scale_name, titles=False, legends=False,
            )
            _merge_registry(registry, panel_registry)
            if panel_symbol_style is not None:
                symbol_style = panel_symbol_style

    # Per-panel chrome (interior tick-label hiding on SHARED axes only,
    # column/row headers) -- see `apply_facet_headers` for why it's one pass
    # over the whole grid rather than folded into the per-panel loop above.
    apply_facet_headers(axes_grid, keys_grid, facet, share_x=share_x, share_y=share_y)

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


def _render_concat_leaf(container, leaf, cspec, scales: dict, true_w: float, true_h: float):
    """`render_leaf` callback for `_layout.render_concat_tree`: draw one
    concat/repeat leaf into a fresh Axes inside `container` (its already-
    sized SubFigure), via the same `_render_view` sequence every other
    render path uses -- parameterized on this leaf's own scale NAMES
    (`concat_0_x`, ...) rather than the assumed global "x"/"y", since
    `scales` (built once for the whole compiled spec) already has an entry
    under whatever name each leaf's marks actually reference.

    Concat children have independent scales by default (unlike a facet's
    panels), so each leaf gets its OWN ax-level legend rather than one
    figure-level dedup -- declared simplification: if two leaves happened
    to share one resolve:"shared" color scale, each still gets its own
    legend rather than a single merged one. Not exercised by any of our
    concat gallery entries (none use a shared color channel).
    """
    ax = container.subplots()
    leaf_cspec = dataclasses.replace(cspec, marks=leaf.marks, axes=leaf.axes)
    _render_view(ax, leaf_cspec, scales, (true_w, true_h),
                 x_scale=leaf.x_scale, y_scale=leaf.y_scale, legend_container=container)
    return ax


def _convert_concat(cspec, iterations: int = 8, damping: float = 0.5):
    """Concat/hconcat/vconcat/repeat render path: a (possibly nested) grid
    of independently-scaled single-view Axes, one per child. See
    `_compile.ConcatInfo`/`ConcatLeaf` for the recovered structure this
    reads; all of the actual grid/box-model geometry (sizing, subfigure
    creation, ragged-grid handling) lives in `_layout.compute_concat_sizes`/
    `render_concat_tree` -- this function is the orchestration around it:
    build a per-leaf `child_axes_px` sizing callback, run the render, decide
    whether another pass is needed. A faceted child (facet-inside-concat) is
    the one shape not rendered; see `_compile.ConcatUnsupported`.

    Sizing is a fixed-point loop over the WHOLE render (not just a resize of
    an already-drawn figure, the way `finalize_figure_size` works): each
    pass rebuilds the figure from scratch with a per-leaf `correction`
    factor -- `id(leaf) -> (w_factor, h_factor)` -- folded (by
    `compute_concat_sizes`) into that leaf's contribution to its ancestors'
    width_ratios/height_ratios. A leaf whose rendered size falls short of
    its true target because of a roughly-fixed chrome overhead (tick label
    widths, ...) that doesn't scale down with a small leaf's target size
    gets a bigger correction factor and more room on the next pass. This is
    necessary (not just nicer) because a single isotropic resize of an
    already-built figure cannot independently satisfy two leaves whose
    chrome-to-target ratios differ -- rebuilding with adjusted RATIOS is the
    only way to give a small, chrome-heavy leaf relatively more room without
    also over-growing an already-converged leaf.

    The naive update (`factor *= target/actual`, no damping) overshoots and
    OSCILLATES rather than converging: a leaf far below target gets a huge
    correction, which -- compounded with every ancestor's ratios and the
    root figure size all changing at once -- overshoots on the next pass,
    triggering a large correction the other way, and so on indefinitely
    (verified empirically: a 60px-target leaf cycled through ~10px / 287px /
    23px / 136px actual over 4 passes and never settled). Raising the
    update to the `damping` power (0.5, i.e. sqrt) tames this into
    consistent convergence (empirically <3% error within 4-5 passes on the
    concat gallery entries).
    """
    scales = build_scales(cspec)

    def leaf_size(leaf) -> tuple[float, float]:
        return child_axes_px(cspec, scales, leaf.width_signal, leaf.height_signal, leaf.x_scale, leaf.y_scale)

    def render_leaf(container, leaf, true_w, true_h):
        return _render_concat_leaf(container, leaf, cspec, scales, true_w, true_h)

    correction: dict[int, tuple[float, float]] = {}
    fig = None
    leaf_axes: list[tuple] = []

    for _ in range(iterations):
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)
        sized_root = compute_concat_sizes(cspec.concat, leaf_size, correction)
        fig = make_concat_root_figure(sized_root.w, sized_root.h)
        leaf_axes = render_concat_tree(fig, sized_root, render_leaf)

        if not leaf_axes:
            break
        fig.canvas.draw()
        max_err = 0.0
        for sized, ax in leaf_axes:
            bbox = ax.get_window_extent()
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            target_w, target_h = sized.true_w, sized.true_h
            max_err = max(max_err, abs(bbox.width - target_w) / target_w, abs(bbox.height - target_h) / target_h)
            leaf_id = id(sized.node)
            fw, fh = correction.get(leaf_id, (1.0, 1.0))
            correction[leaf_id] = (
                fw * (target_w / bbox.width) ** damping,
                fh * (target_h / bbox.height) ** damping,
            )
        if max_err <= 0.02:
            break

    for _, ax in leaf_axes:
        unclip_gridlines(ax)

    return fig


def enable():
    """Register mplaltair as the active Altair renderer (`alt.renderers.enable('mplaltair')`)."""
    import altair as alt

    alt.renderers.enable("mplaltair")
