"""Figure/Axes setup from a CompiledSpec."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from ._compile import ConcatInfo, ConcatLeaf, ConcatUnsupported

_DPI = 100
_FALLBACK_W, _FALLBACK_H = 400, 300
_MIN_INCH = 2.2  # floor for the inner-plot rect; below this, labels/legends clip
_UNSUPPORTED_CHILD_PX = (400.0, 300.0)  # placeholder size for a ConcatUnsupported slot


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


def _resolve_axis_px(explicit_value, cspec, scales: dict, scale_name: str | None, fallback: float) -> float:
    """One axis's target px, general form: `explicit_value` if it's already
    a number, else the exact band-derived size for `scale_name` if that
    names a band/point scale, else `fallback`. `target_axes_px` and
    `child_axes_px` differ only in how they come up with `explicit_value`
    (`cspec.width`/`.height` directly vs. a concat leaf's own width/height
    *signal*) and which scale name to fall back on ("x"/"y" vs. a leaf's own
    namespaced scale) -- both are thin wrappers around this.
    """
    value = explicit_value
    if not isinstance(value, (int, float)):
        value = _band_size_px(cspec, scales, scale_name)
    return value if value is not None else fallback


def target_axes_px(cspec, scales: dict | None = None) -> tuple[float, float]:
    """The target axes-box size (px): Vega's width/height, or an exact
    band-scale size when the spec has no numeric dims. Both are authoritative
    (Vega/the band math intentionally sized the axes) and are never floored;
    when neither is available, `_FALLBACK_W`/`_FALLBACK_H` are generous
    enough on their own that labels/legends don't clip, so no extra floor
    is applied."""
    scales = scales or {}
    w = _resolve_axis_px(cspec.width, cspec, scales, "x", _FALLBACK_W)
    h = _resolve_axis_px(cspec.height, cspec, scales, "y", _FALLBACK_H)
    return w, h


def child_axes_px(cspec, scales: dict, width_signal: str | None, height_signal: str | None,
                   x_scale_name: str | None, y_scale_name: str | None) -> tuple[float, float]:
    """The concat/repeat analog of `target_axes_px`, for one leaf child.

    A concat leaf's width/height isn't `cspec.width`/`cspec.height` (those
    are the whole-spec/single-view fields) -- it's driven by that leaf's own
    `encode.update.width`/`.height` *signal name* (see `ConcatLeaf`), which
    varies per leaf/nesting depth: a literal in `cspec.signals` most of the
    time (`"concat_0_width"`), the reserved name `"width"`/`"height"` when a
    concat shares one dimension across all children (binds to
    `cspec.width`/`cspec.height`, already resolved the same way a
    single-view spec's plain `width`/`height` is), or -- for a band-scale-
    driven leaf -- absent, falling back (via `_resolve_axis_px`) to the same
    band-derived math `target_axes_px` uses, just parameterized on this
    leaf's own scale name instead of the assumed global "x"/"y".
    """
    def explicit(signal_name, global_value):
        if signal_name in ("width", "height"):
            return global_value
        return cspec.signals.get(signal_name) if signal_name else None

    w = _resolve_axis_px(explicit(width_signal, cspec.width), cspec, scales, x_scale_name, _FALLBACK_W)
    h = _resolve_axis_px(explicit(height_signal, cspec.height), cspec, scales, y_scale_name, _FALLBACK_H)
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


def make_facet_figure(nrows: int, ncols: int, axes_px: tuple[float, float],
                       sharex: bool = True, sharey: bool = True):
    """Create a `nrows` x `ncols` grid of Axes sized from the per-panel
    target px (every panel targets the same size -- shared scales, or an
    independently-resolved axis whose own per-panel domain still gets the
    same nominal box size). Returns (fig, axes) with `axes` a 2D array
    (`squeeze=False`), even for a 1xN/Nx1 grid.

    `sharex`/`sharey` default to True (ordinary shared-scale facet); the
    caller passes False for an axis that's `resolve: {scale: independent}`
    -- each panel then keeps its own limits instead of being tied to the
    others'.

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
        layout="constrained", sharex=sharex, sharey=sharey, squeeze=False,
    )
    return fig, axes


def make_concat_root_figure(w_px: float, h_px: float):
    """The root Figure a concat/repeat layout's (possibly nested)
    subfigures get built inside -- just a bare, constrained-layout Figure
    sized from the layout's aggregate target px (the sum of its
    top-level children's own target sizes -- see `__init__._node_target_px`);
    `make_subfigures` does the actual grid creation, once per
    concat/nested-concat level."""
    import matplotlib.pyplot as plt

    return plt.figure(figsize=_px_to_figsize(w_px, h_px), dpi=_DPI, layout="constrained")


def make_subfigures(container, nrows: int, ncols: int,
                     width_ratios: list[float] | None, height_ratios: list[float] | None):
    """One level of a concat/repeat layout's grid: `nrows` x `ncols`
    SubFigures inside `container` (the root Figure, or a parent SubFigure
    for a nested concat-in-concat level), sized proportionally to
    `width_ratios`/`height_ratios` (each child's own aggregate target px --
    see `compute_concat_sizes`). Returns a 2D (`squeeze=False`-style) list
    of lists, mirroring `make_facet_figure`'s Axes grid shape."""
    subfigs = container.subfigures(
        nrows, ncols, squeeze=False, width_ratios=width_ratios, height_ratios=height_ratios,
    )
    return [list(row) for row in subfigs]


# -- concat/repeat geometry ------------------------------------------------
#
# A concat/repeat tree (`_compile.ConcatInfo`, recursive) is rendered as
# nested `Figure.subfigures(...)` grids. Everything here is pure geometry --
# grid shape, box-model size aggregation, subfigure creation -- with no
# knowledge of marks/scales/legends; the actual chart content per leaf is
# supplied by the caller (`__init__._convert_concat`) as callbacks, the same
# separation `apply_facet_headers` draws for a facet's chrome vs. its
# per-panel content.


def _concat_grid_shape(node: ConcatInfo) -> tuple[int, int]:
    """(nrows, ncols) for a `ConcatInfo`, mirroring `_facet_grid_shape`'s (in
    `__init__.py`) wrap math: `columns` absent -> hconcat shape (all children
    in one row); `columns == 1` -> vconcat shape (all children in one
    column); `columns == N > 1` -> a general `alt.concat(..., columns=N)`/
    `repeat` wrap grid. Both cases fall out of the same formula once
    `columns is None` is treated as "as many columns as children"."""
    n = len(node.children)
    ncols = n if node.columns is None else max(1, node.columns)
    nrows = math.ceil(n / ncols) if ncols else 1
    return nrows, ncols


def _grid_max_ratios(sizes: list[tuple[float, float]], nrows: int, ncols: int) -> tuple[list[float], list[float]]:
    """Per-column widths / per-row heights for a `nrows` x `ncols` grid whose
    cells (`sizes`, row-major) disagree on size: each column's width (row's
    height) is the MAX among the cells sharing it -- the same box-model
    reconciliation a CSS/Vega grid layout does. Shared by `compute_concat_sizes`
    (to size a `ConcatInfo` node from its children) and `render_concat_tree`
    (to build that same node's `width_ratios`/`height_ratios`) -- both need
    the identical aggregation, so it isn't computed twice.
    """
    col_w = [0.0] * ncols
    row_h = [0.0] * nrows
    for idx, (w, h) in enumerate(sizes):
        i, j = divmod(idx, ncols)
        col_w[j] = max(col_w[j], w)
        row_h[i] = max(row_h[i], h)
    return col_w, row_h


@dataclass
class SizedNode:
    """One node of a concat/repeat tree, paired with its computed size --
    the result of one bottom-up `compute_concat_sizes` pass, threaded
    straight into `render_concat_tree` rather than recomputed there (a
    leaf's size used to be recomputed once per ANCESTOR level it has, i.e.
    d+1 times for a leaf at depth d, when sizing and rendering were two
    separate recursive walks that both descended the whole subtree; this
    makes it one walk, O(nodes) total).

    `w`/`h` is this node's `correction`-adjusted size (see
    `compute_concat_sizes`) -- what actually drives `width_ratios`/
    `height_ratios` and the root figure's size. `true_w`/`true_h` is the
    UNCORRECTED target (identical to `w`/`h` for a `ConcatInfo`/
    `ConcatUnsupported` node, where no correction ever applies) -- what a
    leaf's content is actually drawn at and measured against for
    convergence; `correction` exists precisely so `w`/`h` (layout
    allocation) and `true_w`/`true_h` (drawing/measurement target) can
    differ for a `ConcatLeaf`.
    """

    node: "ConcatLeaf | ConcatInfo | ConcatUnsupported"
    w: float
    h: float
    true_w: float
    true_h: float
    children: list["SizedNode"] | None = None  # only for a ConcatInfo node


def compute_concat_sizes(node, leaf_size_fn, correction: dict[int, tuple[float, float]] | None = None) -> SizedNode:
    """One bottom-up sizing pass over a concat/repeat (sub)tree: a
    `ConcatLeaf`'s size comes from `leaf_size_fn(leaf) -> (w, h)` (the
    caller's `child_axes_px` wrapper -- kept out of this module since it
    needs `cspec`/`scales`), a `ConcatUnsupported` placeholder gets the
    fixed `_UNSUPPORTED_CHILD_PX` fallback, and a `ConcatInfo` node's size
    is its own children's box-model aggregate (`_grid_max_ratios`).

    `correction`, if given, is `id(leaf) -> (w_factor, h_factor)` (see
    `__init__._convert_concat`) applied multiplicatively to a `ConcatLeaf`'s
    true size before it feeds into any ancestor's aggregate -- so a leaf
    whose rendered size is chronically short of its true target (a roughly
    fixed chrome overhead that doesn't scale down with a small leaf's
    target) gets allocated proportionally more layout room on the next pass,
    without changing what its content is actually measured against.
    """
    if isinstance(node, ConcatInfo):
        nrows, ncols = _concat_grid_shape(node)
        child_sizes = [compute_concat_sizes(c, leaf_size_fn, correction) for c in node.children]
        col_w, row_h = _grid_max_ratios([(c.w, c.h) for c in child_sizes], nrows, ncols)
        total_w, total_h = sum(col_w), sum(row_h)
        return SizedNode(node=node, w=total_w, h=total_h, true_w=total_w, true_h=total_h, children=child_sizes)

    if isinstance(node, ConcatUnsupported):
        w, h = _UNSUPPORTED_CHILD_PX
        return SizedNode(node=node, w=w, h=h, true_w=w, true_h=h)

    true_w, true_h = leaf_size_fn(node)
    fw, fh = (correction or {}).get(id(node), (1.0, 1.0))
    return SizedNode(node=node, w=true_w * fw, h=true_h * fh, true_w=true_w, true_h=true_h)


def _spacer_padded(sf, target_w: float, target_h: float, alloc_w: float, alloc_h: float):
    """Return a SubFigure sized exactly `(target_w, target_h)` inside `sf`
    (which was allocated the larger `(alloc_w, alloc_h)`), top-left aligned
    with invisible spacer cells filling the remainder.

    Vega-Lite's concat/repeat layout defaults to `align: "each"`: every
    child keeps its OWN natural size, not stretched to match a
    differently-sized sibling that happens to share its row/column (e.g. a
    vconcat's second row is a lone 60px-wide line chart, but the first
    row -- an hconcat of a 300px scatter + 60px bar chart -- is 360px wide;
    real Vega-Lite renders the line chart at its own 60px, left-aligned,
    NOT stretched to 360px). `Figure.subfigures(..., width_ratios=...)` has
    no such concept -- it's a strict grid where every cell in a column
    shares that column's width -- so whenever a child's own target is
    smaller than what the grid allocated it, this carves out an inner 2x2
    (or 1x2/2x1, if only one dimension needs it) spacer grid and returns
    just the top-left content cell, which is what a genuinely fixed
    `align: "each"` layout engine would have given it directly.
    """
    eps = 0.5
    need_w = alloc_w - target_w > eps
    need_h = alloc_h - target_h > eps
    if not need_w and not need_h:
        return sf
    width_ratios = [target_w, alloc_w - target_w] if need_w else None
    height_ratios = [target_h, alloc_h - target_h] if need_h else None
    grid = make_subfigures(sf, 2 if need_h else 1, 2 if need_w else 1,
                            width_ratios=width_ratios, height_ratios=height_ratios)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if (i, j) != (0, 0):
                cell.set_visible(False)
    return grid[0][0]


def render_concat_tree(container, sized: SizedNode, render_leaf) -> list[tuple[SizedNode, object]]:
    """Recursively render a sized concat/repeat tree (from
    `compute_concat_sizes`) into `container` (the root Figure, or a parent
    SubFigure for a nested level): one `Figure.subfigures(...)` grid per
    `ConcatInfo` node, using its already-computed children's sizes for
    `width_ratios`/`height_ratios` (no re-sizing here -- see `SizedNode`),
    each child padded down to its own exact size via `_spacer_padded` when
    the grid allocated it more, and a nested `ConcatInfo` child recursing
    into its own (padded) SubFigure exactly like the root does.

    A `ConcatUnsupported` placeholder gets a blank, axis-off Axes plus a
    warning (facet-inside-concat nesting -- see `_compile.ConcatUnsupported`)
    and contributes nothing to the returned list. A `ConcatLeaf` is drawn by
    calling the caller-supplied `render_leaf(subfig, leaf, true_w, true_h)
    -> ax` (kept out of this module since drawing a leaf's content needs
    `cspec`/`scales`/the mark-drawing pipeline, none of which is this
    module's concern).

    Returns every leaf actually drawn as `(SizedNode, ax)` pairs, in tree
    order -- the caller measures `ax`'s rendered size against
    `SizedNode.true_w`/`.true_h` for figure-sizing convergence.
    """
    node = sized.node

    if isinstance(node, ConcatInfo):
        nrows, ncols = _concat_grid_shape(node)
        col_w, row_h = _grid_max_ratios([(c.w, c.h) for c in sized.children], nrows, ncols)
        subfigs = make_subfigures(container, nrows, ncols, width_ratios=col_w, height_ratios=row_h)

        results: list[tuple[SizedNode, object]] = []
        for idx, child_sized in enumerate(sized.children):
            i, j = divmod(idx, ncols)
            sf = _spacer_padded(subfigs[i][j], child_sized.w, child_sized.h, col_w[j], row_h[i])
            results.extend(render_concat_tree(sf, child_sized, render_leaf))

        # Ragged trailing slots in a general `columns=N` wrap grid (repeat/
        # `alt.concat(..., columns=N)` with a child count not a multiple of N).
        for idx in range(len(sized.children), nrows * ncols):
            i, j = divmod(idx, ncols)
            subfigs[i][j].set_visible(False)
        return results

    if isinstance(node, ConcatUnsupported):
        warnings.warn(node.reason)
        ax = container.subplots()
        ax.set_axis_off()
        ax.text(0.5, 0.5, "(unsupported)", ha="center", va="center", transform=ax.transAxes)
        return []

    # ConcatLeaf
    ax = render_leaf(container, node, sized.true_w, sized.true_h)
    return [(sized, ax)]


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
