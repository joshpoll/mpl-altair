"""Apply merged Vega axes (titles, gridlines, orientation) and legends to a
matplotlib Axes/Figure.

Legends consume the mark-drawing handle registry `{scale_name: [(handle,
label)]}` for categorical color; continuous color and size legends are built
directly from the resolved MplScale (no registry entries needed for those).
"""
from __future__ import annotations

_ORIENT_TO_AXIS = {"bottom": "x", "top": "x", "left": "y", "right": "y"}


def apply_axes(ax, cspec, scales: dict | None = None) -> None:
    """Apply grid + orientation from `cspec.axes` to `ax` (not the axis
    title -- see `apply_axis_titles`). A faceted chart draws this per-panel;
    the single-view path additionally calls `apply_axis_titles` right after,
    since a facet's panels share one title (shown once via
    `fig.supxlabel`/`fig.supylabel`) instead of repeating it on every Axes."""
    for axis_entry in cspec.axes:
        orient = axis_entry.get("orient")
        which = _ORIENT_TO_AXIS.get(orient)
        if which is None:
            continue
        axis_obj = ax.xaxis if which == "x" else ax.yaxis

        grid_on = bool(axis_entry.get("grid"))
        ax.grid(grid_on, axis=which)
        # Vega draws log-axis gridlines at every tick (2..9 per decade), not
        # just the decades -- mirror that with mpl's minor gridlines.
        mpl_scale = (scales or {}).get(axis_entry.get("scale"))
        if mpl_scale is not None and mpl_scale.vtype == "log":
            ax.grid(grid_on, axis=which, which="minor")

        if orient == "top":
            axis_obj.set_label_position("top")
            axis_obj.tick_top()
        elif orient == "right":
            axis_obj.set_label_position("right")
            axis_obj.tick_right()


def apply_axis_titles(ax, cspec) -> None:
    """Set the x/y axis label from `cspec.axes`' `title` entries. Split out
    of `apply_axes` because a faceted chart's panels don't call this (they
    share one title figure-wide instead -- see `_convert_faceted`)."""
    for axis_entry in cspec.axes:
        which = _ORIENT_TO_AXIS.get(axis_entry.get("orient"))
        title = axis_entry.get("title")
        if which is None or not title:
            continue
        if which == "x":
            ax.set_xlabel(title)
        else:
            ax.set_ylabel(title)


def apply_facet_headers(axes_grid, keys_grid, facet, share_x: bool = True, share_y: bool = True) -> None:
    """Per-panel chrome for a faceted grid: hide interior tick labels on a
    SHARED axis, set column-header titles / row-header side labels (or, for
    a wrapped facet, a per-cell title).

    Tick-label hiding is done directly (`ax.tick_params(labelbottom=False)`/
    `labelleft=False`) rather than via mpl's `Axes.label_outer()`, for two
    reasons: (1) `label_outer` always hides both axes based on grid
    position alone, with no way to hide only one of x/y -- wrong for an
    independently-resolved axis (`share_x`/`share_y` False), whose every
    panel needs its own tick labels visible regardless of grid position;
    (2) computing "last POPULATED row per column" up front (below) instead
    of calling `label_outer` (which only knows the grid's actual last row)
    handles a ragged wrapped grid correctly in the same pass, rather than
    needing a separate patch-up pass afterward.

    `axes_grid`/`keys_grid` are the 2D (nrows x ncols) arrays built by
    `_facet_panel_keys` et al in `__init__.py` -- `keys_grid[i][j]` is
    `None` for an unused trailing slot in a ragged wrapped grid.
    """
    nrows = len(axes_grid)
    ncols = len(axes_grid[0]) if nrows else 0

    last_row_in_col = {}
    for j in range(ncols):
        populated = [i for i in range(nrows) if keys_grid[i][j] is not None]
        if populated:
            last_row_in_col[j] = populated[-1]

    for i in range(nrows):
        for j in range(ncols):
            key = keys_grid[i][j]
            if key is None:
                continue
            panel_ax = axes_grid[i][j]

            if share_x and i != last_row_in_col.get(j):
                panel_ax.tick_params(labelbottom=False)
            if share_y and j != 0:
                panel_ax.tick_params(labelleft=False)

            if facet.kind == "wrap":
                panel_ax.set_title(str(key[facet.col_field]))
            else:
                if i == 0 and facet.col_field is not None:
                    panel_ax.set_title(str(key[facet.col_field]))
                if j == ncols - 1 and facet.row_field is not None:
                    panel_ax.annotate(
                        str(key[facet.row_field]), xy=(1.02, 0.5),
                        xycoords="axes fraction", rotation=-90,
                        va="center", ha="left",
                    )

    if facet.kind == "wrap":
        for j in range(ncols):
            populated_rows = [i for i in range(nrows) if keys_grid[i][j] is not None]
            if populated_rows and populated_rows[-1] != nrows - 1:
                axes_grid[populated_rows[-1]][j].xaxis.set_tick_params(labelbottom=True)


def unclip_gridlines(ax) -> None:
    """Unclip gridlines so the ones at the axes limits always render.

    A gridline at a domain edge sits exactly on the axes clip boundary, and
    whether its 1px stroke survives rasterization is pixel-alignment luck
    (e.g. the x-max gridline vanishing while the y-max one renders). Vega
    always draws boundary gridlines. Gridlines only ever span the axes rect,
    so unclipping cannot bleed anywhere else.
    """
    for tick in (*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks(),
                 *ax.xaxis.get_minor_ticks(), *ax.yaxis.get_minor_ticks()):
        tick.gridline.set_clip_on(False)


def _place_legend(ax, leg, legend_objs: list) -> None:
    """Track a newly-created legend; keep any earlier legend visible.

    `ax.legend(...)` detaches whatever the axes' previous legend was, so a
    second categorical legend (e.g. color + shape) needs the first one
    re-attached via `add_artist` once the second is created.
    """
    if legend_objs:
        ax.add_artist(legend_objs[-1])
    legend_objs.append(leg)


def _legend_items(cspec, scales: dict, registry: dict, symbol_style: dict | None, handle_ax):
    """Resolve `cspec.legends` + the mark-drawing registry into a sequence of
    drawable legend pieces, shared by `apply_legends` (ax-level) and
    `apply_legends_faceted` (figure-level) -- everything about a legend spec
    that doesn't depend on *where* it's attached lives here.

    Yields `(kind, payload, title)`:
      - `("colorbar", scalar_mappable, title)` for a continuous color scale.
      - `("handles", (handles, labels), title)` for a size or categorical
        legend -- the two only differ in how their handles are built, not in
        how the caller attaches them, so both funnel into one shape.

    `handle_ax` supplies the `.scatter([], [], ...)` factory a size legend
    uses for its proxy swatches (any Axes works; nothing about the phantom
    handles is ever actually drawn on it) and its Figure's `dpi`.
    """
    from matplotlib.cm import ScalarMappable

    for legend_spec in cspec.legends:
        scale_name = (
            legend_spec.get("fill")
            or legend_spec.get("stroke")
            or legend_spec.get("size")
            or legend_spec.get("shape")
        )
        if scale_name is None:
            continue
        mpl_scale = scales.get(scale_name)
        title = legend_spec.get("title", scale_name)

        if "size" in legend_spec:
            if mpl_scale is None or not isinstance(mpl_scale.domain, list):
                continue
            lo, hi = mpl_scale.domain[0], mpl_scale.domain[1]
            mid = (lo + hi) / 2
            handles, labels = [], []
            from ._marks import _px_to_pt_area

            style = symbol_style or {}
            for v in (lo, mid, hi):
                handles.append(handle_ax.scatter(
                    [], [], s=_px_to_pt_area(mpl_scale.size_for(v), handle_ax.figure.dpi),
                    facecolor=style.get("facecolor", "gray"),
                    edgecolor=style.get("edgecolor", "none"),
                    linewidths=style.get("linewidth", 0.0),
                ))
                labels.append(f"{v:g}")
            yield "handles", (handles, labels), title
            continue

        if mpl_scale is not None and mpl_scale.vtype in ("linear", "log", "sqrt") and scale_name not in registry:
            # Continuous color scale -> colorbar (not a discrete-handle legend).
            cmap, norm = mpl_scale.color_cmap_norm()
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            yield "colorbar", sm, title
            continue

        entries = registry.get(scale_name)
        if not entries:
            continue
        yield "handles", ([h for h, _ in entries], [l for _, l in entries]), title


def apply_legends(fig, ax, cspec, scales: dict, registry: dict, symbol_style: dict | None = None) -> None:
    """Build legends/colorbars from `cspec.legends` + the mark-drawing registry.

    `symbol_style` is the swatch style recorded by a symbol mark's size scale
    (see `_marks.draw_symbol`/`draw_marks`), used to style the size legend's
    handles in the mark's own style rather than generic filled discs.
    """
    legend_objs: list = []
    for kind, payload, title in _legend_items(cspec, scales, registry, symbol_style, ax):
        if kind == "colorbar":
            fig.colorbar(payload, ax=ax, label=title)
            continue
        handles, labels = payload
        # Placement is intentionally ax-anchored here (each panel/single-view
        # Axes gets its own legend slot just outside its own right edge) --
        # `apply_legends_faceted` instead anchors to the whole figure.
        leg = ax.legend(handles, labels, title=title, loc="upper left",
                         bbox_to_anchor=(1.02, 1), frameon=False)
        _place_legend(ax, leg, legend_objs)


def apply_legends_faceted(fig, axes_list, cspec, scales: dict, registry: dict, symbol_style: dict | None = None) -> None:
    """Figure-level equivalent of `apply_legends` for a faceted chart: one
    `fig.legend`/`fig.colorbar` shared across all panels instead of a legend
    per Axes (the registry is already deduplicated across panels by the
    caller). `axes_list` is every visible panel Axes, used as the colorbar's
    `ax=` (mpl shrinks the whole panel grid to make room) and as the anchor
    for size-legend proxy handles.
    """
    ref_ax = axes_list[0]
    for kind, payload, title in _legend_items(cspec, scales, registry, symbol_style, ref_ax):
        if kind == "colorbar":
            fig.colorbar(payload, ax=axes_list, label=title)
            continue
        handles, labels = payload
        # Placement is intentionally figure-anchored here (one legend for
        # the whole panel grid, reserved as its own outside-right column by
        # constrained layout) -- `apply_legends` instead anchors to the
        # single owning Axes.
        fig.legend(handles, labels, title=title, loc="outside right upper")
