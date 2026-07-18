"""Apply merged Vega axes (titles, gridlines, orientation) and legends to a
matplotlib Axes/Figure.

Legends consume the mark-drawing handle registry `{scale_name: [(handle,
label)]}` for categorical color; continuous color and size legends are built
directly from the resolved MplScale (no registry entries needed for those).
"""
from __future__ import annotations

_ORIENT_TO_AXIS = {"bottom": "x", "top": "x", "left": "y", "right": "y"}


def apply_axes(ax, cspec, scales: dict | None = None) -> None:
    for axis_entry in cspec.axes:
        orient = axis_entry.get("orient")
        which = _ORIENT_TO_AXIS.get(orient)
        if which is None:
            continue
        axis_obj = ax.xaxis if which == "x" else ax.yaxis

        title = axis_entry.get("title")
        if title:
            if which == "x":
                ax.set_xlabel(title)
            else:
                ax.set_ylabel(title)

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


def apply_legends(fig, ax, cspec, scales: dict, registry: dict, symbol_style: dict | None = None) -> None:
    """Build legends/colorbars from `cspec.legends` + the mark-drawing registry.

    `symbol_style` is the swatch style recorded by a symbol mark's size scale
    (see `_marks.draw_symbol`/`draw_marks`), used to style the size legend's
    handles in the mark's own style rather than generic filled discs.
    """
    from matplotlib.cm import ScalarMappable

    legend_objs: list = []
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
                handles.append(ax.scatter(
                    [], [], s=_px_to_pt_area(mpl_scale.size_for(v), fig.dpi),
                    facecolor=style.get("facecolor", "gray"),
                    edgecolor=style.get("edgecolor", "none"),
                    linewidths=style.get("linewidth", 0.0),
                ))
                labels.append(f"{v:g}")
            leg = ax.legend(handles, labels, title=title, loc="upper left",
                             bbox_to_anchor=(1.02, 1), frameon=False)
            _place_legend(ax, leg, legend_objs)
            continue

        if mpl_scale is not None and mpl_scale.vtype in ("linear", "log", "sqrt") and scale_name not in registry:
            # Continuous color scale -> colorbar (not a discrete-handle legend).
            cmap, norm = mpl_scale.color_cmap_norm()
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=title)
            continue

        entries = registry.get(scale_name)
        if not entries:
            continue
        handles = [h for h, _ in entries]
        labels = [l for _, l in entries]
        leg = ax.legend(handles, labels, title=title, loc="upper left",
                         bbox_to_anchor=(1.02, 1), frameon=False)
        _place_legend(ax, leg, legend_objs)
