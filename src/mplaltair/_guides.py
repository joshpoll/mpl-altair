"""Apply merged Vega axes (titles, gridlines, orientation) and legends to a
matplotlib Axes/Figure.

Legends consume the mark-drawing handle registry `{scale_name: [(handle,
label)]}` for categorical color; continuous color and size legends are built
directly from the resolved MplScale (no registry entries needed for those).
"""
from __future__ import annotations

_ORIENT_TO_AXIS = {"bottom": "x", "top": "x", "left": "y", "right": "y"}


def apply_axes(ax, cspec) -> None:
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

        ax.grid(bool(axis_entry.get("grid")), axis=which)

        if orient == "top":
            axis_obj.set_label_position("top")
            axis_obj.tick_top()
        elif orient == "right":
            axis_obj.set_label_position("right")
            axis_obj.tick_right()


def _place_legend(ax, leg, legend_objs: list) -> None:
    """Track a newly-created legend; keep any earlier legend visible.

    `ax.legend(...)` detaches whatever the axes' previous legend was, so a
    second categorical legend (e.g. color + shape) needs the first one
    re-attached via `add_artist` once the second is created.
    """
    if legend_objs:
        ax.add_artist(legend_objs[-1])
    legend_objs.append(leg)


def apply_legends(fig, ax, cspec, scales: dict, registry: dict) -> None:
    """Build legends/colorbars from `cspec.legends` + the mark-drawing registry."""
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

            for v in (lo, mid, hi):
                handles.append(ax.scatter([], [], s=_px_to_pt_area(mpl_scale.size_for(v)), color="gray"))
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
