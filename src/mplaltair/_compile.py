"""Compile an Altair/Vega-Lite spec down to a resolved, mpl-friendly structure.

Pipeline: Vega-Lite dict -> Vega dict (vl-convert) -> pre-transformed Vega dict
(vegafusion, datasets inlined) -> CompiledSpec (our own resolved view).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import vl_convert as vlc
import vegafusion as vf


def vl_to_vega(vl_spec: dict) -> dict:
    """Compile a Vega-Lite spec to a Vega spec via vl-convert."""
    return vlc.vegalite_to_vega(vl_spec)


def inline_transforms(vg_spec: dict) -> tuple[dict, list]:
    """Evaluate all Vega transforms, inlining resulting datasets.

    Returns (pre_transformed_spec, warnings).
    """
    pre, warnings = vf.runtime.pre_transform_spec(vg_spec, preserve_interactivity=False)
    if isinstance(pre, str):
        pre = json.loads(pre)
    return pre, warnings


def get_plot_dims(spec_dict: dict) -> tuple[float | None, float | None]:
    """Return (width, height) from a Vega spec's top-level dims if numeric.

    Band-scale-driven charts often omit a numeric width/height (a signal expr
    is used instead); those are handled later in _layout with a fallback.
    """
    width = spec_dict.get("width")
    height = spec_dict.get("height")
    width = width if isinstance(width, (int, float)) else None
    height = height if isinstance(height, (int, float)) else None
    return width, height


def _merge_axes(axes: list[dict]) -> list[dict]:
    """Merge duplicate (scale, orient) axis entries into one.

    Vega emits a grid-only entry (title null, grid may be true) and a labeled
    entry (has title, grid false) per (scale, orient). We merge: grid=True if
    any entry has grid true; title/format taken from whichever entry has them.
    """
    merged: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for ax in axes:
        key = (ax.get("scale"), ax.get("orient"))
        if key not in merged:
            merged[key] = {}
            order.append(key)
        out = merged[key]
        for k, v in ax.items():
            if k == "grid":
                out["grid"] = out.get("grid", False) or bool(v)
            elif v is not None and k not in out:
                out[k] = v
    return [merged[k] for k in order]


def _is_cell_group(m: dict) -> bool:
    """True for the facet's `cell` group mark (the one wrapping the real,
    per-panel marks) -- Vega-Lite tags it `style: "cell"`."""
    if m.get("type") != "group":
        return False
    style = m.get("style")
    if isinstance(style, list):
        return "cell" in style
    return style == "cell"


# Header/footer role groups that carry the facet's recovered axis definitions
# (titles, grid) -- the `cell` group itself also carries a (grid-only) axis
# entry per continuous position scale, so it's included by the caller too.
_FACET_HEADER_ROLES = ("row-header", "column-header", "column-footer")


def _collect_facet_axes(marks: list[dict]) -> list[dict]:
    """Gather every axis dict living inside a faceted spec's header/footer/cell
    role groups, for `_merge_axes` to fold into one entry per (scale, orient)
    -- mirrors how a single-view spec's top-level `axes` list works, since a
    faceted spec never has one (titles/grid live inside these role groups
    instead)."""
    collected: list[dict] = []
    for m in marks:
        if m.get("type") != "group":
            continue
        if m.get("role") in _FACET_HEADER_ROLES or _is_cell_group(m):
            collected.extend(m.get("axes", []))
    return collected


@dataclass
class IndependentScale:
    """One independently-resolved (`resolve: {scale: {<axis>: "independent"}}`)
    facet axis. `local_name` is the scale name the cell's own marks/axes
    reference (Vega-Lite always names these "child_x"/"child_y", nested
    inside the `cell` group's own `scales` list rather than the compiled
    spec's top-level `scales` -- an independent axis has NO top-level scale
    at all). `scale_spec` is that nested scale's raw Vega dict (type,
    zero/nice flags, ...); `field` is the data field its marks read off (so
    a panel's domain can be computed straight from that panel's own rows --
    see `_scales.build_panel_scale` for why: vegafusion never inlines a
    per-facet-value domain-summary dataset, only Vega's own per-cell group
    cloning would evaluate one, which happens after our pipeline runs)."""

    local_name: str
    scale_spec: dict
    field: str


@dataclass
class FacetInfo:
    """Structure recovered from a faceted spec's `cell` group + `layout` block.

    `kind` is one of "row", "column", "grid" (row x column), or "wrap"
    (`.facet(field, columns=N)`). `row_values`/`col_values` are the ordered
    (already-sorted, matching Vega's own groupby order) header label values
    for that dimension; `wrap_columns` is only set for kind "wrap".
    `cell_marks` is the facet `cell` group's own inner mark list (the real
    per-row mark defs, e.g. a `symbol`/`rect` mark reading `from.data ==
    "facet"`) -- rendering plugs a per-panel-filtered "facet" dataset entry
    in and draws these directly, one panel at a time. `independent` holds an
    `IndependentScale` per axis ("x" and/or "y") that isn't shared across
    panels; an axis absent from this dict uses the ordinary shared top-level
    scale, same as before independent resolution existed.
    """

    dataset: str
    kind: str
    row_field: str | None
    col_field: str | None
    row_values: list
    col_values: list
    wrap_columns: int | None
    cell_marks: list[dict]
    independent: dict[str, IndependentScale] = field(default_factory=dict)


def _facet_domain_values(datasets: dict[str, list[dict]], dataset_name: str, field_name: str) -> list:
    rows = datasets.get(dataset_name)
    if not rows:
        return []
    return [r[field_name] for r in rows if field_name in r]


def _field_for_scale(marks: list[dict], scale_name: str) -> str | None:
    """The data field a mark list's `encode.update` channel scale-qualified
    with `scale_name` reads -- e.g. for `{"y": {"field": "val", "scale":
    "child_y"}}`, `_field_for_scale(marks, "child_y") == "val"`. Used to
    recover which field an independently-resolved facet axis aggregates,
    since the (per-facet-value) domain vegafusion would otherwise hand us
    isn't inlined -- see `IndependentScale`."""
    for m in marks:
        for entry in m.get("encode", {}).get("update", {}).values():
            if isinstance(entry, dict) and entry.get("scale") == scale_name and "field" in entry:
                return entry["field"]
    return None


def _parse_independent_scales(cell: dict) -> dict[str, "IndependentScale"]:
    """Recover `cell`'s own nested (independently-resolved) scales, keyed by
    axis ("x"/"y"). Vega-Lite names these "child_x"/"child_y" -- the same
    naming convention as the `child_width`/`child_height` size signals."""
    cell_scales = {s["name"]: s for s in cell.get("scales", [])}
    cell_marks = cell.get("marks", [])
    independent: dict[str, IndependentScale] = {}
    for axis, local_name in (("x", "child_x"), ("y", "child_y")):
        if local_name not in cell_scales:
            continue
        scale_field = _field_for_scale(cell_marks, local_name)
        if scale_field is None:
            continue
        independent[axis] = IndependentScale(
            local_name=local_name, scale_spec=cell_scales[local_name], field=scale_field,
        )
    return independent


def _parse_facet(spec: dict, datasets: dict[str, list[dict]]) -> "FacetInfo | None":
    """Recover facet structure from a compiled+pre-transformed Vega spec, or
    None for a non-faceted (or concat) spec.

    Detection: a faceted spec's top-level `marks` contains exactly one `cell`
    style group with `from.facet`; that group's inner marks are the real
    per-panel marks. This does NOT collide with the `pathgroup` grouping a
    plain (non-faceted) line/area chart with a color channel uses -- that
    group also has `from.facet` but is never tagged `style: "cell"`.

    `kind` (row/column/grid/wrap) is read off the *names* of the facet's
    title role groups, which are stable across compiler versions:
    "row-title" (row facet), "column-title" (column facet; a *wrapped* facet
    reuses the same role but names the mark "facet-title" instead, so name
    -- not role -- disambiguates it), or both (row x column grid facet).
    """
    marks = spec.get("marks", [])
    cell = next((m for m in marks if _is_cell_group(m)), None)
    if cell is None:
        return None
    facet = cell.get("from", {}).get("facet")
    if not facet:
        return None
    groupby = facet.get("groupby", [])
    if not groupby:
        return None

    names = {m.get("name") for m in marks if m.get("type") == "group"}
    if "facet-title" in names:
        kind = "wrap"
    elif "row-title" in names and "column-title" in names:
        kind = "grid"
    elif "row-title" in names:
        kind = "row"
    elif "column-title" in names:
        kind = "column"
    else:
        # Unrecognized title-group naming (future compiler version?) --
        # best-effort guess from the groupby field count.
        kind = "grid" if len(groupby) == 2 else "wrap"

    row_field = col_field = None
    row_values: list = []
    col_values: list = []
    wrap_columns: int | None = None

    if kind == "grid":
        row_field, col_field = groupby[0], groupby[1]
        row_values = _facet_domain_values(datasets, "row_domain", row_field)
        col_values = _facet_domain_values(datasets, "column_domain", col_field)
    elif kind == "row":
        row_field = groupby[0]
        row_values = _facet_domain_values(datasets, "row_domain", row_field)
    elif kind == "column":
        col_field = groupby[0]
        col_values = _facet_domain_values(datasets, "column_domain", col_field)
    else:  # wrap
        col_field = groupby[0]
        col_values = _facet_domain_values(datasets, "facet_domain", col_field)
        cols = spec.get("layout", {}).get("columns")
        wrap_columns = int(cols) if isinstance(cols, (int, float)) else None

    # Robust fallback (unexpected/missing domain-summary dataset): derive
    # sorted-unique values straight from the main facet dataset.
    facet_rows = datasets.get(facet["data"], [])
    if kind in ("grid", "row") and not row_values:
        row_values = sorted({r[row_field] for r in facet_rows if row_field in r})
    if kind in ("grid", "column", "wrap") and not col_values:
        col_values = sorted({r[col_field] for r in facet_rows if col_field in r})
    if kind == "wrap" and not wrap_columns:
        wrap_columns = max(1, math.ceil(math.sqrt(len(col_values))))

    return FacetInfo(
        dataset=facet["data"], kind=kind,
        row_field=row_field, col_field=col_field,
        row_values=row_values, col_values=col_values,
        wrap_columns=wrap_columns, cell_marks=cell.get("marks", []),
        independent=_parse_independent_scales(cell),
    )


@dataclass
class ConcatLeaf:
    """A single-view child of a concat/hconcat/vconcat/repeat layout: its own
    marks, merged axes, x/y scale *names* (Vega-Lite gives every concat
    child its own namespaced position scales by default, e.g. "concat_0_x"
    -- these are ordinary entries in the compiled spec's top-level `scales`,
    just like a single-view chart's "x"/"y", only name-prefixed), and the
    signal name its own width/height is driven by (a literal in
    `cspec.signals`, or -- for a band-scale-driven child -- absent, falling
    back to the same band-derived-size math a single-view chart's `x`/`y`
    already uses, just parameterized on this child's own scale names)."""

    marks: list[dict]
    axes: list[dict]
    x_scale: str | None
    y_scale: str | None
    width_signal: str | None
    height_signal: str | None


@dataclass
class ConcatUnsupported:
    """A concat/repeat child we recognized but don't render -- currently
    just a faceted child (facet-inside-concat nesting; concat-inside-facet
    isn't a thing Vega-Lite produces). `reason` is shown as a warning and as
    placeholder text on that child's Axes."""

    reason: str


@dataclass
class ConcatInfo:
    """Structure recovered from a concat/hconcat/vconcat/repeat spec (or
    nested concat sub-tree): an ordered list of children, each a
    `ConcatLeaf` (single view), nested `ConcatInfo` (concat-of-concat, e.g.
    `vconcat` containing an `hconcat`), or `ConcatUnsupported`.

    `columns`, mirroring `FacetInfo.wrap_columns`, is read straight off this
    node's own `layout.columns`: absent -> hconcat shape (all children in
    one row), `1` -> vconcat shape (all children in one column), `N > 1` ->
    a general `alt.concat(..., columns=N)` wrap grid. `_concat_grid_shape`
    in `__init__.py` turns this into an actual (nrows, ncols).
    """

    children: list["ConcatLeaf | ConcatInfo | ConcatUnsupported"]
    columns: int | None


def _is_concat_node(m: dict) -> bool:
    """True for a `group` mark that's a concat/repeat child or sub-tree
    (never a facet's own header/footer/title/cell groups, and never a plain
    single-view chart's `pathgroup` line/area grouping) -- anything
    `type: "group"` with no `from.facet`. A facet's `cell` group is the one
    `type: "group"` mark that DOES have `from.facet`, so it's excluded here;
    everything else inside a facet's top-level marks (row_header,
    column_header, ...) also has no `from.facet`, but `CompiledSpec.from_vega`
    only calls `_parse_concat` after `_parse_facet` has already claimed the
    spec, so those never reach this function on a genuine facet spec."""
    return m.get("type") == "group" and "facet" not in m.get("from", {})


_ORIENT_TO_XY = {"bottom": "x", "top": "x", "left": "y", "right": "y"}


def _concat_leaf_scale_names(cell: dict) -> tuple[str | None, str | None]:
    """A concat leaf's own x/y scale names, read off its `axes` list's
    orientation (bottom/top -> x, left/right -> y) rather than assumed from
    a naming convention -- concat leaf scale names vary (`concat_0_x` for
    `hconcat`/`vconcat`, `child__column_val_x` for `repeat`)."""
    x_scale = y_scale = None
    for ax in cell.get("axes", []):
        which = _ORIENT_TO_XY.get(ax.get("orient"))
        if which == "x" and x_scale is None:
            x_scale = ax.get("scale")
        elif which == "y" and y_scale is None:
            y_scale = ax.get("scale")
    return x_scale, y_scale


def _parse_concat_leaf(cell: dict) -> ConcatLeaf:
    update = cell.get("encode", {}).get("update", {})
    width_sig = (update.get("width") or {}).get("signal")
    height_sig = (update.get("height") or {}).get("signal")
    x_scale, y_scale = _concat_leaf_scale_names(cell)
    return ConcatLeaf(
        marks=cell.get("marks", []), axes=_merge_axes(cell.get("axes", [])),
        x_scale=x_scale, y_scale=y_scale,
        width_signal=width_sig, height_signal=height_sig,
    )


def _parse_concat(node: dict) -> "ConcatInfo | None":
    """Recover concat structure from `node` (the top-level pre-transformed
    spec, or a nested concat-shaped `group` mark found inside another
    `ConcatInfo`'s children) -- None if `node`'s marks aren't all concat-
    shaped (i.e. `node` isn't actually a concat/repeat spec)."""
    marks = node.get("marks", [])
    if not marks or not all(m.get("type") == "group" for m in marks):
        return None

    children: list["ConcatLeaf | ConcatInfo | ConcatUnsupported"] = []
    for m in marks:
        if _is_cell_group(m) and m.get("from", {}).get("facet"):
            children.append(ConcatUnsupported(
                reason="a faceted chart inside a concat/repeat layout is not yet supported"
            ))
        elif _is_cell_group(m):
            children.append(_parse_concat_leaf(m))
        elif _is_concat_node(m):
            nested = _parse_concat(m)
            if nested is None:
                return None  # not actually concat-shaped after all
            children.append(nested)
        else:
            return None

    columns = node.get("layout", {}).get("columns")
    ncols = int(columns) if isinstance(columns, (int, float)) else None
    return ConcatInfo(children=children, columns=ncols)


@dataclass
class CompiledSpec:
    width: float | None
    height: float | None
    datasets: dict[str, list[dict]] = field(default_factory=dict)
    scales: dict[str, dict] = field(default_factory=dict)
    marks: list[dict] = field(default_factory=list)
    axes: list[dict] = field(default_factory=list)
    legends: list[dict] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    facet: "FacetInfo | None" = None
    concat: "ConcatInfo | None" = None

    @classmethod
    def from_vega(cls, spec: dict) -> "CompiledSpec":
        datasets = {
            d["name"]: d["values"]
            for d in spec.get("data", [])
            if "values" in d
        }

        scales = {s["name"]: s for s in spec.get("scales", [])}

        signals = {
            s["name"]: s["value"]
            for s in spec.get("signals", [])
            if "value" in s
        }

        width, height = get_plot_dims(spec)
        # A faceted spec has no top-level width/height; the per-panel size
        # lives in `child_width`/`child_height` signals instead. Only the
        # literal (non band-scale-driven) case is a plain signal `value`
        # here -- the band-scale-driven case has no literal and falls
        # through to `_layout`'s existing band-derived-size fallback, driven
        # off the same top-level x/y scales the facet's panels all share.
        if width is None and isinstance(signals.get("child_width"), (int, float)):
            width = signals["child_width"]
        if height is None and isinstance(signals.get("child_height"), (int, float)):
            height = signals["child_height"]

        facet_info = _parse_facet(spec, datasets)
        # `_parse_concat` is only tried once a genuine facet has been ruled
        # out -- see `_is_concat_node`'s docstring for why this ordering
        # matters (a facet's own header/footer/title groups would otherwise
        # look concat-shaped).
        concat_info = _parse_concat(spec) if facet_info is None else None
        raw_axes = list(spec.get("axes", []) or [])
        if facet_info is not None:
            raw_axes += _collect_facet_axes(spec.get("marks", []))

        return cls(
            width=width,
            height=height,
            datasets=datasets,
            scales=scales,
            marks=spec.get("marks", []),
            axes=_merge_axes(raw_axes),
            legends=spec.get("legends", []),
            signals=signals,
            facet=facet_info,
            concat=concat_info,
        )


def compile_chart(vl_spec: dict) -> tuple[CompiledSpec, list]:
    """Run the full VL -> Vega -> pre-transform -> CompiledSpec pipeline."""
    vg_spec = vl_to_vega(vl_spec)
    pre_spec, warnings = inline_transforms(vg_spec)
    return CompiledSpec.from_vega(pre_spec), warnings
