"""Flatten Vega mark trees and draw them onto a matplotlib Axes."""
from __future__ import annotations

import itertools
import warnings
from typing import Any, Iterator, NamedTuple

import matplotlib as mpl
import numpy as np
import pandas as pd


# Vega-Lite's compiled-in default mark color/point-stroke -- a literal hex,
# not a scale-resolved value, so it never passes through an MplScale. Single
# choke point: every literal color pulled off an encode entry funnels through
# `_resolve_color_literal` so mpl styles can restyle single-color marks too
# (declared heuristic, see plan M5 #2).
_VL_DEFAULT_BLUE = "#4c78a8"


def _resolve_color_literal(value):
    """Map VL's literal default blue to the mpl prop-cycle's first color.

    Any other literal value (explicit user color, 'transparent', etc.) passes
    through unchanged; None passes through unchanged.
    """
    if isinstance(value, str) and value.lower() == _VL_DEFAULT_BLUE:
        return "C0"
    return value


class DrawableMark(NamedTuple):
    mark: dict
    facet: dict | None  # {"dataset": name, "groupby": [fields]} or None


def walk_drawable_marks(marks: list[dict]) -> Iterator[DrawableMark]:
    """Flatten a Vega marks array, descending into facet groups (e.g. pathgroup).

    A facet group is a `group` mark with `from.facet.{data, name, groupby}`;
    its inner marks read `from.data == facet.name`. We yield those inner marks
    with the source dataset + groupby fields attached. Plain marks pass through.
    """
    for m in marks:
        if m.get("type") == "group" and "facet" in m.get("from", {}):
            facet = m["from"]["facet"]
            info = {"dataset": facet["data"], "groupby": facet.get("groupby", [])}
            for inner in m.get("marks", []):
                yield DrawableMark(inner, info)
        else:
            yield DrawableMark(m, None)


def resolve_channel(entry: dict, row: dict, scales: dict, paired_scale=None):
    """Resolve one Vega encode channel entry to a data-space value for a single row.

    A nested `offset` sub-entry (band-group offsets, histogram px-nudges) is
    ignored here -- band-x/band-y bar drawing resolves field-valued offsets
    itself (see `_resolve_position_with_offset`); signal-valued offsets are
    cosmetic pixel nudges from Vega's rendering path that we don't reproduce
    (not pixel-fidelity chasing, by design -- no warning, this is expected).
    """
    if entry is None:
        return None
    if "signal" in entry:
        warnings.warn(f"unsupported signal-valued channel {entry!r}; skipping")
        return None
    if "field" in entry:
        field = entry["field"]
        value = row.get(field)
        scale_name = entry.get("scale", paired_scale)
        if scale_name is not None and scale_name in scales:
            return scales[scale_name].to_data(value)
        return value
    if "value" in entry:
        # A literal value. If also scale-qualified it's a scaled literal (e.g. a
        # fixed data-space y2 of 0) -- data-space passthrough either way.
        return entry["value"]
    return None


def _resolve_position_with_offset(entry: dict, row: dict, scales: dict):
    """Resolve an x/y encode entry, honoring a field-valued `offset` sub-entry.

    `offset: {field, scale}` is how Vega encodes grouped-bar xOffset bands:
    combine the outer scale's cat_index with the offset scale's band_offset.
    """
    if entry is None:
        return None
    base_entry = {k: v for k, v in entry.items() if k != "offset"}
    base = resolve_channel(base_entry, row, scales)
    offset_entry = entry.get("offset")
    if not offset_entry or "field" not in offset_entry:
        return base
    outer_scale = scales.get(entry.get("scale"))
    offset_scale = scales.get(offset_entry.get("scale"))
    if outer_scale is None or offset_scale is None:
        return base
    delta = offset_scale.band_offset(row.get(offset_entry["field"]), outer_scale.band_frac)
    return base + delta


def _px_to_pt_area(px_area: float, dpi: float = 100) -> float:
    """Vega symbol `size` is the true area (px^2) of a circular glyph, d3-style
    (radius = sqrt(size/pi), so diameter = 2*sqrt(size/pi)). mpl scatter `s` is
    the SQUARE of the marker diameter in points, not the circle's area -- so
    the conversion needs the extra 4/pi on top of the px->pt unit conversion:
      s_pt2 = (4/pi) * size_px2 * (72/dpi)^2
    """
    import math

    return (4 / math.pi) * px_area * (72 / dpi) ** 2


def _px_to_pt_linear(px: float, dpi: float = 100) -> float:
    """Convert a linear px extent (e.g. stroke width) to points."""
    return px * 72 / dpi


def draw_symbol(ax, mark: dict, cspec, scales: dict, registry: dict, dpi: float = 100.0) -> None:
    update = mark.get("encode", {}).get("update", {})
    dataset_name = mark.get("from", {}).get("data")
    rows = cspec.datasets.get(dataset_name, [])
    if not rows:
        return

    x_entry, y_entry = update.get("x"), update.get("y")
    fill_entry = update.get("fill")
    stroke_entry = update.get("stroke")
    size_entry = update.get("size")

    fill_scale = scales.get(fill_entry.get("scale")) if fill_entry and "scale" in fill_entry else None
    stroke_scale = scales.get(stroke_entry.get("scale")) if stroke_entry and "scale" in stroke_entry else None
    size_scale = scales.get(size_entry.get("scale")) if size_entry and "scale" in size_entry else None

    fill_literal = fill_entry.get("value") if fill_entry and "value" in fill_entry and "field" not in fill_entry else None
    if fill_literal == "transparent":
        fill_literal = "none"
    fill_literal = _resolve_color_literal(fill_literal)
    stroke_literal = stroke_entry.get("value") if stroke_entry and "value" in stroke_entry and "field" not in stroke_entry else None
    stroke_literal = _resolve_color_literal(stroke_literal)

    def sizes_for(grows: list[dict]):
        if size_scale is not None:
            field = size_entry["field"]
            return [_px_to_pt_area(size_scale.size_for(r.get(field)), dpi) for r in grows]
        if size_entry and "value" in size_entry:
            return _px_to_pt_area(size_entry["value"], dpi)
        return _px_to_pt_area(30, dpi)  # mpl/vega default point area

    # VL's default point stroke is 2px; when a point is stroke-only (no fill,
    # or fill+stroke both present) the ring thickness should match that, not
    # mpl's own default linewidth.
    default_edge_lw = _px_to_pt_linear(2, dpi)

    color_scale = fill_scale or stroke_scale
    color_is_fill = fill_scale is not None
    color_field = (fill_entry if color_is_fill else stroke_entry)["field"] if color_scale is not None else None

    if color_scale is not None and color_scale.vtype in ("ordinal", "nominal", "band", "point"):
        # Categorical: one scatter call per category (own handle -> legend order == domain order).
        df = pd.DataFrame(rows)
        cats = color_scale.categories or list(dict.fromkeys(df[color_field]))
        for cat in cats:
            sub = df[df[color_field] == cat]
            if sub.empty:
                continue
            grows = sub.to_dict("records")
            xs = [resolve_channel(x_entry, r, scales) for r in grows]
            ys = [resolve_channel(y_entry, r, scales) for r in grows]
            s = sizes_for(grows)
            color = color_scale.color_for(cat)
            kwargs: dict[str, Any] = dict(s=s, label=str(cat))
            if color_is_fill:
                kwargs["c"] = color
                if stroke_literal is not None:
                    kwargs["edgecolors"] = stroke_literal
                    kwargs["linewidths"] = default_edge_lw
            else:
                kwargs["edgecolors"] = color
                kwargs["c"] = fill_literal if fill_literal is not None else "none"
                kwargs["linewidths"] = default_edge_lw
            handle = ax.scatter(xs, ys, **kwargs)
            registry.setdefault(color_scale.name, []).append((handle, str(cat)))
    elif color_scale is not None:
        # Continuous color: single scatter call, per-point cmap(norm(v)) colors.
        cmap, norm = color_scale.color_cmap_norm()
        xs = [resolve_channel(x_entry, r, scales) for r in rows]
        ys = [resolve_channel(y_entry, r, scales) for r in rows]
        vals = [r.get(color_field) for r in rows]
        s = sizes_for(rows)
        if color_is_fill:
            ax.scatter(xs, ys, s=s, c=vals, cmap=cmap, norm=norm)
        else:
            edge_colors = [cmap(norm(v)) for v in vals]
            ax.scatter(xs, ys, s=s, edgecolors=edge_colors, linewidths=default_edge_lw,
                       c=fill_literal if fill_literal is not None else "none")
        # No per-point legend handles for a continuous scale; _guides builds a
        # colorbar directly from the scale (looked up by name), not the registry.
    else:
        xs = [resolve_channel(x_entry, r, scales) for r in rows]
        ys = [resolve_channel(y_entry, r, scales) for r in rows]
        s = sizes_for(rows)
        lw = default_edge_lw if stroke_literal is not None else None
        ax.scatter(xs, ys, s=s, c=fill_literal, edgecolors=stroke_literal, linewidths=lw)


def draw_line(ax, mark: dict, facet: dict | None, cspec, scales: dict, registry: dict) -> None:
    update = mark.get("encode", {}).get("update", {})
    x_entry, y_entry = update.get("x"), update.get("y")
    stroke_entry = update.get("stroke")
    sort_field = mark.get("sort", {}).get("field")  # "x" means: sort by the x channel's field

    if facet is not None:
        rows = cspec.datasets.get(facet["dataset"], [])
        groupby = facet["groupby"]
        df = pd.DataFrame(rows)
        groups = list(df.groupby(groupby, sort=False))
    else:
        rows = cspec.datasets.get(mark.get("from", {}).get("data"), [])
        groups = [(None, pd.DataFrame(rows))]

    stroke_scale = scales.get(stroke_entry.get("scale")) if stroke_entry and "scale" in stroke_entry else None
    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    color_cycle = itertools.cycle(colors)

    for group_key, gdf in groups:
        group_rows = gdf.to_dict("records")
        xs = [resolve_channel(x_entry, r, scales) for r in group_rows]
        ys = [resolve_channel(y_entry, r, scales) for r in group_rows]

        if sort_field == "x":
            order = np.argsort(pd.Series(xs, dtype="object").astype(str)) if any(isinstance(v, str) for v in xs) else np.argsort(xs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]

        keep = [not (pd.isna(x) or pd.isna(y)) for x, y in zip(xs, ys)]
        xs = [x for x, k in zip(xs, keep) if k]
        ys = [y for y, k in zip(ys, keep) if k]

        label = group_key if isinstance(group_key, str) else (group_key[0] if group_key else None)

        kwargs: dict[str, Any] = {}
        if stroke_scale is not None and label is not None:
            # color_for() indexes by domain position, not groupby iteration order,
            # so line colors always match legend/domain order.
            kwargs["color"] = stroke_scale.color_for(label)
            kwargs["label"] = label
        elif stroke_scale is not None:
            kwargs["color"] = next(color_cycle)
        elif stroke_entry and "value" in stroke_entry:
            kwargs["color"] = _resolve_color_literal(stroke_entry["value"])

        (handle,) = ax.plot(xs, ys, **kwargs)
        if stroke_scale is not None and label is not None:
            registry.setdefault(stroke_scale.name, []).append((handle, label))


def draw_area(ax, mark: dict, facet: dict | None, cspec, scales: dict, registry: dict) -> None:
    update = mark.get("encode", {}).get("update", {})
    x_entry, y_entry, y2_entry = update.get("x"), update.get("y"), update.get("y2")
    fill_entry = update.get("fill")

    if facet is not None:
        rows = cspec.datasets.get(facet["dataset"], [])
        groupby = facet["groupby"]
        df = pd.DataFrame(rows)
        groups = list(df.groupby(groupby, sort=False))
    else:
        rows = cspec.datasets.get(mark.get("from", {}).get("data"), [])
        groups = [(None, pd.DataFrame(rows))]

    fill_scale = scales.get(fill_entry.get("scale")) if fill_entry and "scale" in fill_entry else None
    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    color_cycle = itertools.cycle(colors)

    for group_key, gdf in groups:
        group_rows = gdf.to_dict("records")
        xs = [resolve_channel(x_entry, r, scales) for r in group_rows]
        ys = [resolve_channel(y_entry, r, scales) for r in group_rows]
        y2s = [resolve_channel(y2_entry, r, scales) for r in group_rows] if y2_entry else [0] * len(group_rows)

        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        y2s = [y2s[i] for i in order]

        label = group_key if isinstance(group_key, str) else (group_key[0] if group_key else None)
        if fill_scale is not None and label is not None:
            color = fill_scale.color_for(label)
        elif fill_entry and "value" in fill_entry:
            color = _resolve_color_literal(fill_entry["value"])
        else:
            color = next(color_cycle)

        handle = ax.fill_between(xs, y2s, ys, color=color, label=label)
        if fill_scale is not None and label is not None:
            registry.setdefault(fill_scale.name, []).append((handle, label))


def _entry_is_real(entry: dict | None) -> bool:
    """True if `entry` names an actual data field/literal, not a group-span signal ref.

    A y-only (or x-only) rule encode still carries an `x` entry pointing at
    the enclosing group's width/height (`{field: {signal: null, group:
    "width", ...}}`) to mean "span the full axis"; that's not a real channel.
    """
    if entry is None:
        return False
    if "field" in entry:
        return isinstance(entry["field"], str)
    if "value" in entry:
        return True
    return False


def draw_rule(ax, mark: dict, cspec, scales: dict, registry: dict) -> None:
    update = mark.get("encode", {}).get("update", {})
    x_entry, x2_entry = update.get("x"), update.get("x2")
    y_entry, y2_entry = update.get("y"), update.get("y2")
    rows = cspec.datasets.get(mark.get("from", {}).get("data"), [])
    stroke_entry = update.get("stroke")
    color = stroke_entry.get("value") if stroke_entry and "value" in stroke_entry else None
    color = _resolve_color_literal(color)

    x_real, y_real = _entry_is_real(x_entry), _entry_is_real(y_entry)
    x2_real, y2_real = _entry_is_real(x2_entry), _entry_is_real(y2_entry)

    if x_real and x2_real and y_real and y2_real:
        from matplotlib.collections import LineCollection

        segments = []
        for r in rows:
            x1 = resolve_channel(x_entry, r, scales)
            x2 = resolve_channel(x2_entry, r, scales)
            y1 = resolve_channel(y_entry, r, scales)
            y2 = resolve_channel(y2_entry, r, scales)
            segments.append([(x1, y1), (x2, y2)])
        ax.add_collection(LineCollection(segments, colors=color or "C0"))
        ax.autoscale_view()
    elif y_real and not x_real:
        for r in rows:
            ax.axhline(resolve_channel(y_entry, r, scales), color=color)
    elif x_real and not y_real:
        for r in rows:
            ax.axvline(resolve_channel(x_entry, r, scales), color=color)
    else:
        # Both x and y are real single points without x2/y2 -- best-effort
        # fallback: a full-height vertical line through x (not pixel-exact).
        for r in rows:
            ax.axvline(resolve_channel(x_entry, r, scales), color=color)


def _detect_tick(update: dict) -> bool:
    """A rect mark is a tick when its width or height is a small fixed literal px value."""
    for key in ("width", "height"):
        entry = update.get(key)
        if entry and "value" in entry and "field" not in entry and "signal" not in entry:
            if entry["value"] <= 6:
                return True
    return False


def _draw_tick(ax, rows: list[dict], update: dict, scales: dict, x_scale, y_scale, dpi: float = 100.0) -> None:
    """Thin-extent rect (mark_tick): draw as short line segments.

    Orientation comes from the encode shape, not a fixed axis guess: an `xc`
    entry paired with a thin literal `width` means the value axis is x and
    the tick is a VERTICAL stroke at x=xc spanning the category band on y
    (mirror: `yc` + thin `height` -> horizontal stroke spanning the x band).
    The band's half-length uses that band scale's `band_frac` (not a fixed
    constant), so tick length matches the actual bandwidth; the stroke's
    thickness is the encode's thin px value converted to pt.
    """
    from matplotlib.collections import LineCollection

    xc_entry = update.get("xc")
    yc_entry = update.get("yc")
    x_entry = update.get("x")
    y_entry = update.get("y")
    width_entry = update.get("width")
    height_entry = update.get("height")

    def _is_thin(entry):
        return bool(entry) and "value" in entry and "field" not in entry and entry["value"] <= 6

    segments = []
    if xc_entry is not None and _is_thin(width_entry):
        # Value axis is x; category axis is y -- vertical strokes.
        cat_entry = y_entry or yc_entry
        band_frac = y_scale.band_frac if y_scale is not None else 0.7
        half = band_frac / 2
        for r in rows:
            xv = resolve_channel(xc_entry, r, scales)
            yv = resolve_channel(cat_entry, r, scales)
            segments.append([(xv, yv - half), (xv, yv + half)])
        thickness_px = width_entry["value"]
    elif yc_entry is not None and _is_thin(height_entry):
        # Value axis is y; category axis is x -- horizontal strokes.
        cat_entry = x_entry or xc_entry
        band_frac = x_scale.band_frac if x_scale is not None else 0.7
        half = band_frac / 2
        for r in rows:
            yv = resolve_channel(yc_entry, r, scales)
            xv = resolve_channel(cat_entry, r, scales)
            segments.append([(xv - half, yv), (xv + half, yv)])
        thickness_px = height_entry["value"]
    else:
        # Fallback: neither shape matched (unexpected encode) -- best-effort
        # single-point segments so drawing doesn't crash.
        x_entry_fb = x_entry or xc_entry
        y_entry_fb = y_entry or yc_entry
        for r in rows:
            xv = resolve_channel(x_entry_fb, r, scales) if x_entry_fb else None
            yv = resolve_channel(y_entry_fb, r, scales) if y_entry_fb else None
            segments.append([(xv, yv), (xv, yv)])
        thickness_px = 1

    fill_entry = update.get("fill") or update.get("stroke")
    color = fill_entry.get("value") if fill_entry and "value" in fill_entry else None
    color = _resolve_color_literal(color)
    linewidth = _px_to_pt_linear(thickness_px, dpi)
    ax.add_collection(LineCollection(segments, colors=color or "C0", linewidths=linewidth))
    ax.autoscale_view()


def _draw_bar_band_x(ax, rows: list[dict], update: dict, scales: dict, x_scale, fill_field, fill_scale, registry: dict) -> None:
    """Vertical bar (x band/point, y/y2 linear); handles stacked (fill split) and
    grouped (xOffset) variants -- y/y2 are already pre-stacked by vegafusion."""
    x_entry = update["x"]
    y_entry, y2_entry = update.get("y"), update.get("y2")
    offset_entry = x_entry.get("offset")

    if offset_entry and "field" in offset_entry:
        offset_scale = scales.get(offset_entry["scale"])
        width = offset_scale.sub_band_width(x_scale.band_frac) if offset_scale else x_scale.band_frac
    else:
        width = x_scale.band_frac

    def xpos(row):
        return _resolve_position_with_offset(x_entry, row, scales)

    if fill_field is not None:
        df = pd.DataFrame(rows)
        groups = df.groupby(fill_field, sort=False)
        cats = fill_scale.categories if fill_scale and fill_scale.categories else list(dict.fromkeys(df[fill_field]))
        for cat in cats:
            if cat not in groups.groups:
                continue
            grows = groups.get_group(cat).to_dict("records")
            xs = [xpos(r) for r in grows]
            ys = [resolve_channel(y_entry, r, scales) for r in grows]
            y2s = [resolve_channel(y2_entry, r, scales) for r in grows] if y2_entry else [0] * len(grows)
            heights = [y - y2 for y, y2 in zip(ys, y2s)]
            color = fill_scale.color_for(cat) if fill_scale else None
            handle = ax.bar(xs, heights, bottom=y2s, width=width, color=color, label=str(cat))
            if fill_scale:
                registry.setdefault(fill_scale.name, []).append((handle, str(cat)))
    else:
        xs = [xpos(r) for r in rows]
        ys = [resolve_channel(y_entry, r, scales) for r in rows]
        y2s = [resolve_channel(y2_entry, r, scales) for r in rows] if y2_entry else [0] * len(rows)
        heights = [y - y2 for y, y2 in zip(ys, y2s)]
        fill_entry = update.get("fill")
        fill_val = fill_entry.get("value") if fill_entry and "value" in fill_entry else None
        fill_val = _resolve_color_literal(fill_val)
        ax.bar(xs, heights, bottom=y2s, width=width, color=fill_val)


def _draw_bar_band_y(ax, rows: list[dict], update: dict, scales: dict, y_scale, fill_field, fill_scale, registry: dict) -> None:
    """Horizontal bar (y band/point, x/x2 linear); mirrors `_draw_bar_band_x`."""
    y_entry = update["y"]
    x_entry, x2_entry = update.get("x"), update.get("x2")
    offset_entry = y_entry.get("offset")

    if offset_entry and "field" in offset_entry:
        offset_scale = scales.get(offset_entry["scale"])
        width = offset_scale.sub_band_width(y_scale.band_frac) if offset_scale else y_scale.band_frac
    else:
        width = y_scale.band_frac

    def ypos(row):
        return _resolve_position_with_offset(y_entry, row, scales)

    if fill_field is not None:
        df = pd.DataFrame(rows)
        groups = df.groupby(fill_field, sort=False)
        cats = fill_scale.categories if fill_scale and fill_scale.categories else list(dict.fromkeys(df[fill_field]))
        for cat in cats:
            if cat not in groups.groups:
                continue
            grows = groups.get_group(cat).to_dict("records")
            ys = [ypos(r) for r in grows]
            xs = [resolve_channel(x_entry, r, scales) for r in grows]
            x2s = [resolve_channel(x2_entry, r, scales) for r in grows] if x2_entry else [0] * len(grows)
            widths = [x - x2 for x, x2 in zip(xs, x2s)]
            color = fill_scale.color_for(cat) if fill_scale else None
            handle = ax.barh(ys, widths, left=x2s, height=width, color=color, label=str(cat))
            if fill_scale:
                registry.setdefault(fill_scale.name, []).append((handle, str(cat)))
    else:
        ys = [ypos(r) for r in rows]
        xs = [resolve_channel(x_entry, r, scales) for r in rows]
        x2s = [resolve_channel(x2_entry, r, scales) for r in rows] if x2_entry else [0] * len(rows)
        widths = [x - x2 for x, x2 in zip(xs, x2s)]
        fill_entry = update.get("fill")
        fill_val = fill_entry.get("value") if fill_entry and "value" in fill_entry else None
        fill_val = _resolve_color_literal(fill_val)
        ax.barh(ys, widths, left=x2s, height=width, color=fill_val)


def _draw_bar_linear(ax, rows: list[dict], update: dict, scales: dict) -> None:
    """Histogram / linear-x rect: x/x2 are scaled bin-edge fields, y2 a scaled-value 0."""
    x_entry, x2_entry = update.get("x"), update.get("x2")
    y_entry, y2_entry = update.get("y"), update.get("y2")

    xs = [_resolve_position_with_offset(x_entry, r, scales) for r in rows]
    x2s = [_resolve_position_with_offset(x2_entry, r, scales) for r in rows] if x2_entry else [0.0] * len(rows)
    ys = [resolve_channel(y_entry, r, scales) for r in rows]
    y2s = [resolve_channel(y2_entry, r, scales) for r in rows] if y2_entry else [0.0] * len(rows)

    lefts = [min(a, b) for a, b in zip(xs, x2s)]
    widths = [abs(b - a) for a, b in zip(xs, x2s)]
    heights = [y - y2 for y, y2 in zip(ys, y2s)]

    fill_entry = update.get("fill")
    fill_val = fill_entry.get("value") if fill_entry and "value" in fill_entry else None
    fill_val = _resolve_color_literal(fill_val)
    ax.bar(lefts, heights, width=widths, bottom=y2s, align="edge", color=fill_val)


def draw_rect(ax, mark: dict, cspec, scales: dict, registry: dict) -> None:
    update = mark.get("encode", {}).get("update", {})
    dataset_name = mark.get("from", {}).get("data")
    rows = cspec.datasets.get(dataset_name, [])
    if not rows:
        return

    # `xc`/`yc` (center-anchored, used by mark_tick) are treated as x/y here.
    x_entry = update.get("x") or update.get("xc")
    y_entry = update.get("y") or update.get("yc")
    x_scale = scales.get(x_entry.get("scale")) if x_entry and "scale" in x_entry else None
    y_scale = scales.get(y_entry.get("scale")) if y_entry and "scale" in y_entry else None
    x_is_band = x_scale is not None and x_scale.vtype in ("band", "point")
    y_is_band = y_scale is not None and y_scale.vtype in ("band", "point")

    if _detect_tick(update):
        _draw_tick(ax, rows, update, scales, x_scale, y_scale)
        return

    fill_entry = update.get("fill")
    fill_scale = scales.get(fill_entry.get("scale")) if fill_entry and "scale" in fill_entry else None
    fill_field = fill_entry.get("field") if fill_scale is not None else None

    if x_is_band and not y_is_band:
        _draw_bar_band_x(ax, rows, update, scales, x_scale, fill_field, fill_scale, registry)
    elif y_is_band and not x_is_band:
        _draw_bar_band_y(ax, rows, update, scales, y_scale, fill_field, fill_scale, registry)
    elif not x_is_band and not y_is_band:
        _draw_bar_linear(ax, rows, update, scales)
    else:
        warnings.warn("rect mark with both x and y band scales not supported; skipping")


def draw_marks(ax, cspec, scales: dict) -> dict:
    """Draw all drawable marks; returns a legend-handle registry {scale_name: [(handle, label)]}."""
    registry: dict[str, list] = {}
    for mark, facet in walk_drawable_marks(cspec.marks):
        mtype = mark.get("type")
        if mtype == "symbol":
            draw_symbol(ax, mark, cspec, scales, registry)
        elif mtype == "line":
            draw_line(ax, mark, facet, cspec, scales, registry)
        elif mtype == "area":
            draw_area(ax, mark, facet, cspec, scales, registry)
        elif mtype == "rect":
            draw_rect(ax, mark, cspec, scales, registry)
        elif mtype == "rule":
            draw_rule(ax, mark, cspec, scales, registry)
        else:
            warnings.warn(f"mark type {mtype!r} not yet supported; skipping")
    return registry
