"""Resolve Vega scale domains and apply them to matplotlib Axes.

Domain resolution handles the three verified patterns:
  1. data-ref  {data, field(s), sort}       -> scan the inlined dataset column(s)
  2. signal    (data("X")[0] || {}).min/max -> read the inlined min/max row dataset
  3. bin       {signal: "[NAME.start, NAME.stop]"} -> read the bins signal value
Literal list domains pass through unchanged. Anything else warns and returns None.
"""
from __future__ import annotations

import itertools
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib as mpl
import pandas as pd

_MINMAX_RE = re.compile(r'data\("([^"]+)"\)\[0\].*\.(min|max)$')
_BIN_RE = re.compile(r'^\[\s*([A-Za-z0-9_]+)\.start\s*,\s*\1\.stop\s*\]$')

# M4: named-range -> mpl colormap-name table for {"scheme": name} color scales.
_SCHEME_TABLE = {
    "tableau10": "tab10",
    "tableau20": "tab20",
    "category10": "tab10",
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "blues": "Blues",
    "greens": "Greens",
    "oranges": "Oranges",
    "reds": "Reds",
    "purples": "Purples",
    "greys": "Greys",
    "redblue": "RdBu",
    "blueorange": "coolwarm",  # approximate match (no exact d3 blueorange in mpl)
    "turbo": "turbo",
}


def coerce_temporal(values):
    """Coerce a sequence of ISO date strings (or datetimes) to pandas Timestamps."""
    return pd.to_datetime(pd.Series(list(values)))


def _resolve_signal_domain(domain: list, cspec) -> list | None:
    """Domain expressed as two {signal: ...} min/max refs into a domain dataset."""
    lo = hi = None
    for entry in domain:
        sig = entry.get("signal", "") if isinstance(entry, dict) else ""
        m = _MINMAX_RE.search(sig)
        if not m:
            return None
        dataset_name, which = m.group(1), m.group(2)
        rows = cspec.datasets.get(dataset_name)
        if not rows:
            return None
        val = rows[0].get(which)
        if which == "min":
            lo = val
        else:
            hi = val
    if lo is None or hi is None:
        return None
    return [lo, hi]


def _resolve_data_ref_domain(domain: dict, cspec, vtype: str):
    """Domain expressed as {data, field(s), sort}."""
    dataset_name = domain.get("data")
    rows = cspec.datasets.get(dataset_name)
    if rows is None:
        return None
    fields = domain.get("field", domain.get("fields"))
    fields = [fields] if isinstance(fields, str) else fields
    if vtype in ("linear", "log", "sqrt", "time", "utc"):
        vals = [row[f] for f in fields for row in rows if row.get(f) is not None]
        if not vals:
            return None
        return [min(vals), max(vals)]
    # ordinal/nominal/band/point (categorical) -> unique, sorted category list.
    cats: list = []
    seen = set()
    for row in rows:
        for f in fields:
            v = row.get(f)
            if v is not None and v not in seen:
                seen.add(v)
                cats.append(v)
    return sorted(cats)


def _resolve_multi_dataset_domain(domain: dict, cspec, vtype: str):
    """Domain expressed as {fields: [{data, field}, ...]} -- a union domain
    across multiple datasets, one per sibling layer sharing this scale
    (layered charts: each layer gets its own flattened dataset even when the
    underlying source data is identical)."""
    refs = domain.get("fields")
    if not isinstance(refs, list) or not refs or not all(isinstance(r, dict) for r in refs):
        return None
    if vtype in ("linear", "log", "sqrt", "time", "utc"):
        vals = []
        for ref in refs:
            rows = cspec.datasets.get(ref.get("data"))
            if not rows:
                continue
            f = ref.get("field")
            vals.extend(row[f] for row in rows if row.get(f) is not None)
        if not vals:
            return None
        return [min(vals), max(vals)]
    cats: list = []
    seen = set()
    for ref in refs:
        rows = cspec.datasets.get(ref.get("data")) or []
        f = ref.get("field")
        for row in rows:
            v = row.get(f)
            if v is not None and v not in seen:
                seen.add(v)
                cats.append(v)
    return sorted(cats)


def _resolve_bin_domain(domain: dict, cspec) -> list | None:
    sig = domain.get("signal", "")
    m = _BIN_RE.match(sig.strip())
    if not m:
        return None
    bins_signal_name = m.group(1)
    bins = cspec.signals.get(bins_signal_name)
    if not isinstance(bins, dict):
        return None
    return [bins.get("start"), bins.get("stop")]


def resolve_domain(scale_spec: dict, cspec) -> Any:
    """Resolve a Vega scale's `domain` to a concrete list, or None (warn) if unresolvable."""
    domain = scale_spec.get("domain")
    vtype = scale_spec.get("type", "linear")
    name = scale_spec.get("name", "?")

    if isinstance(domain, list):
        if domain and isinstance(domain[0], dict) and "signal" in domain[0]:
            resolved = _resolve_signal_domain(domain, cspec)
            if resolved is not None:
                return resolved
            warnings.warn(f"scale {name!r}: could not resolve signal domain {domain!r}")
            return None
        return domain  # literal list domain, pass through

    if isinstance(domain, dict):
        if "signal" in domain:
            resolved = _resolve_bin_domain(domain, cspec)
            if resolved is not None:
                return resolved
            warnings.warn(f"scale {name!r}: could not resolve signal domain {domain!r}")
            return None
        if "data" in domain:
            resolved = _resolve_data_ref_domain(domain, cspec, vtype)
            if resolved is not None:
                return resolved
            warnings.warn(f"scale {name!r}: could not resolve data-ref domain {domain!r}")
            return None
        if "fields" in domain:
            resolved = _resolve_multi_dataset_domain(domain, cspec, vtype)
            if resolved is not None:
                return resolved
            warnings.warn(f"scale {name!r}: could not resolve multi-dataset domain {domain!r}")
            return None

    warnings.warn(f"scale {name!r}: unrecognized domain shape {domain!r}")
    return None


def _nice(lo: float, hi: float) -> tuple[float, float]:
    """d3-style nice(): round [lo, hi] outward to a nice step (1/2/2.5/5/10 * 10^n)."""
    import math

    if lo == hi:
        return lo, hi
    span = hi - lo
    step_raw = span / 10
    magnitude = 10 ** math.floor(math.log10(step_raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * magnitude
        if step >= step_raw:
            break
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def _nice_log(lo: float, hi: float) -> tuple[float, float]:
    """Log-scale nice(): round outward to the nearest power of 10.

    d3/Vega's linear nice() (floor/ceil to a 1/2/2.5/5/10 step) can round a
    positive lower bound down to zero or negative -- invalid on a log axis.
    Log scales nice to powers of ten instead.
    """
    import math

    if lo <= 0 or hi <= 0:
        return lo, hi
    return 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))


def _apply_zero_nice(domain: list, scale_spec: dict) -> list:
    if not (isinstance(domain, list) and len(domain) == 2
            and isinstance(domain[0], (int, float)) and isinstance(domain[1], (int, float))):
        return domain
    lo, hi = domain
    is_log = scale_spec.get("type") == "log"
    if scale_spec.get("zero") and not is_log:
        # zero:true is meaningless (and never emitted by VL) for log scales.
        lo, hi = min(lo, 0), max(hi, 0)
    if scale_spec.get("nice"):
        lo, hi = _nice_log(lo, hi) if is_log else _nice(lo, hi)
    return [lo, hi]


def _band_paddings(vtype: str, scale_spec: dict) -> tuple[float, float]:
    """Resolve (paddingInner, paddingOuter) for a band/point scale.

    Point scales are, per d3, band scales with paddingInner pinned to 1 and a
    single `padding` value used as paddingOuter. Band scales default
    paddingInner to 0.1 (VL's bar default) and paddingOuter to half of that
    when unspecified -- matches the stacked-bar ground-truth fixture
    (paddingInner 0.1, paddingOuter 0.05).
    """
    if vtype == "point":
        paddingInner = 1.0
        paddingOuter = scale_spec.get("paddingOuter", scale_spec.get("padding", 0.5))
        return paddingInner, paddingOuter
    paddingInner = scale_spec.get("paddingInner", scale_spec.get("padding", 0.1))
    paddingOuter = scale_spec.get("paddingOuter", paddingInner / 2)
    return paddingInner, paddingOuter


def _scheme_to_cmap_name(scheme: str) -> str:
    name = _SCHEME_TABLE.get(scheme.lower())
    if name is None:
        warnings.warn(f"unknown color scheme {scheme!r}; falling back to tab10")
        name = "tab10"
    return name


def categorical_palette(scale_spec: dict, n: int) -> list:
    """Resolve a categorical color scale's range to a concrete list of >= n colors."""
    rng = scale_spec.get("range")
    if isinstance(rng, list):
        colors = rng
    elif isinstance(rng, dict) and "scheme" in rng:
        cmap = mpl.colormaps[_scheme_to_cmap_name(rng["scheme"])]
        if hasattr(cmap, "colors"):  # discrete/qualitative colormap (tab10, tab20, ...)
            colors = list(cmap.colors)
        else:  # continuous colormap used categorically -> sample n evenly
            colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    else:
        if rng not in (None, "category"):
            warnings.warn(f"unrecognized categorical color range {rng!r}; using prop_cycle")
        colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    if len(colors) < n:
        colors = list(itertools.islice(itertools.cycle(colors), n))
    return colors


@dataclass
class MplScale:
    name: str
    vtype: str
    domain: Any
    scale_spec: dict
    categories: list | None = None
    paddingInner: float = 0.0
    paddingOuter: float = 0.0
    _palette: list | None = field(default=None, repr=False, compare=False)
    _cat_map: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.categories:
            self._cat_map = {v: i for i, v in enumerate(self.categories)}

    # -- band/point ---------------------------------------------------
    @property
    def band_frac(self) -> float:
        """Fraction of a unit slot occupied by the bar/band itself (1 - paddingInner)."""
        return 1 - self.paddingInner

    def cat_index(self, value) -> int:
        if self._cat_map is None:
            raise ValueError(f"scale {self.name!r} has no resolved categories")
        return self._cat_map[value]

    def band_offset(self, value, band_frac_outer: float) -> float:
        """Within-slot offset (axis units) for a secondary band scale (e.g. xOffset).

        Divides the outer scale's usable band width evenly among this scale's
        k categories and centers each item's sub-band. Ignores this scale's
        own paddingInner for prototype simplicity (a small gap between grouped
        bars is not reproduced).
        """
        k = len(self.categories) if self.categories else 1
        j = self.cat_index(value)
        usable = band_frac_outer
        width = usable / k
        return -usable / 2 + (j + 0.5) * width

    def sub_band_width(self, band_frac_outer: float) -> float:
        k = len(self.categories) if self.categories else 1
        return band_frac_outer / k

    # -- color ---------------------------------------------------------
    def color_for(self, value):
        """Categorical color for `value`, indexed by domain position (not data order)."""
        if self._palette is None:
            n = len(self.categories) if self.categories else 1
            self._palette = categorical_palette(self.scale_spec, n)
        idx = self.cat_index(value)
        return self._palette[idx % len(self._palette)]

    def color_cmap_norm(self):
        """(cmap, norm) for a continuous color scale."""
        from matplotlib.colors import LogNorm, Normalize

        rng = self.scale_spec.get("range")
        if isinstance(rng, dict) and "scheme" in rng:
            cmap = mpl.colormaps[_scheme_to_cmap_name(rng["scheme"])]
        elif isinstance(rng, str) and rng in ("ramp", "heatmap"):
            cmap = mpl.colormaps[mpl.rcParams["image.cmap"]]
        else:
            cmap = mpl.colormaps[mpl.rcParams["image.cmap"]]
        lo, hi = self.domain[0], self.domain[1]
        norm_cls = LogNorm if self.vtype == "log" else Normalize
        return cmap, norm_cls(vmin=lo, vmax=hi)

    # -- size ------------------------------------------------------------
    def size_for(self, value) -> float:
        """Map a data value to a Vega symbol `size` (true area, px^2), via
        range interpolation. Caller converts px^2 -> mpl scatter `s` (pt^2)
        with `_px_to_pt_area` (single conversion choke point)."""
        lo, hi = self.domain[0], self.domain[1]
        rng = self.scale_spec.get("range", [4, 361])
        r0, r1 = rng[0], rng[1]
        if hi == lo or value is None:
            px_area = r1
        else:
            t = (value - lo) / (hi - lo)
            t = min(max(t, 0.0), 1.0)
            if self.vtype == "sqrt":
                v = (r0 ** 0.5 + t * (r1 ** 0.5 - r0 ** 0.5)) ** 2
            else:
                v = r0 + t * (r1 - r0)
            px_area = v
        return px_area

    # -- generic -----------------------------------------------------------
    def to_data(self, value):
        """Map a raw spec/data value into mpl data space."""
        if self.vtype in ("band", "point"):
            return self.cat_index(value)
        if self.vtype in ("time", "utc"):
            return coerce_temporal([value]).iloc[0]
        return value


def build_scales(cspec) -> dict[str, MplScale]:
    scales: dict[str, MplScale] = {}
    for name, spec in cspec.scales.items():
        vtype = spec.get("type", "linear")
        domain = resolve_domain(spec, cspec)
        if vtype in ("linear", "log", "sqrt", "time", "utc"):
            domain = _apply_zero_nice(domain, spec)
        if vtype in ("time", "utc") and isinstance(domain, list):
            domain = list(coerce_temporal(domain))

        categories = None
        paddingInner = paddingOuter = 0.0
        if vtype in ("band", "point"):
            categories = domain if isinstance(domain, list) else []
            paddingInner, paddingOuter = _band_paddings(vtype, spec)
        elif vtype in ("ordinal", "nominal"):
            # Categorical scales (color, shape, ...) resolve a domain the same
            # way band scales do; expose it as `categories` for color_for().
            categories = domain if isinstance(domain, list) else []

        scales[name] = MplScale(
            name=name, vtype=vtype, domain=domain, scale_spec=spec,
            categories=categories, paddingInner=paddingInner, paddingOuter=paddingOuter,
        )
    return scales


def apply_position_scale(ax, mpl_scale: MplScale, axis: str) -> None:
    """Apply a resolved position scale (x or y) to an Axes: limits, scale kind, date formatting."""
    set_lim = ax.set_xlim if axis == "x" else ax.set_ylim
    set_scale = ax.set_xscale if axis == "x" else ax.set_yscale

    if mpl_scale.vtype in ("band", "point"):
        import matplotlib.ticker as mticker

        categories = mpl_scale.categories or []
        n = len(categories)
        axis_obj = ax.xaxis if axis == "x" else ax.yaxis
        axis_obj.set_major_locator(mticker.FixedLocator(list(range(n))))
        axis_obj.set_major_formatter(mticker.FixedFormatter([str(c) for c in categories]))
        pad_extra = mpl_scale.paddingOuter
        set_lim(-0.5 - pad_extra, n - 0.5 + pad_extra)
        if axis == "y":
            # Vega band/point y scales run top-down (first category at top);
            # mpl's default y-axis is bottom-up.
            ax.invert_yaxis()
        return

    if mpl_scale.vtype == "log":
        set_scale("log")

    domain = mpl_scale.domain
    if isinstance(domain, list) and len(domain) == 2:
        set_lim(domain[0], domain[1])

    if mpl_scale.vtype in ("time", "utc"):
        import matplotlib.dates as mdates

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        axis_obj = ax.xaxis if axis == "x" else ax.yaxis
        axis_obj.set_major_locator(locator)
        axis_obj.set_major_formatter(formatter)
