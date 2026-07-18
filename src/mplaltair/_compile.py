"""Compile an Altair/Vega-Lite spec down to a resolved, mpl-friendly structure.

Pipeline: Vega-Lite dict -> Vega dict (vl-convert) -> pre-transformed Vega dict
(vegafusion, datasets inlined) -> CompiledSpec (our own resolved view).
"""
from __future__ import annotations

import json
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

    @classmethod
    def from_vega(cls, spec: dict) -> "CompiledSpec":
        width, height = get_plot_dims(spec)

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

        return cls(
            width=width,
            height=height,
            datasets=datasets,
            scales=scales,
            marks=spec.get("marks", []),
            axes=_merge_axes(spec.get("axes", [])),
            legends=spec.get("legends", []),
            signals=signals,
        )


def compile_chart(vl_spec: dict) -> tuple[CompiledSpec, list]:
    """Run the full VL -> Vega -> pre-transform -> CompiledSpec pipeline."""
    vg_spec = vl_to_vega(vl_spec)
    pre_spec, warnings = inline_transforms(vg_spec)
    return CompiledSpec.from_vega(pre_spec), warnings
