"""Dev script: compile a named example chart and print its CompiledSpec summary.

Usage: uv run scripts/dump_spec.py <chart_name>
where <chart_name> is one of the keys in CHARTS below.
"""
from __future__ import annotations

import sys

import altair as alt
import pandas as pd

from mplaltair._compile import compile_chart

df = pd.DataFrame({
    "cat": ["a", "a", "b", "b", "c", "c"],
    "grp": ["x", "y", "x", "y", "x", "y"],
    "val": [3, 5, 2, 7, 4, 1],
})
ts = pd.DataFrame({
    "t": pd.date_range("2024-01-01", periods=8, freq="ME"),
    "v": [1.0, 3, 2, 5, 4, 6, 5, 7],
    "s": list("abababab"),
})

CHARTS = {
    "scatter": alt.Chart(df).mark_point().encode(x="val:Q", y="grp:N"),
    "scatter_q": alt.Chart(ts).mark_point().encode(x="v:Q", y="v:Q"),
    "line_temporal": alt.Chart(ts).mark_line().encode(x="t:T", y="v:Q", color="s:N"),
}


def main(name: str) -> None:
    chart = CHARTS[name]
    cspec, warnings_ = compile_chart(chart.to_dict())
    print(f"== {name} ==")
    print("warnings:", warnings_)
    print("width/height:", cspec.width, cspec.height)
    print("datasets:", {k: len(v) for k, v in cspec.datasets.items()})
    print("scales:", {k: (s.get("type"), s.get("domain")) for k, s in cspec.scales.items()})
    print("marks:", [(m.get("type"), m.get("name")) for m in cspec.marks])
    print("axes:", cspec.axes)
    print("legends:", cspec.legends)
    print("signals:", cspec.signals)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scatter")
