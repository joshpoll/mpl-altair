"""M2 end-to-end verification: scatter + multi-series temporal line -> PNGs.

Run with: MPLBACKEND=Agg uv run python scripts/try_m2.py
"""
from __future__ import annotations

import os

import altair as alt
import numpy as np
import pandas as pd

import mplaltair

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(0)
scatter_df = pd.DataFrame({
    "val": rng.uniform(-5, 20, size=40),
    "v2": rng.uniform(0, 100, size=40),
})
scatter_chart = alt.Chart(scatter_df).mark_point().encode(x="val:Q", y="v2:Q")

ts = pd.DataFrame({
    "t": list(pd.date_range("2024-01-01", periods=8, freq="ME")) * 2,
    "v": [1.0, 3, 2, 5, 4, 6, 5, 7, 2.0, 4, 3, 6, 5, 7, 6, 8],
    "s": ["a"] * 8 + ["b"] * 8,
})
line_chart = alt.Chart(ts).mark_line().encode(x="t:T", y="v:Q", color="s:N")

fig1 = mplaltair.convert(scatter_chart)
fig1.savefig(os.path.join(OUT, "scatter.png"))
print("scatter xlim:", fig1.axes[0].get_xlim(), "ylim:", fig1.axes[0].get_ylim())
print("scatter xlabel/ylabel:", fig1.axes[0].get_xlabel(), fig1.axes[0].get_ylabel())
print("scatter data extent: val", scatter_df.val.min(), scatter_df.val.max(),
      "v2", scatter_df.v2.min(), scatter_df.v2.max())

fig2 = mplaltair.convert(line_chart)
fig2.savefig(os.path.join(OUT, "line_temporal.png"))
lines = fig2.axes[0].get_lines()
print("line count:", len(lines))
print("line colors:", [l.get_color() for l in lines])
print("line labels:", [l.get_label() for l in lines])
print("line xlabel/ylabel:", fig2.axes[0].get_xlabel(), fig2.axes[0].get_ylabel())

print("PNG sizes:",
      os.path.getsize(os.path.join(OUT, "scatter.png")),
      os.path.getsize(os.path.join(OUT, "line_temporal.png")))
