"""M3+M4 end-to-end verification: bars/area/tick/rule + color/size scales -> PNGs.

Run with: MPLBACKEND=Agg uv run python scripts/try_m3m4.py
"""
from __future__ import annotations

import os

import altair as alt
import numpy as np
import pandas as pd

import mplaltair

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def save(chart, name):
    fig = mplaltair.convert(chart)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    print(f"{name}: saved ({os.path.getsize(path)} bytes)")
    return fig


# --- simple bar ---------------------------------------------------------
simple_df = pd.DataFrame({"cat": ["a", "b", "c"], "val": [3, 7, 5]})
simple_chart = alt.Chart(simple_df).mark_bar().encode(x="cat:N", y="val:Q")
fig = save(simple_chart, "bar_simple.png")
ax = fig.axes[0]
print("  simple bar: patch heights", [p.get_height() for p in ax.patches],
      "x positions", [p.get_x() for p in ax.patches])

# --- stacked bar (color) -------------------------------------------------
stack_df = pd.DataFrame({
    "cat": ["a", "a", "b", "b", "c", "c"],
    "grp": ["x", "y", "x", "y", "x", "y"],
    "val": [3, 5, 2, 7, 4, 1],
})
stack_chart = alt.Chart(stack_df).mark_bar().encode(x="cat:N", y="sum(val):Q", color="grp:N")
fig = save(stack_chart, "bar_stacked.png")
ax = fig.axes[0]
print("  stacked bar: legend labels", [t.get_text() for t in ax.get_legend().get_texts()])
print("  stacked bar: patch colors", [p.get_facecolor() for p in ax.patches])

# --- grouped bar (xOffset) -----------------------------------------------
grouped_chart = alt.Chart(stack_df).mark_bar().encode(x="cat:N", xOffset="grp:N", y="val:Q", color="grp:N")
fig = save(grouped_chart, "bar_grouped.png")
ax = fig.axes[0]
print("  grouped bar: x positions", sorted(round(p.get_x(), 3) for p in ax.patches))

# --- horizontal bar --------------------------------------------------------
hbar_chart = alt.Chart(simple_df).mark_bar().encode(y="cat:N", x="val:Q")
fig = save(hbar_chart, "bar_horizontal.png")

# --- histogram (bin=True + count()) --------------------------------------
rng = np.random.default_rng(1)
hist_df = pd.DataFrame({"v": rng.uniform(1, 7, size=9)})
hist_chart = alt.Chart(hist_df).mark_bar().encode(x=alt.X("v:Q", bin=True), y="count()")
fig = save(hist_chart, "hist.png")

# --- stacked area ----------------------------------------------------------
area_df = pd.DataFrame({
    "t": [1, 2, 3, 1, 2, 3],
    "grp": ["x", "x", "x", "y", "y", "y"],
    "val": [3, 5, 2, 1, 2, 4],
})
area_chart = alt.Chart(area_df).mark_area().encode(x="t:Q", y="val:Q", color="grp:N")
fig = save(area_chart, "area_stacked.png")

# --- tick plot ---------------------------------------------------------
tick_chart = alt.Chart(simple_df).mark_tick().encode(x="val:Q", y="cat:N")
fig = save(tick_chart, "tick.png")

# --- rule (y-only) ----------------------------------------------------------
rule_df = pd.DataFrame({"y": [4.0]})
rule_chart = alt.Chart(rule_df).mark_rule().encode(y="y:Q")
fig = save(rule_chart, "rule.png")

# --- scatter with categorical color legend ---------------------------------
cat_scatter_df = pd.DataFrame({
    "val": rng.uniform(0, 10, size=30),
    "v2": rng.uniform(0, 10, size=30),
    "cat": rng.choice(["a", "b", "c"], size=30),
})
cat_scatter_chart = alt.Chart(cat_scatter_df).mark_point().encode(x="val:Q", y="v2:Q", color="cat:N")
fig = save(cat_scatter_chart, "scatter_cat_color.png")
ax = fig.axes[0]
leg = ax.get_legend()
print("  scatter cat color: legend labels", [t.get_text() for t in leg.get_texts()])

# --- scatter with continuous color (colorbar) ------------------------------
cont_scatter_chart = alt.Chart(cat_scatter_df).mark_point().encode(x="val:Q", y="v2:Q", color="val:Q")
fig = save(cont_scatter_chart, "scatter_cont_color.png")
print("  scatter cont color: num axes (incl colorbar)", len(fig.axes))

# --- scatter with size encoding ---------------------------------------------
size_scatter_chart = alt.Chart(cat_scatter_df).mark_point().encode(x="val:Q", y="v2:Q", size="v2:Q")
fig = save(size_scatter_chart, "scatter_size.png")

# --- multi-series line (recheck color/legend order) ------------------------
ts = pd.DataFrame({
    "t": list(pd.date_range("2024-01-01", periods=8, freq="ME")) * 2,
    "v": [1.0, 3, 2, 5, 4, 6, 5, 7, 2.0, 4, 3, 6, 5, 7, 6, 8],
    "s": ["a"] * 8 + ["b"] * 8,
})
line_chart = alt.Chart(ts).mark_line().encode(x="t:T", y="v:Q", color="s:N")
fig = save(line_chart, "line_multi.png")
ax = fig.axes[0]
lines = ax.get_lines()
leg = ax.get_legend()
print("  multi-line: line labels", [l.get_label() for l in lines], "colors", [l.get_color() for l in lines])
print("  multi-line: legend labels", [t.get_text() for t in leg.get_texts()])
print("  multi-line: legend handle colors", [h.get_color() for h in leg.legend_handles])

print("\nAll M3+M4 charts rendered.")
