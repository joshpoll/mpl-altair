"""Shared gallery chart builders: used by scripts/gallery.py (renders +
contact sheet) and tests/test_smoke.py (parametrized smoke tests).

Each entry in CHARTS is (name, builder_fn, kind). `builder_fn()` returns a
fresh Altair Chart (fresh each call -- Altair charts carry mutable state, so
never share one instance across gallery + tests). `kind` tags the chart's
expected mpl-artist shape for the smoke tests' per-type assertions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt

_rng = np.random.default_rng(0)

_simple_df = pd.DataFrame({"cat": ["a", "b", "c"], "val": [3, 7, 5]})
_stack_df = pd.DataFrame({
    "cat": ["a", "a", "b", "b", "c", "c"],
    "grp": ["x", "y", "x", "y", "x", "y"],
    "val": [3, 5, 2, 7, 4, 1],
})
_hist_df = pd.DataFrame({"v": _rng.uniform(1, 7, size=60)})
_scatter_df = pd.DataFrame({
    "val": _rng.uniform(0, 10, size=30),
    "v2": _rng.uniform(0, 10, size=30),
    "cat": _rng.choice(["a", "b", "c"], size=30),
})
_log_df = pd.DataFrame({
    "val": _rng.uniform(1, 1000, size=30),
    "v2": _rng.uniform(1, 1000, size=30),
})
_ts_single = pd.DataFrame({
    "t": pd.date_range("2024-01-01", periods=8, freq="ME"),
    "v": [1.0, 3, 2, 5, 4, 6, 5, 7],
})
_ts_multi = pd.DataFrame({
    "t": list(pd.date_range("2024-01-01", periods=8, freq="ME")) * 2,
    "v": [1.0, 3, 2, 5, 4, 6, 5, 7, 2.0, 4, 3, 6, 5, 7, 6, 8],
    "s": ["a"] * 8 + ["b"] * 8,
})
_area_df = pd.DataFrame({
    "t": [1, 2, 3, 1, 2, 3],
    "grp": ["x", "x", "x", "y", "y", "y"],
    "val": [3, 5, 2, 1, 2, 4],
})
_rule_df = pd.DataFrame({"y": [4.0]})


def bar_simple():
    return alt.Chart(_simple_df).mark_bar().encode(x="cat:N", y="val:Q")


def bar_stacked():
    return alt.Chart(_stack_df).mark_bar().encode(x="cat:N", y="sum(val):Q", color="grp:N")


def bar_grouped():
    return alt.Chart(_stack_df).mark_bar().encode(
        x="cat:N", xOffset="grp:N", y="val:Q", color="grp:N"
    )


def bar_horizontal():
    return alt.Chart(_simple_df).mark_bar().encode(y="cat:N", x="val:Q")


def histogram():
    return alt.Chart(_hist_df).mark_bar().encode(x=alt.X("v:Q", bin=True), y="count()")


def scatter_plain():
    return alt.Chart(_scatter_df).mark_point().encode(x="val:Q", y="v2:Q")


def scatter_cat_color():
    return alt.Chart(_scatter_df).mark_point().encode(x="val:Q", y="v2:Q", color="cat:N")


def scatter_cont_color():
    return alt.Chart(_scatter_df).mark_point().encode(x="val:Q", y="v2:Q", color="val:Q")


def scatter_size():
    return alt.Chart(_scatter_df).mark_point().encode(x="val:Q", y="v2:Q", size="v2:Q")


def scatter_log():
    return alt.Chart(_log_df).mark_point().encode(
        x=alt.X("val:Q", scale=alt.Scale(type="log")),
        y=alt.Y("v2:Q", scale=alt.Scale(type="log")),
    )


def line_single():
    return alt.Chart(_ts_single).mark_line().encode(x="t:T", y="v:Q")


def line_multi_temporal():
    return alt.Chart(_ts_multi).mark_line().encode(x="t:T", y="v:Q", color="s:N")


def area_stacked():
    return alt.Chart(_area_df).mark_area().encode(x="t:Q", y="val:Q", color="grp:N")


def tick():
    return alt.Chart(_simple_df).mark_tick().encode(x="val:Q", y="cat:N")


def rule():
    return alt.Chart(_rule_df).mark_rule().encode(y="y:Q")


def layered_line_point():
    base = alt.Chart(_ts_single).encode(x="t:T", y="v:Q")
    return base.mark_line() + base.mark_point()


# (name, builder, kind) -- kind drives the smoke tests' per-type assertions.
CHARTS = [
    ("bar_simple", bar_simple, "bar"),
    ("bar_stacked", bar_stacked, "bar_legend"),
    ("bar_grouped", bar_grouped, "bar_legend"),
    ("bar_horizontal", bar_horizontal, "bar"),
    ("histogram", histogram, "histogram"),
    ("scatter_plain", scatter_plain, "scatter"),
    ("scatter_cat_color", scatter_cat_color, "scatter_legend"),
    ("scatter_cont_color", scatter_cont_color, "scatter_colorbar"),
    ("scatter_size", scatter_size, "scatter"),
    ("scatter_log", scatter_log, "scatter"),
    ("line_single", line_single, "line"),
    ("line_multi_temporal", line_multi_temporal, "line_multi_legend"),
    ("area_stacked", area_stacked, "area_legend"),
    ("tick", tick, "tick"),
    ("rule", rule, "rule"),
    ("layered_line_point", layered_line_point, "layered"),
]

# Underlying-data row counts for exact bar/patch-count smoke assertions.
ROW_COUNTS = {
    "bar_simple": len(_simple_df),
    "bar_horizontal": len(_simple_df),
    "bar_stacked": len(_stack_df),
    "bar_grouped": len(_stack_df),
}
