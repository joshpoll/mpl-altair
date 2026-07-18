"""Smoke tests: every gallery chart converts to a Figure with the expected
mpl-artist shape (bars -> patches, scatter -> collections, lines -> Line2D,
legend/colorbar presence). Run with: MPLBACKEND=Agg uv run pytest -q
"""
from __future__ import annotations

import base64
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from gallery_charts import CHARTS, ROW_COUNTS  # noqa: E402

import mplaltair  # noqa: E402


@pytest.mark.parametrize("name,builder,kind", CHARTS, ids=[c[0] for c in CHARTS])
def test_gallery_chart_converts(name, builder, kind):
    fig = mplaltair.convert(builder())
    try:
        assert fig is not None
        ax = fig.axes[0]
        assert len(fig.axes) >= 1

        if kind == "bar":
            assert len(ax.patches) == ROW_COUNTS[name]
        elif kind == "bar_legend":
            assert len(ax.patches) == ROW_COUNTS[name]
            assert ax.get_legend() is not None
        elif kind == "histogram":
            assert len(ax.patches) >= 1
        elif kind == "scatter":
            assert len(ax.collections) >= 1
        elif kind == "scatter_legend":
            assert len(ax.collections) >= 1
            assert ax.get_legend() is not None
        elif kind == "scatter_colorbar":
            assert len(ax.collections) >= 1
            assert len(fig.axes) >= 2  # colorbar is its own Axes
        elif kind == "line":
            assert len(ax.get_lines()) >= 1
        elif kind == "line_multi_legend":
            assert len(ax.get_lines()) >= 2
            assert ax.get_legend() is not None
        elif kind == "area_legend":
            assert len(ax.collections) >= 2  # one fill_between PolyCollection per series
            assert ax.get_legend() is not None
        elif kind == "tick":
            assert len(ax.collections) >= 1
        elif kind == "rule":
            assert len(ax.collections) >= 1 or len(ax.get_lines()) >= 1
        elif kind == "layered":
            assert len(ax.get_lines()) >= 1
            assert len(ax.collections) >= 1
        else:
            pytest.fail(f"unhandled kind {kind!r}")
    finally:
        plt.close(fig)


def test_embedding_in_caller_supplied_axes():
    """convert(chart, ax=...) draws into a caller-supplied Axes rather than
    creating its own Figure -- e.g. side-by-side in a subplot grid."""
    bar_builder = dict((n, b) for n, b, _ in CHARTS)["bar_simple"]
    scatter_builder = dict((n, b) for n, b, _ in CHARTS)["scatter_plain"]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    try:
        out1 = mplaltair.convert(bar_builder(), ax=ax1)
        out2 = mplaltair.convert(scatter_builder(), ax=ax2)

        assert out1 is fig
        assert out2 is fig
        assert len(ax1.patches) >= 1
        assert len(ax2.collections) >= 1
    finally:
        plt.close(fig)


def test_renderer_returns_png_mimebundle():
    from gallery_charts import bar_simple
    from mplaltair._renderer import renderer

    out = renderer(bar_simple().to_dict())
    assert isinstance(out, dict)
    assert "image/png" in out

    decoded = base64.b64decode(out["image/png"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_axes_box_matches_vega_inner_plot_rect():
    """Vega width/height describe the INNER plot rect, not the whole figure --
    after convert() the axes box itself (not the figure) should match those
    px dims, within a couple percent (constrained-layout chrome can shift
    slightly between the sizing pass and the final draw)."""
    from gallery_charts import scatter_plain
    from mplaltair._compile import compile_chart

    chart = scatter_plain()
    cspec, _ = compile_chart(chart.to_dict())

    fig = mplaltair.convert(chart)
    try:
        ax = fig.axes[0]
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        assert abs(bbox.width - cspec.width) <= 0.02 * cspec.width
        assert abs(bbox.height - cspec.height) <= 0.02 * cspec.height
    finally:
        plt.close(fig)


def test_tick_mark_draws_vertical_segments():
    """mark_tick with x=Q, y=N (value axis x, category axis y) should draw
    VERTICAL line segments (x0 == x1) at each point, spanning the y band's
    band_frac in axis units -- not horizontal dashes."""
    from gallery_charts import tick
    from mplaltair._compile import compile_chart
    from mplaltair._scales import build_scales

    chart = tick()
    cspec, _ = compile_chart(chart.to_dict())
    scales = build_scales(cspec)
    expected_extent = scales["y"].band_frac

    fig = mplaltair.convert(chart)
    try:
        ax = fig.axes[0]
        collections = ax.collections
        assert len(collections) >= 1
        lc = collections[0]
        segments = lc.get_segments()
        assert len(segments) >= 1
        for seg in segments:
            (x0, y0), (x1, y1) = seg
            assert x0 == pytest.approx(x1)
            assert abs(y1 - y0) == pytest.approx(expected_extent)
    finally:
        plt.close(fig)


def test_style_none_uses_callers_active_rcparams():
    """style=None must not apply our vega-lite.mplstyle -- bar colors should
    come from whatever prop_cycle is active in the caller's mpl session."""
    from gallery_charts import bar_simple

    custom_colors = ["#123456", "#abcdef"]
    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=custom_colors)}):
        fig = mplaltair.convert(bar_simple(), style=None)
        try:
            ax = fig.axes[0]
            face = ax.patches[0].get_facecolor()
            from matplotlib.colors import to_rgba

            assert face == to_rgba(custom_colors[0])
        finally:
            plt.close(fig)
