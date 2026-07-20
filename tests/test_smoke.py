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
from gallery_charts import CHARTS, CONCAT_LEAF_COUNTS, FACET_PANEL_COUNTS, ROW_COUNTS  # noqa: E402

import mplaltair  # noqa: E402


@pytest.mark.parametrize("name,builder,kind", CHARTS, ids=[c[0] for c in CHARTS])
def test_gallery_chart_converts(name, builder, kind):
    fig = mplaltair.convert(builder())
    try:
        assert fig is not None
        ax = fig.axes[0]
        assert len(fig.axes) >= 1

        if kind == "facet":
            n_panels, n_visible = FACET_PANEL_COUNTS[name]
            assert len(fig.axes) == n_panels
            visible_axes = [a for a in fig.axes if a.get_visible()]
            assert len(visible_axes) == n_visible
            for panel_ax in visible_axes:
                n_artists = len(panel_ax.patches) + len(panel_ax.collections)
                assert n_artists >= 1, "panel drew no marks"
            # Shared scales: every visible panel has identical x/y limits.
            xlims = {panel_ax.get_xlim() for panel_ax in visible_axes}
            ylims = {panel_ax.get_ylim() for panel_ax in visible_axes}
            assert len(xlims) == 1
            assert len(ylims) == 1
            if name == "facet_color_legend":
                assert fig.legends, "expected a figure-level legend"
        elif kind == "facet_independent":
            visible_axes = [a for a in fig.axes if a.get_visible()]
            assert len(visible_axes) == 2
            for panel_ax in visible_axes:
                assert len(panel_ax.collections) >= 1, "panel drew no marks"
            # Independent y: panels must NOT share limits, and (unlike a
            # shared axis) every panel keeps its own y tick labels visible.
            ylims = {panel_ax.get_ylim() for panel_ax in visible_axes}
            assert len(ylims) == 2
            for panel_ax in visible_axes:
                labels = [t.get_text() for t in panel_ax.get_yticklabels() if t.get_text()]
                assert labels, "independent-y panel should show its own y tick labels"
        elif kind == "concat":
            leaves = [a for a in fig.axes if a.get_visible() and a.has_data()]
            assert len(leaves) == CONCAT_LEAF_COUNTS[name]
            for leaf_ax in leaves:
                n_artists = len(leaf_ax.patches) + len(leaf_ax.collections) + len(leaf_ax.get_lines())
                assert n_artists >= 1, "concat child drew no marks"
        elif kind == "bar":
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


def test_facet_panel_size_matches_target():
    """Each panel's axes box should match the per-panel target px size (the
    same target every panel shares, since only shared-scale facets are
    supported) -- the multi-Axes analog of
    `test_axes_box_matches_vega_inner_plot_rect`."""
    from gallery_charts import facet_column
    from mplaltair._compile import compile_chart
    from mplaltair._layout import target_axes_px
    from mplaltair._scales import build_scales

    chart = facet_column()
    cspec, _ = compile_chart(chart.to_dict())
    scales = build_scales(cspec)
    target_w, target_h = target_axes_px(cspec, scales)

    fig = mplaltair.convert(chart)
    try:
        fig.canvas.draw()
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            bbox = ax.get_window_extent()
            assert abs(bbox.width - target_w) <= 0.02 * target_w
            assert abs(bbox.height - target_h) <= 0.02 * target_h
    finally:
        plt.close(fig)


def test_convert_with_ax_raises_for_faceted_chart():
    """A faceted chart needs the whole figure -- convert(chart, ax=...) must
    refuse rather than silently drawing into just one panel."""
    from gallery_charts import facet_column

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError):
            mplaltair.convert(facet_column(), ax=ax)
    finally:
        plt.close(fig)


def test_convert_with_ax_raises_for_concat_chart():
    """Same requirement as a facet -- a concat/repeat layout needs the whole
    figure (possibly nested subfigures), not one caller-supplied Axes."""
    from gallery_charts import concat_hconcat

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError):
            mplaltair.convert(concat_hconcat(), ax=ax)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("name", ["concat_hconcat", "concat_vconcat", "concat_nested", "repeat_chart"])
def test_concat_leaf_size_within_tolerance(name):
    """Each concat/repeat leaf's axes box should approximate its own target
    px size. Unlike the facet/single-view 2% tolerance (`finalize_figure_size`
    resizes ONE uniform target), `_convert_concat`'s per-leaf `correction`
    fixed-point loop (see its docstring) is a coarser approximation for a
    grid of heterogeneously-sized, independently-scaled children -- this
    checks the tolerance it actually achieves on the gallery entries
    (empirically <=5%), not the tighter 2% bound.
    """
    from gallery_charts import CHARTS as _CHARTS
    from mplaltair._compile import compile_chart
    from mplaltair._layout import child_axes_px, compute_concat_sizes
    from mplaltair._scales import build_scales

    builder = {c[0]: c[1] for c in _CHARTS}[name]
    chart = builder()
    cspec, _ = compile_chart(chart.to_dict())
    scales = build_scales(cspec)

    def leaf_size(leaf):
        return child_axes_px(cspec, scales, leaf.width_signal, leaf.height_signal, leaf.x_scale, leaf.y_scale)

    def collect_targets(sized):
        if sized.children is None:
            return [(sized.true_w, sized.true_h)]
        out = []
        for child in sized.children:
            out.extend(collect_targets(child))
        return out

    targets = collect_targets(compute_concat_sizes(cspec.concat, leaf_size))

    fig = mplaltair.convert(chart)
    try:
        fig.canvas.draw()
        leaves = [a for a in fig.axes if a.get_visible() and a.has_data()]
        assert len(leaves) == len(targets)
        for ax, (target_w, target_h) in zip(leaves, targets):
            bbox = ax.get_window_extent()
            assert abs(bbox.width - target_w) <= 0.05 * target_w
            assert abs(bbox.height - target_h) <= 0.05 * target_h
    finally:
        plt.close(fig)


def test_independent_facet_scale_resolution_no_warning():
    """Phase 4: independent facet scale resolution is supported now -- it
    must not emit the old "not supported; using shared scales" warning."""
    import warnings

    from gallery_charts import facet_independent_y

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = mplaltair.convert(facet_independent_y())
    try:
        assert not any("independent" in str(w.message) for w in caught)
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


@pytest.mark.parametrize("name", ["bar_simple", "line_single", "tick"])
def test_boundary_gridlines_render(name, tmp_path):
    """Gridlines at the axes limits must survive rasterization (they sit on
    the clip boundary, and figure resizing recreates tick objects with
    clipping re-enabled -- both regressions seen in these charts)."""
    import io

    import numpy as np
    from PIL import Image

    builder = {c[0]: c[1] for c in CHARTS}[name]
    fig = mplaltair.convert(builder())
    try:
        ax = fig.axes[0]
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        im = np.asarray(Image.open(buf).convert("L"))
        h, w = im.shape
        bbox = ax.get_window_extent()

        def has_grid_pixels(px_lo, px_hi, axis):
            band = im[:, px_lo:px_hi] if axis == "col" else im[h - px_hi:h - px_lo, :]
            grey = (band > 200) & (band < 245)
            return grey.sum() > 20

        for gl, axis, edge_px in (
            (ax.xaxis.get_gridlines(), "col", bbox.x1),
            (ax.yaxis.get_gridlines(), "row", bbox.y1),
        ):
            if not gl or not gl[0].get_visible():
                continue
            lo, hi = int(edge_px) - 2, int(edge_px) + 2
            assert has_grid_pixels(lo, hi, axis), (
                f"{name}: no gridline pixels at the {axis} boundary (~{edge_px:.0f}px)"
            )
    finally:
        plt.close(fig)
