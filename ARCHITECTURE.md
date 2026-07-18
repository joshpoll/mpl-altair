# Architecture

This document explains how mpl-altair turns an Altair chart into a matplotlib figure.

## The pipeline

A chart goes through four stages.

1. Altair produces a Vega-Lite spec. We call `chart.to_dict()` and get the chart as a plain dict.
2. vl-convert compiles the Vega-Lite spec to a Vega spec. This runs the real Vega-Lite compiler, so every default and inference rule is applied for us. The output spells out the scales, the marks, the axes, and the legends that the chart needs.
3. vegafusion evaluates the Vega spec's data transforms. We call `pre_transform_spec`, and it returns the same spec with every dataset replaced by concrete rows. After this stage, derived fields such as stack bounds, bin edges, and aggregate counts exist as plain columns in the data.
4. Our interpreter reads the evaluated Vega spec and draws it with matplotlib calls in data coordinates. Matplotlib owns axis scaling, tick placement, tick formatting, legends, and layout.

The point of stages 2 and 3 is that we never reimplement Vega-Lite semantics. An earlier project (matplotlib/mpl-altair, 2018) parsed Vega-Lite specs directly in Python and had to reimplement defaults, binning, aggregation, and stacking by hand, and it stalled under that load. In our design the official compiler and the official transform engine do that work, and the interpreter only maps already resolved structures onto matplotlib.

## The one rule the interpreter follows

The interpreter plots in data coordinates and never reads a Vega scale's `range` for position. A Vega spec maps data to pixels, so its scales carry both a `domain` (data values) and a `range` (pixel values). We use only the domain, the scale type, and the band paddings. Matplotlib then does its own mapping from data to pixels. This is why there is no y axis flip anywhere in the code. Vega's pixel y axis points down, but we never touch pixel values, and matplotlib's data y axis already points up.

The same rule drives the styling design. We do not copy Vega's rendered colors and fonts. Instead, the spec's named color ranges resolve against whatever matplotlib style is active, so a matplotlib style sheet controls the whole look of the chart.

## Module map

All modules live in `src/mplaltair/`.

- `__init__.py` holds `convert()`, which runs the whole pipeline, and `enable()`, which registers the Altair renderer.
- `_compile.py` wraps vl-convert and vegafusion and slices the evaluated spec into a `CompiledSpec` dataclass.
- `_scales.py` resolves scale domains and turns each Vega scale into an `MplScale` object.
- `_marks.py` walks the mark tree and draws each mark with matplotlib calls.
- `_guides.py` applies axis titles and grids, builds legends and colorbars, and unclips gridlines.
- `_layout.py` sizes the figure so the axes box matches the spec's plot area.
- `_renderer.py` is the Altair renderer entry point. It renders a chart to a PNG mime bundle for notebooks.
- `_style/vega-lite.mplstyle` is the default style sheet that approximates Vega-Lite's look.

## Compiling (`_compile.py`)

`compile_chart` runs vl-convert and vegafusion and returns a `CompiledSpec` with these fields.

- `datasets` maps each dataset name to its rows. Only datasets with inline values are kept.
- `scales` maps each scale name to its raw Vega scale dict.
- `marks` is the raw Vega mark list.
- `axes` is the Vega axes list after merging. The compiler emits two entries per axis, one that only draws the grid and one that draws labels, so we merge them into one entry per scale and orientation.
- `legends` is the raw Vega legends list.
- `signals` maps signal names to their static values. Signals with update expressions are skipped.
- `width` and `height` are the plot area size in pixels when the spec states them as numbers.

## Scales (`_scales.py`)

A Vega scale's domain can appear in several shapes, because vegafusion inlines the data that the domain refers to. `resolve_domain` handles the shapes we have observed.

- A data reference, e.g. `{"data": "data_0", "field": "cat"}`. We scan the named dataset. For a quantitative scale we take the min and max. For a categorical scale we collect the unique values.
- A pair of signal expressions of the form `(data("X")[0] || {}).min` and `.max`. vegafusion precomputes the min and max into a one row dataset, and we read that row.
- A bin signal of the form `[B.start, B.stop]`. We read the bin parameters from the matching static signal.
- A literal list, which passes through unchanged.

Anything else produces a warning and an unresolved domain, and the axis then falls back to matplotlib's autoscaling.

After resolution we apply the spec's `zero` flag (extend the domain to include zero) and `nice` flag (round the limits outward to round numbers, with a separate power of ten version for log scales), because Vega applies those at runtime and the compiled spec only carries the flags.

Each scale becomes an `MplScale`. The important methods are:

- `to_data(value)` maps a raw value into matplotlib data space. For most scales this is a passthrough. For temporal scales it parses the ISO date string. For band and point scales it returns the category's integer index.
- `cat_index`, `band_offset`, and `sub_band_width` handle categorical positioning. Categories sit at integer positions 0 to n-1. A secondary band scale, which Vega uses for grouped bars, maps a category to an offset inside its parent slot.
- `color_for(value)` maps a category to a color by its index in the scale's domain, so mark colors and legend entries always agree.
- `color_cmap_norm()` returns a colormap and a Normalize for continuous color scales.
- `size_for(value)` interpolates a value into the scale's range, which Vega states as symbol area in square pixels.

`apply_position_scale` pushes an x or y scale onto the Axes. It sets the axis limits from the domain, switches to a log scale when needed, installs date locators for temporal scales, and installs fixed category ticks for band and point scales.

Color resolution follows the styling rule above. A named range such as `"category"` resolves to the active matplotlib color cycle, and `"ramp"` resolves to the active default colormap. An explicit scheme name goes through a small table that maps Vega scheme names to matplotlib colormap names. The bundled style sheet sets the color cycle to Vega-Lite's default categorical palette and the default colormap to Blues, so the default output matches Vega-Lite, and any other matplotlib style replaces the palette wholesale.

## Marks (`_marks.py`)

`walk_drawable_marks` flattens the mark tree. Most charts have a flat list of marks, but line and area marks arrive wrapped in a group mark that partitions the data by series. For those we yield the inner mark together with the partition fields, and the drawer groups the rows with pandas and draws one artist per series.

`resolve_channel` evaluates one encode entry for one row. A Vega encode entry is a small dict, and the cases are:

- `{"scale": s, "field": f}` reads the field from the row and maps it through the scale's `to_data`.
- `{"value": v}` is a literal.
- `{"scale": s, "value": v}` is a literal in data space, e.g. a bar baseline of zero.
- `{"field": f}` with no scale uses the scale paired with that channel, e.g. `x2` uses the x scale.
- An entry with an `offset` sub dict adds a secondary band scale offset. Grouped bars use this.
- Signal expressions are skipped with a warning.

Each mark type has a drawer that builds arrays from the rows and makes one matplotlib call per series rather than per row.

- Bars are `rect` marks. Band position bars become `ax.bar` or `ax.barh` with one call per color series, with heights and bottoms taken from the stack fields that vegafusion computed. Bars on a linear axis, which is how histograms compile, use the bin edge fields for position and width.
- A thin `rect` (a fixed width or height of a few pixels) is a tick mark. We detect the thin dimension from the encode and draw a LineCollection of short strokes that span the category band.
- `symbol` marks become `ax.scatter`. Categorical color draws one scatter call per category so each gets a legend handle. Continuous color computes per point colors from the colormap. Sizes convert from Vega's area in square pixels to matplotlib's point squared units, including the factor of 4 over pi that accounts for Vega measuring true circle area while matplotlib measures squared diameter.
- `line` and `area` marks group by series, sort by x, drop undefined rows, and call `ax.plot` or `ax.fill_between`.
- `rule` marks become `axvline` or `axhline` when they span one axis, and a LineCollection when both endpoints are given.

Drawers register their artists in a registry keyed by scale name, together with a label per series. The legend code consumes this registry. After drawing, every new artist gets `clip_on` set to False, because Vega does not clip marks to the plot area and marks at a domain edge must render fully.

## Guides (`_guides.py`)

Axis handling is deliberately thin. We set the axis title and the grid visibility from the spec, and we let matplotlib choose tick positions and formats. Log axes also enable minor gridlines because Vega draws a gridline at every log tick. The one workaround is `unclip_gridlines`, which turns off clipping on gridlines so the lines at the axes limits are not lost to rasterization at the clip boundary. This is only partly effective. See the known issues section of the README.

Legends come from the spec's legend list. A categorical legend takes its handles from the mark registry. A continuous color legend becomes a colorbar built from the scale's colormap and Normalize. A size legend builds three proxy swatches at the domain minimum, midpoint, and maximum, styled like the chart's marks.

## Layout (`_layout.py`)

Vega's `width` and `height` describe the inner plot rectangle, and everything else (labels, titles, legends) sits outside it. Matplotlib's figure size describes the whole canvas. `finalize_figure_size` reconciles the two. It draws the figure once, measures the axes box, grows the figure by the deficit, and repeats once more if needed. The result is that the axes box matches the spec's plot area to within a couple of percent, and marker sizes read correctly relative to the plot. When the spec has no numeric width, which happens for band scale charts whose width comes from a step size, we compute a width from the category count and the step. Very small results are floored at a readable minimum, which is a deliberate divergence from Vega-Lite. When the caller passes their own Axes we skip all of this, because the caller owns layout.

## The renderer entry point (`_renderer.py`)

The package registers a renderer under Altair's `altair.vegalite.v6.renderer` entry point group. The renderer receives a Vega-Lite spec dict, runs `convert`, saves the figure to a PNG in memory, and returns a mime bundle. After `mplaltair.enable()`, a chart displayed in a notebook renders through matplotlib.

## Testing

`tests/test_smoke.py` converts every gallery chart and asserts on the artist structure, e.g. a bar chart yields the expected number of patches and a categorical color chart yields a legend. It also checks the axes box sizing, the tick mark orientation, the renderer mime bundle, and boundary gridline pixels. `scripts/gallery.py` renders every chart twice, once with vl-convert's own PNG renderer as ground truth and once with mpl-altair, and writes an HTML page that shows the pairs side by side. The gallery is for human review, and the goal is a semantic match, not a pixel match.
