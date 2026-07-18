"""Render the full chart gallery: our mplaltair.convert() vs. vl-convert
ground truth, side by side, into scripts/out/gallery.html.

Run with: MPLBACKEND=Agg uv run python scripts/gallery.py
"""
from __future__ import annotations

import os
import sys
import traceback

import vl_convert as vlc

sys.path.insert(0, os.path.dirname(__file__))
import mplaltair
from gallery_charts import CHARTS

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def save_ground_truth(chart, name: str) -> str | None:
    vl_spec = chart.to_dict()
    png_bytes = vlc.vegalite_to_png(vl_spec)
    path = os.path.join(OUT, f"{name}_vl.png")
    with open(path, "wb") as f:
        f.write(png_bytes)
    return os.path.basename(path)


def save_ours(chart, name: str, **convert_kwargs) -> str | None:
    fig = mplaltair.convert(chart, **convert_kwargs)
    path = os.path.join(OUT, f"{name}_ours.png")
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return os.path.basename(path)


def main():
    rows = []
    for name, builder, kind in CHARTS:
        row = {"name": name, "kind": kind, "vl_img": None, "our_img": None, "error": None}
        try:
            row["vl_img"] = save_ground_truth(builder(), name)
        except Exception:
            row["error"] = f"ground-truth failed:\n{traceback.format_exc()}"
            print(f"[{name}] ground-truth FAILED")
        try:
            row["our_img"] = save_ours(builder(), name)
            print(f"[{name}] ours OK")
        except Exception:
            err = f"ours failed:\n{traceback.format_exc()}"
            row["error"] = (row["error"] or "") + "\n" + err
            print(f"[{name}] ours FAILED")
        rows.append(row)

    # Extra row: mpl-style theming payoff -- same stacked bar, style='dark_background'.
    theme_row = {"name": "bar_stacked (style='dark_background')", "kind": "theme demo"}
    try:
        from gallery_charts import bar_stacked
        theme_row["vl_img"] = save_ground_truth(bar_stacked(), "bar_stacked_theme_ref")
        theme_row["our_img"] = save_ours(bar_stacked(), "bar_stacked_dark", style="dark_background")
        theme_row["error"] = None
        print("[theme demo] OK")
    except Exception:
        theme_row["error"] = traceback.format_exc()
        print("[theme demo] FAILED")
    rows.append(theme_row)

    html_path = os.path.join(OUT, "gallery.html")
    with open(html_path, "w") as f:
        f.write(_render_html(rows))
    print(f"\nWrote {html_path}")

    n_ok = sum(1 for r in rows if r["error"] is None)
    print(f"{n_ok}/{len(rows)} charts rendered without error.")


def _render_html(rows) -> str:
    def img_cell(fname):
        if not fname:
            return "<td>(failed)</td>"
        # mtime query string busts the browser image cache: the PNG filenames
        # are stable across regenerations, and browsers happily reuse cached
        # images on plain refresh (especially for file:// pages).
        mtime = int(os.path.getmtime(os.path.join(OUT, fname)))
        return f'<td><img src="{fname}?v={mtime}" style="max-width:420px;border:1px solid #ccc"></td>'

    trs = []
    for r in rows:
        err_html = f"<pre style='color:#b00;white-space:pre-wrap'>{r['error']}</pre>" if r.get("error") else ""
        trs.append(f"""
        <tr>
          <td><b>{r['name']}</b><br><span style="color:#888">{r.get('kind','')}</span>{err_html}</td>
          {img_cell(r.get('vl_img'))}
          {img_cell(r.get('our_img'))}
        </tr>""")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>mpl-altair gallery</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ padding: 0.75rem; vertical-align: top; text-align: left; border-bottom: 1px solid #eee; }}
</style>
</head>
<body>
<h1>mpl-altair gallery</h1>
<p>Ground truth (vl-convert, pixel renderer) vs. mplaltair.convert() (native matplotlib). Semantic match expected, not pixel match.</p>
<table>
<tr><th>chart</th><th>vl-convert ground truth</th><th>mplaltair</th></tr>
{''.join(trs)}
</table>
</body>
</html>
"""


if __name__ == "__main__":
    main()
