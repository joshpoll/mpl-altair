"""Altair renderer entry point: VL spec dict -> MIME bundle.

Registered under the `altair.vegalite.v6.renderer` entry-point group (see
pyproject.toml) as `mplaltair`; `mplaltair.enable()` activates it.
"""
from __future__ import annotations

import base64
import io

from . import convert


def renderer(spec: dict) -> dict:
    """Render a Vega-Lite spec dict to a PNG MIME bundle.

    Altair's renderer protocol calls this with the resolved VL spec dict and
    expects a mimetype -> data dict back (base64-encoded for binary types).
    """
    import matplotlib.pyplot as plt

    fig = convert(spec)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return {"image/png": encoded}
