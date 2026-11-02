"""Find the grid of a composite image (equal panels pasted side by side or stacked).

Two signs of panels along an axis, for a size that divides the image evenly:
- seams: the pixel jump between neighbouring columns (rows) spikes at every boundary;
- repetition (columns only): each panel holds one trench in the same place, so the
  brightness profile across the image repeats once per panel (panels' profiles correlate).
The panel size is the smallest one showing either sign with no cell lying on a cut. An
image showing neither is one panel. Within a set, images that show no sign (an empty
trench gives a flat profile) take the size most of the set's images agree on.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

MIN_PANEL = 16
SEAM_RATIO = 3.0  # a seam's jump must be this many times the typical jump
REPEAT_CORR = 0.7  # mean correlation between panels' brightness profiles


def jumps(a: np.ndarray, axis: int) -> np.ndarray:
    """Mean absolute step between neighbouring columns (axis=1) or rows (axis=0)."""
    return np.abs(np.diff(a.astype(np.float32), axis=axis)).mean(axis=1 - axis)


def repetition(a: np.ndarray, axis: int, p: int) -> float:
    """Mean correlation between the panels' brightness profiles along axis."""
    prof = a.astype(np.float32).mean(axis=1 - axis).reshape(-1, p)
    z = prof - prof.mean(1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-6
    c = z @ z.T
    k = len(c)
    return float((c.sum() - k) / (k * (k - 1)))


def panel_size(a: np.ndarray, mask: np.ndarray, axis: int) -> tuple[int, str]:
    """(panel size along axis, why); the whole length when no panels are found."""
    n = a.shape[axis]
    d = jumps(a, axis)
    typical = np.median(d) + 1e-6
    for p in range(MIN_PANEL, n // 2 + 1):
        if n % p:
            continue
        cuts = np.arange(p, n, p)
        before, after = np.take(mask, cuts - 1, axis=axis), np.take(mask, cuts, axis=axis)
        if ((before == after) & (before > 0)).mean() > 0.002:  # a cell runs across a cut
            continue
        seam = float(d[cuts - 1].min() / typical)  # d[i] is the step between i and i + 1
        rep = repetition(a, axis, p)
        if seam >= SEAM_RATIO:
            return p, f"seams {seam:.0f}x"
        if axis == 1 and rep >= REPEAT_CORR:
            return p, f"repeats r={rep:.2f}"
    return n, ""


def cuts_clear(mask: np.ndarray, axis: int, p: int) -> bool:
    n = mask.shape[axis]
    if n % p:
        return False
    cuts = np.arange(p, n, p)
    before, after = np.take(mask, cuts - 1, axis=axis), np.take(mask, cuts, axis=axis)
    return ((before == after) & (before > 0)).mean() <= 0.002


def set_grid(samples: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Counter, Counter]:
    """Panel heights and widths found across a set's sample images (sizes that split)."""
    hs, ws = Counter(), Counter()
    for a, m in samples:
        h, _ = panel_size(a, m, 0)
        w, _ = panel_size(a, m, 1)
        if h < a.shape[0]:
            hs[h] += 1
        if w < a.shape[1]:
            ws[w] += 1
    return hs, ws


def consensus(found: Counter, n_samples: int) -> int | None:
    """The panel size most samples agree on, if at least a third of them found it."""
    if not found:
        return None
    size, n = found.most_common(1)[0]
    return size if n >= n_samples / 3 else None


def apply(a: np.ndarray, mask: np.ndarray, panel_h: int | None, panel_w: int | None) -> dict:
    """The grid for one image of a set: the set's panel size where it fits cleanly."""
    ph = panel_h if panel_h and cuts_clear(mask, 0, panel_h) else a.shape[0]
    pw = panel_w if panel_w and cuts_clear(mask, 1, panel_w) else a.shape[1]
    return {"panel_h": ph, "panel_w": pw, "rows": a.shape[0] // ph, "cols": a.shape[1] // pw}


def grid(a: np.ndarray, mask: np.ndarray) -> dict:
    ph, why_h = panel_size(a, mask, 0)
    pw, why_w = panel_size(a, mask, 1)
    return {"panel_h": ph, "panel_w": pw, "rows": a.shape[0] // ph, "cols": a.shape[1] // pw,
            "why_rows": why_h, "why_cols": why_w}


def split(a: np.ndarray, mask: np.ndarray, g: dict):
    """Yield (row, col, image panel, mask panel)."""
    for r in range(g["rows"]):
        for c in range(g["cols"]):
            sl = (slice(r * g["panel_h"], (r + 1) * g["panel_h"]), slice(c * g["panel_w"], (c + 1) * g["panel_w"]))
            yield r, c, a[sl], mask[sl]
