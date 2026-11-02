"""Preview how each real set would be split into panels (cuts in red), for checking by eye.

    pixi run python -m tools.split_preview OUT_DIR
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries

from .pack import read, to_8bit
from .panels import apply, consensus, set_grid

INDEX = Path("/data/training-data-explorer/index.json")
SYNTHETIC = re.compile(r"symbac|synth", re.I)
PHI = 0.6180339887498949


def plane(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        return a[..., :3].mean(-1) if a.shape[-1] in (3, 4) else a[..., a.shape[-1] // 2]
    return a


def overlay(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    import colorsys
    rgb = np.stack([to_8bit(a)] * 3, -1)
    b = find_boundaries(m, mode="inner")
    for l in np.unique(m[b]):
        rgb[b & (m == l)] = [int(255 * c) for c in colorsys.hsv_to_rgb((int(l) * PHI) % 1, 0.9, 1)]
    return rgb


def set_plan(units: list, root: Path, n: int = 20):
    samples = []
    for u in random.Random(0).sample(units, min(n, len(units))):
        a, m = plane(read(root / u["images"][0].lstrip("/"))), read(root / u["masks"][0].lstrip("/"))
        if m.ndim == 2 and a.shape == m.shape:
            samples.append((a, m))
    hs, ws = set_grid(samples)
    return consensus(hs, len(samples)), consensus(ws, len(samples)), samples


def main(out: Path) -> None:
    index = json.loads(INDEX.read_text()); root = Path(index["root"])
    out.mkdir(parents=True, exist_ok=True)
    plans, tiles = [], []
    for i, f in enumerate(index["folders"]):
        if SYNTHETIC.search(f["path"]):
            continue
        ph, pw, samples = set_plan(f["units"], root)
        if not samples:
            continue
        a, m = samples[0]
        g = apply(a, m, ph, pw)
        rgb = overlay(a, m)
        full = Image.fromarray(rgb); d = ImageDraw.Draw(full)
        for c in range(1, g["cols"]):
            d.line([(c * g["panel_w"] - 0.5, 0), (c * g["panel_w"] - 0.5, a.shape[0])], fill=(255, 0, 0), width=max(1, a.shape[1] // 600))
        for r in range(1, g["rows"]):
            d.line([(0, r * g["panel_h"] - 0.5), (a.shape[1], r * g["panel_h"] - 0.5)], fill=(255, 0, 0), width=max(1, a.shape[0] // 300))
        s = min(1400 / full.width, 260 / full.height); full = full.resize((max(1, int(full.width * s)), max(1, int(full.height * s))), Image.NEAREST)
        k = min(g["cols"], 6) if g["cols"] > 1 else 1
        crop = Image.fromarray(rgb[: g["panel_h"] if g["rows"] > 1 else a.shape[0], : k * g["panel_w"]]); dc = ImageDraw.Draw(crop)
        for c in range(1, k):
            dc.line([(c * g["panel_w"] - 0.5, 0), (c * g["panel_w"] - 0.5, crop.height)], fill=(255, 0, 0), width=1)
        z = min(4.0, 260 / crop.height, 500 / crop.width); crop = crop.resize((max(1, int(crop.width * z)), max(1, int(crop.height * z))), Image.NEAREST)
        label = f"{i}: {f['path'].split('/', 3)[-1][-80:]}   {a.shape[0]}x{a.shape[1]} -> {g['rows']}x{g['cols']} panels of {g['panel_h']}x{g['panel_w']}"
        tile = Image.new("RGB", (1950, max(full.height, crop.height) + 22), (25, 25, 25))
        ImageDraw.Draw(tile).text((4, 4), label, fill="white"); tile.paste(full, (0, 22)); tile.paste(crop, (1430, 22))
        tiles.append(tile)
        plans.append({"folder": f["path"], "panel_h": ph, "panel_w": pw, "units": len(f["units"]), "example": [g["rows"], g["cols"]]})
    (out / "split_plan.json").write_text(json.dumps(plans, indent=1))
    per = 12
    for k in range(0, len(tiles), per):
        page = tiles[k:k + per]; H = sum(t.height + 8 for t in page)
        sheet = Image.new("RGB", (1950, H), (12, 12, 12)); y = 0
        for t in page:
            sheet.paste(t, (0, y)); y += t.height + 8
        sheet.save(out / f"split_preview_{k // per + 1}.png")
    print(len(tiles), "sets previewed ->", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
