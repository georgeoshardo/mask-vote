"""Pack image/mask pairs into one corpus batch, ready to upload to Hugging Face.

    pixi run pack --name 2026-11-02 --from-export /path/to/export --portrait

Corpus layout (a new batch never changes existing files except the batch list):

    batches.json                     {"batches": ["batches/<name>.json", ...]}
    batches/<name>.json              {"batch": name, "items": [{id, image, mask, w, h, cells, set}]}
    items/<name>/<id>.jpg            image, contrast 5th-95th percentile of non-zero pixels, 8-bit
    items/<name>/<id>_mask.png       labels as colour: label = R*65536 + G*256 + B, 0 = background

Nothing uploaded names a person: items carry an anonymous set number ("set-07"), and the
mapping from set numbers to source folders stays in private/sets.json (never uploaded),
so a folder keeps its number across batches. Batch names must not contain user IDs.

Masks are relabelled so every connected piece is its own cell. Images longer than
MAX_SIDE are shrunk (masks with nearest-neighbour, so labels survive).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage.measure import label

MAX_SIDE = 2048
SETS = Path(__file__).resolve().parent.parent / "private" / "sets.json"
USER_ID = re.compile(r"[a-z]{2,3}\d{2,5}", re.I)  # two or three letters then digits


def read(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        return np.squeeze(tifffile.imread(path))
    with Image.open(path) as im:
        return np.squeeze(np.asarray(im))


def to_8bit(a: np.ndarray, lo_pct: float = 5, hi_pct: float = 95) -> np.ndarray:
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    a = a.astype(np.float32)
    real = a[a > 0]
    lo, hi = np.percentile(real, (lo_pct, hi_pct)) if real.size else (0.0, 1.0)
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def labels_to_rgb(m: np.ndarray) -> np.ndarray:
    m = label(m, background=0, connectivity=2).astype(np.uint32)
    return np.stack([(m >> 16) & 255, (m >> 8) & 255, m & 255], -1).astype(np.uint8)


def pack_one(image: Path, mask: Path, out_dir: Path, item_id: str, portrait: bool = False) -> dict | None:
    im, m = read(image), read(mask)
    if portrait and im.shape[0] <= im.shape[1]:
        return None  # phone-shaped only: taller than wide
    if im.ndim == 3 and im.shape[-1] in (3, 4):
        pass
    elif im.ndim != 2:
        return None  # 3D stacks are skipped
    if m.ndim != 2 or m.shape != im.shape[:2]:
        return None
    g, rgb = Image.fromarray(to_8bit(im)), Image.fromarray(labels_to_rgb(m))
    if max(g.size) > MAX_SIDE:
        s = MAX_SIDE / max(g.size)
        size = (max(1, round(g.width * s)), max(1, round(g.height * s)))
        g, rgb = g.resize(size, Image.BOX), rgb.resize(size, Image.NEAREST)
    g.save(out_dir / f"{item_id}.jpg", quality=90)
    rgb.save(out_dir / f"{item_id}_mask.png", optimize=True)
    return {"w": g.width, "h": g.height, "cells": int(label(m, background=0, connectivity=2).max())}


def pairs_from_export(export: Path, sets: list[str]) -> list[tuple[Path, Path, str]]:
    """Image/mask pairs from a training-data-explorer export (its manifest.tsv)."""
    units: dict[str, dict] = {}
    for r in csv.DictReader(open(export / "manifest.tsv"), delimiter="\t"):
        if "extra_channels/" not in r["exported"]:
            units.setdefault(r["unit"], {})[r["role"]] = r["exported"]
    out = []
    for u in units.values():
        if "image" in u and "mask" in u and (not sets or any(u["image"].startswith(s) for s in sets)):
            if (export / u["image"]).exists() and (export / u["mask"]).exists():
                out.append((export / u["image"], export / u["mask"], u["image"].split("/")[0]))
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="batch name, e.g. 2026-11-02 (no names or user IDs)")
    p.add_argument("--from-export", type=Path, required=True)
    p.add_argument("--sets", nargs="*", default=[], help="only sets whose folder starts with one of these")
    p.add_argument("--n", type=int, default=0, help="random sample of this many pairs (0 = all)")
    p.add_argument("--portrait", action="store_true", help="only images taller than wide (single trenches)")
    p.add_argument("--out", type=Path, default=Path("corpus"))
    a = p.parse_args()
    if USER_ID.search(a.name) or not re.fullmatch(r"[A-Za-z0-9_.-]+", a.name):
        raise SystemExit(f"batch name {a.name!r} looks like it contains a user ID; use e.g. a date")
    pairs = pairs_from_export(a.from_export, a.sets)
    SETS.parent.mkdir(exist_ok=True)
    set_ids = json.loads(SETS.read_text()) if SETS.exists() else {}  # source folder -> set-NN
    if a.n and a.n < len(pairs):
        pairs = random.Random(0).sample(pairs, a.n)
    out_dir = a.out / "items" / a.name
    if (a.out / "batches" / f"{a.name}.json").exists():
        raise SystemExit(f"batch {a.name} already exists; give the new batch another name")
    out_dir.mkdir(parents=True)
    (a.out / "batches").mkdir(exist_ok=True)
    items = []
    for image, mask, source in pairs:
        item_id = f"{a.name}-{len(items):05d}"
        info = pack_one(image, mask, out_dir, item_id, a.portrait)
        if info:
            set_ids.setdefault(source, f"set-{len(set_ids) + 1:02d}")
            items.append({"id": item_id, "image": f"items/{a.name}/{item_id}.jpg",
                          "mask": f"items/{a.name}/{item_id}_mask.png", **info, "set": set_ids[source]})
    SETS.write_text(json.dumps(set_ids, indent=1))
    (a.out / "batches" / f"{a.name}.json").write_text(json.dumps({"batch": a.name, "items": items}))
    listing = a.out / "batches.json"
    batches = json.loads(listing.read_text())["batches"] if listing.exists() else []
    batches.append(f"batches/{a.name}.json")
    listing.write_text(json.dumps({"batches": batches}, indent=1))
    print(f"batch {a.name}: {len(items)} items ({len(pairs) - len(items)} skipped) -> {a.out}")


if __name__ == "__main__":
    main()
