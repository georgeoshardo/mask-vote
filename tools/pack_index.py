"""Pack a balanced random sample of real trench panels from the whole training data.

    pixi run python -m tools.pack_index --name 2026-11-03 --total 10000

Reads every real set in the training-data explorer's index (the NAS copy), splits
composite and multi-trench images into panels (tools.panels, one panel size per set),
and keeps panels that contain cells, are taller than wide, and are not already in the
corpus (same pixels as an existing item). The sample is balanced: every set gets the
same quota, and quota a small set cannot fill is shared out among the larger ones.

Writes a sharded batch (2,000 items per shard):
    batches/<name>/shard-NNN.json       {"batch", "shard", "items": [...]}
    items/<name>/shard-NNN/<id>.jpg, <id>_mask.png
and adds each shard to batches.json as {"path", "n"}. The private item -> source list
goes to private/items_<name>.tsv (never uploaded).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label

from .pack import MAX_SIDE, SETS, USER_ID, labels_to_rgb, pairs_from_export, read, to_8bit
from .panels import apply, consensus, set_grid

INDEX = Path("/data/training-data-explorer/index.json")
EXPORT = Path("/mnt/nas/home/croc_incoming/training_data_exports/selected")
PRIVATE = Path(__file__).resolve().parent.parent / "private"
SHARD = 2000
SYNTHETIC = re.compile(r"symbac|synth", re.I)
EXCLUDE_FILE = Path(__file__).resolve().parent.parent / "private" / "exclude.txt"  # kept off the repo: it names folders
EXCLUDE = tuple(l.strip() for l in EXCLUDE_FILE.read_text().splitlines()
                if l.strip() and not l.startswith("#")) if EXCLUDE_FILE.exists() else ()


def real_folder(path: str) -> bool:
    return not SYNTHETIC.search(path) and not any(path.endswith(e) for e in EXCLUDE)


def set_key(folder: str) -> str:
    """The same key the explorer export uses (so set numbers carry over)."""
    parts = [p for p in folder.split("/") if p]
    parts = parts[2:] if parts[:1] == ["mnt"] else parts[1:] if parts[:1] == ["home"] else parts
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", p) for p in parts)


def plane(a: np.ndarray) -> np.ndarray | None:
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        return a[..., :3].mean(-1)
    return a if a.ndim == 2 else None


def digest(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes() + str(a.shape).encode(), digest_size=12).hexdigest()


def survey_set(job: tuple) -> list[dict]:
    """Every eligible panel of one set: {unit, r, c, h, w, hash}."""
    root, units, seed = job
    samples = []
    for u in random.Random(seed).sample(units, min(20, len(units))):
        a, m = plane(read(root / u["images"][0].lstrip("/"))), read(root / u["masks"][0].lstrip("/"))
        if a is not None and m.ndim == 2 and a.shape == m.shape:
            samples.append((a, m))
    if not samples:
        return []
    hs, ws = set_grid(samples)
    ph, pw = consensus(hs, len(samples)), consensus(ws, len(samples))
    out = []
    for k, u in enumerate(units):
        a, m = plane(read(root / u["images"][0].lstrip("/"))), read(root / u["masks"][0].lstrip("/"))
        if a is None or m.ndim != 2 or a.shape != m.shape:
            continue
        g = apply(a, m, ph, pw)
        for r in range(g["rows"]):
            for c in range(g["cols"]):
                sl = (slice(r * g["panel_h"], (r + 1) * g["panel_h"]), slice(c * g["panel_w"], (c + 1) * g["panel_w"]))
                pa, pm = a[sl], m[sl]
                if pa.shape[0] > pa.shape[1] and pm.max() > 0:
                    out.append({"unit": k, "r": r, "c": c, "ph": g["panel_h"], "pw": g["panel_w"], "hash": digest(pa)})
    return out


def quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Equal quota per set, with what small sets cannot use shared among the rest."""
    left, q, out = dict(counts), 0, {}
    budget = total
    while left and budget > 0:
        q = budget // len(left)
        small = {s: n for s, n in left.items() if n <= q}
        if not small:
            out.update({s: q for s in left}); budget -= q * len(left); break
        for s, n in small.items():
            out[s] = n; budget -= n; del left[s]
    for s in left:
        out.setdefault(s, 0)
    return out


def encode(job: tuple) -> dict:
    """Write one panel's image and mask; return its item record."""
    src_img, src_mask, cand, item_id, rel_dir, out_dir = job
    a, m = plane(read(src_img)), read(src_mask)
    sl = (slice(cand["r"] * cand["ph"], (cand["r"] + 1) * cand["ph"]), slice(cand["c"] * cand["pw"], (cand["c"] + 1) * cand["pw"]))
    g, rgb = Image.fromarray(to_8bit(a[sl])), Image.fromarray(labels_to_rgb(m[sl]))
    if max(g.size) > MAX_SIDE:
        s = MAX_SIDE / max(g.size)
        size = (max(1, round(g.width * s)), max(1, round(g.height * s)))
        g, rgb = g.resize(size, Image.BOX), rgb.resize(size, Image.NEAREST)
    g.save(out_dir / f"{item_id}.jpg", quality=90)
    rgb.save(out_dir / f"{item_id}_mask.png", optimize=True)
    return {"id": item_id, "image": f"{rel_dir}/{item_id}.jpg", "mask": f"{rel_dir}/{item_id}_mask.png",
            "w": g.width, "h": g.height, "cells": int(label(m[sl], background=0, connectivity=2).max())}


def existing_hashes() -> set[str]:
    """Pixel hashes of the images already in the first batch (built from the export)."""
    seen = set()
    for image, _, _ in pairs_from_export(EXPORT, []):
        a = plane(read(image))
        if a is not None and a.shape[0] > a.shape[1]:
            seen.add(digest(a))
    return seen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--total", type=int, default=10000)
    p.add_argument("--out", type=Path, default=Path("corpus"))
    a = p.parse_args()
    if USER_ID.search(a.name) or (a.out / "batches" / a.name).exists():
        raise SystemExit(f"bad or existing batch name {a.name!r}")
    index = json.loads(INDEX.read_text()); root = Path(index["root"])
    folders = [f for f in index["folders"] if real_folder(f["path"])]
    print(f"{len(folders)} real sets; surveying panels…", flush=True)
    with ProcessPoolExecutor(16) as pool:
        found = list(pool.map(survey_set, [(root, f["units"], i) for i, f in enumerate(folders)]))
    seen = existing_hashes()
    cands: dict[str, list] = defaultdict(list)
    for f, panels in zip(folders, found):
        for c in panels:
            if c["hash"] not in seen:  # skip copies of existing items and of each other
                seen.add(c["hash"]); cands[f["path"]].append(c)
    want = quotas({k: len(v) for k, v in cands.items()}, a.total)
    rng = random.Random(0)
    by_folder = {f["path"]: f for f in folders}
    set_ids = json.loads(SETS.read_text()) if SETS.exists() else {}
    picked = []
    for path in sorted(cands):
        for c in rng.sample(cands[path], want[path]):
            key = set_key(path)
            set_ids.setdefault(key, f"set-{len(set_ids) + 1:02d}")
            picked.append((path, c, set_ids[key]))
    rng.shuffle(picked)
    SETS.write_text(json.dumps(set_ids, indent=1))
    jobs, meta = [], []
    for n, (path, c, sid) in enumerate(picked):
        shard = n // SHARD
        rel_dir = f"items/{a.name}/shard-{shard:03d}"
        out_dir = a.out / rel_dir; out_dir.mkdir(parents=True, exist_ok=True)
        u = by_folder[path]["units"][c["unit"]]
        jobs.append((root / u["images"][0].lstrip("/"), root / u["masks"][0].lstrip("/"), c, f"{a.name}-{n:06d}", rel_dir, out_dir))
        meta.append((shard, sid, u["images"][0], c))
    print(f"packing {len(jobs)} panels…", flush=True)
    with ProcessPoolExecutor(16) as pool:
        items = list(pool.map(encode, jobs, chunksize=32))
    shards = defaultdict(list)
    for it, (shard, sid, _, _) in zip(items, meta):
        shards[shard].append({**it, "set": sid})
    (a.out / "batches" / a.name).mkdir(parents=True)
    listing = a.out / "batches.json"
    batches = json.loads(listing.read_text())["batches"] if listing.exists() else []
    for s, its in sorted(shards.items()):
        path = f"batches/{a.name}/shard-{s:03d}.json"
        (a.out / path).write_text(json.dumps({"batch": a.name, "shard": s, "items": its}))
        batches.append({"path": path, "n": len(its)})
    listing.write_text(json.dumps({"batches": batches}, indent=1))
    PRIVATE.mkdir(exist_ok=True)
    with open(PRIVATE / f"items_{a.name}.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "set", "source_image", "panel_row", "panel_col"])
        for it, (_, sid, src, c) in zip(items, meta):
            w.writerow([it["id"], sid, src, c["r"], c["c"]])
    per_set = defaultdict(int)
    for _, _, sid in picked:
        per_set[sid] += 1
    print(f"batch {a.name}: {len(items)} items in {len(shards)} shards; per set: {dict(sorted(per_set.items()))}")
    print(f"eligible panels per set before sampling: {sorted(((set_ids[set_key(k)], len(v)) for k, v in cands.items()))}")


if __name__ == "__main__":
    main()
