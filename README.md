# Mask Vote

A phone page that shows random image/mask pairs, one at a time, for colleagues to rate
good (✓) or bad (✕). Swipe right/left or use the arrow keys; tap the image to hide the
outlines. Behind a shared password.

- **Page:** static, served by GitHub Pages from `docs/`.
- **Corpus:** a public Hugging Face dataset, added to in batches.
- **Votes:** a Google Sheet, written by an Apps Script web app (`apps_script/Code.gs`).

## Corpus layout

```
batches.json                    {"batches": ["batches/<name>.json", ...]}
batches/<name>.json             {"batch": name, "items": [{id, image, mask, w, h, cells, source}]}
items/<name>/<id>.jpg           image: 8-bit, 5th–95th percentile of non-zero pixels
items/<name>/<id>_mask.png      labels as colour (R·65536 + G·256 + B; 0 = background)
```

The page picks random items this device hasn't seen yet. New batches join the pool, and
nothing already uploaded changes.

## Setup (once)

1. **Votes sheet.** Make a Google Sheet. Open Extensions → Apps Script and replace the
   code with `apps_script/Code.gs`. Under Project Settings → Script properties, add
   `PASSWORD` set to the shared password. Then Deploy → New deployment → Web app, with
   *Execute as* Me and *Who has access* Anyone. Copy the `/exec` URL into
   `VOTE_URL` in `docs/config.js`.
2. **Corpus.** Run `huggingface-cli login`, then
   `pixi run upload --repo ghardo/mask-vote-corpus --batch <name>`. Check that
   `CORPUS_URL` in `docs/config.js` points at that dataset.
3. **Page.** Push this repo to GitHub and enable Settings → Pages → Deploy from branch
   `main`, folder `/docs`.

## Adding data

```bash
pixi run pack --name 2026-12-01_more --from-export /path/to/explorer/export --portrait
pixi run upload --repo ghardo/mask-vote-corpus --batch 2026-12-01_more
```

`--from-export` takes a folder exported from the training-data explorer (its
`manifest.tsv`); `--sets` keeps only sets whose folder name starts with the given
prefixes; `--portrait` keeps only images taller than wide (single trenches, which suit a
phone); `--n` takes a random sample.

## Votes

These go to the `votes` sheet, one row each: `time, item, vote (up/down), voter, device,
batch`. Skips aren't recorded. Item IDs match `batches/<name>.json`, which gives each
item's anonymous set number (`set-07`). `private/sets.json`, kept on this machine and
never uploaded, maps the numbers back to source folders.

## Local test

`pixi run serve` serves the page, `corpus/` and a stand-in vote endpoint (password
`test`, votes written to `votes.tsv`) at http://192.168.1.211:8770.
