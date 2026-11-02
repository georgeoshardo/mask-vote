"""Upload a packed batch to the Hugging Face dataset, or remove items from one.

    pixi run upload --repo ghardo/mask-vote-corpus --batch 2026-11-03
    pixi run upload --repo ghardo/mask-vote-corpus --remove-sets set-10 --from-batch 2026-11-02

Needs `huggingface-cli login` once (a token with write access). Creates the dataset as
public if it does not exist yet (the page reads it without a token). The remote
batches.json is merged, not overwritten, so batches can be added from any machine.
Entries are {"path", "n"} (n = items in that list); plain paths from older batches are
upgraded to that form.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


def remote_listing(repo: str) -> list:
    try:
        return json.loads(Path(hf_hub_download(repo, "batches.json", repo_type="dataset",
                                               force_download=True)).read_text())["batches"]
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []


def normalise(entries: list, corpus: Path) -> list[dict]:
    out = []
    for e in entries:
        if isinstance(e, str):
            e = {"path": e, "n": len(json.loads((corpus / e).read_text())["items"])}
        out.append(e)
    return out


def add_batch(api: HfApi, repo: str, batch: str, corpus: Path) -> None:
    local = json.loads((corpus / "batches.json").read_text())["batches"]
    mine = [e for e in normalise(local, corpus) if e["path"].startswith(f"batches/{batch}")]
    if not mine:
        raise SystemExit(f"no batch {batch} in {corpus}/batches.json")
    remote = normalise(remote_listing(repo), corpus)
    if any(e["path"] in {r["path"] for r in remote} for e in mine):
        raise SystemExit(f"{batch} is already on {repo}")
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=corpus,
                            allow_patterns=[f"items/{batch}/**", f"items/{batch}/*"])
    ops = [CommitOperationAdd(e["path"], str(corpus / e["path"])) for e in mine]
    ops.append(CommitOperationAdd("batches.json", json.dumps({"batches": remote + mine}, indent=1).encode()))
    api.create_commit(repo, repo_type="dataset", operations=ops, commit_message=f"List batch {batch}")
    print(f"{batch}: {sum(e['n'] for e in mine)} items in {len(mine)} lists uploaded to https://huggingface.co/datasets/{repo}")


def remove_sets(api: HfApi, repo: str, batch_path: str, sets: set[str], corpus: Path) -> None:
    """Drop the items of the given sets from one item list, here and on the hub."""
    listing = corpus / batch_path
    data = json.loads(listing.read_text())
    gone = [it for it in data["items"] if it["set"] in sets]
    data["items"] = [it for it in data["items"] if it["set"] not in sets]
    listing.write_text(json.dumps(data))
    remote = [e if e != batch_path and (not isinstance(e, dict) or e["path"] != batch_path)
              else {"path": batch_path, "n": len(data["items"])} for e in remote_listing(repo)]
    ops = [CommitOperationDelete(it[k]) for it in gone for k in ("image", "mask")]
    ops += [CommitOperationAdd(batch_path, str(listing)),
            CommitOperationAdd("batches.json", json.dumps({"batches": normalise(remote, corpus)}, indent=1).encode())]
    api.create_commit(repo, repo_type="dataset", operations=ops, commit_message=f"Remove {len(gone)} items")
    local = corpus / "batches.json"
    entries = [e if (e if isinstance(e, str) else e["path"]) != batch_path else {"path": batch_path, "n": len(data["items"])}
               for e in json.loads(local.read_text())["batches"]]
    local.write_text(json.dumps({"batches": entries}, indent=1))
    print(f"removed {len(gone)} items of {sorted(sets)} from {batch_path}; {len(data['items'])} remain")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--batch")
    p.add_argument("--remove-sets", nargs="*")
    p.add_argument("--from-batch", help="with --remove-sets: the (unsharded) batch to remove them from")
    p.add_argument("--corpus", type=Path, default=Path("corpus"))
    a = p.parse_args()
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=False, exist_ok=True)
    if a.remove_sets:
        remove_sets(api, a.repo, f"batches/{a.from_batch}.json", set(a.remove_sets), a.corpus)
    elif a.batch:
        add_batch(api, a.repo, a.batch, a.corpus)
    else:
        raise SystemExit("give --batch, or --remove-sets with --from-batch")


if __name__ == "__main__":
    main()
