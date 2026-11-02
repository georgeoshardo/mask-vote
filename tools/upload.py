"""Upload one packed batch to the Hugging Face dataset and add it to the remote batch list.

    pixi run upload --repo ghardo/mask-vote-corpus --batch 2026-11-02

Needs `huggingface-cli login` once (a token with write access). Creates the dataset as
public if it does not exist yet (the page reads it without a token). The remote
batches.json is merged, not overwritten, so batches can be added from any machine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--batch", required=True)
    p.add_argument("--corpus", type=Path, default=Path("corpus"))
    a = p.parse_args()
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=False, exist_ok=True)
    try:
        remote = json.loads(Path(hf_hub_download(a.repo, "batches.json", repo_type="dataset")).read_text())["batches"]
    except (EntryNotFoundError, RepositoryNotFoundError):
        remote = []
    entry = f"batches/{a.batch}.json"
    if entry in remote:
        raise SystemExit(f"{a.batch} is already on {a.repo}")
    items = a.corpus / "items" / a.batch
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=items, path_in_repo=f"items/{a.batch}",
                      commit_message=f"Add images and masks of batch {a.batch}")
    listing = json.dumps({"batches": remote + [entry]}, indent=1).encode()
    api.create_commit(a.repo, repo_type="dataset", commit_message=f"List batch {a.batch}", operations=[
        CommitOperationAdd(entry, str(a.corpus / entry)),
        CommitOperationAdd("batches.json", listing),
    ])
    print(f"{a.batch}: {len(list(items.iterdir()))} files uploaded to https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
