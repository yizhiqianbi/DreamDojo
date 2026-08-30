#!/usr/bin/env python3
"""Download the DreamDojo 2B pretrain checkpoint and all jokeru datasets.

Both hub clients resume partial files, so rerunning this command is safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from modelscope_hub import HubApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--owner", default="jokeru")
    return parser.parse_args()


def download_checkpoint(max_workers: int) -> None:
    destination = PROJECT_ROOT / "checkpoints" / "DreamDojo"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading nvidia/DreamDojo 2B_pretrain to {destination}")
    snapshot_download(
        repo_id="nvidia/DreamDojo",
        allow_patterns=["2B_pretrain/**"],
        local_dir=destination,
        max_workers=max_workers,
    )

    vae_destination = PROJECT_ROOT / "checkpoints" / "Wan2.1"
    vae_destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the public Wan2.1 VAE to {vae_destination}")
    hf_hub_download(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        filename="Wan2.1_VAE.pth",
        local_dir=vae_destination,
    )

    embedding_destination = PROJECT_ROOT / "datasets"
    embedding_destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the public CR1 empty-prompt embedding to {embedding_destination}")
    hf_hub_download(
        repo_id="Cocoyawn32/cosmos-predict2p5-cr1-empty-embedding",
        repo_type="dataset",
        filename="cr1_empty_string_text_embeddings.pt",
        local_dir=embedding_destination,
    )

    reason_cache = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
    print(f"Caching the frozen Cosmos-Reason1-7B condition encoder in {reason_cache}")
    snapshot_download(
        repo_id="nvidia/Cosmos-Reason1-7B",
        revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
        cache_dir=reason_cache,
        max_workers=max_workers,
    )


def download_datasets(owner: str, max_workers: int) -> None:
    destination = PROJECT_ROOT / "datasets" / owner
    destination.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    page = api.list_repos("dataset", owner=owner, page_number=1, page_size=50)
    repos = sorted(repo.repo_id for repo in page.items)
    if len(repos) != page.total_count:
        raise RuntimeError(
            f"Expected {page.total_count} {owner} datasets, but only listed {len(repos)}"
        )
    print(f"Found {len(repos)} datasets owned by {owner}")
    for index, repo_id in enumerate(repos, start=1):
        repo_name = repo_id.split("/", 1)[1]
        local_dir = destination / repo_name
        print(f"[{index}/{len(repos)}] Downloading {repo_id} to {local_dir}")
        api.download_repo(
            repo_id,
            "dataset",
            revision="master",
            local_dir=local_dir,
            max_workers=max_workers,
        )


def main() -> None:
    args = parse_args()
    if not args.skip_checkpoint:
        download_checkpoint(args.max_workers)
    if not args.skip_datasets:
        download_datasets(args.owner, args.max_workers)


if __name__ == "__main__":
    main()
