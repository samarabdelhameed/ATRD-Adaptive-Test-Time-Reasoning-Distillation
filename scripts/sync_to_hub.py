#!/usr/bin/env python3
"""
Sync to Hugging Face Hub

Uploads LoRA adapter checkpoints to Hugging Face Hub
for backup and Kaggle notebook access.

Usage:
    python scripts/sync_to_hub.py --adapter-path checkpoints/sft/final_adapter --repo-id username/atrd-adapter
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def sync_adapter(
    adapter_path: str,
    repo_id: str,
    commit_message: str = "Update LoRA adapter",
    private: bool = True,
) -> bool:
    """Upload adapter to Hugging Face Hub.

    Args:
        adapter_path: Local path to adapter directory.
        repo_id: HuggingFace repo ID (e.g., 'username/model-name').
        commit_message: Commit message for the upload.
        private: Whether the repo should be private.

    Returns:
        True if upload was successful.
    """
    from huggingface_hub import HfApi

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        print(f"✗ Adapter path does not exist: {adapter_dir}")
        return False

    api = HfApi()

    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
        print(f"Repository: https://huggingface.co/{repo_id}")

        # Upload the adapter directory
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print(f"✓ Adapter uploaded to {repo_id}")
        return True

    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync LoRA adapter to Hugging Face Hub"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/atrd-adapter)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Update LoRA adapter",
        help="Commit message",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repo public (default: private)",
    )
    args = parser.parse_args()

    success = sync_adapter(
        args.adapter_path,
        args.repo_id,
        args.message,
        private=not args.public,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
