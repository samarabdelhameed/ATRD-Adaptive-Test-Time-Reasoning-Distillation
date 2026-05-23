#!/usr/bin/env python3
"""
Sync to Hugging Face Hub

Uploads LoRA adapter checkpoints to Hugging Face Hub
for backup and Kaggle notebook access.

Usage:
    python scripts/sync_to_hub.py --adapter-path checkpoints/sft/final_adapter --repo-id username/atrd-adapter
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
    from huggingface_hub import HfApi, login

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.error("Adapter path does not exist: %s", adapter_dir)
        return False

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        logger.error("adapter_config.json not found in %s", adapter_dir)
        return False
    try:
        with open(adapter_config) as f:
            json.load(f)
    except json.JSONDecodeError as e:
        logger.error("adapter_config.json is not valid JSON: %s", e)
        return False

    safetensors_files = list(adapter_dir.glob("*.safetensors"))
    if not safetensors_files:
        logger.error("No .safetensors files found in %s", adapter_dir)
        return False
    empty_safetensors = [f for f in safetensors_files if f.stat().st_size == 0]
    if empty_safetensors:
        logger.error(
            "Empty .safetensors files found: %s",
            ", ".join(f.name for f in empty_safetensors),
        )
        return False

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        logger.error(
            "Hugging Face token not found. Set HF_TOKEN environment variable "
            "or run `huggingface-cli login`."
        )
        return False

    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
        logger.info("Repository: https://huggingface.co/%s", repo_id)
    except Exception as e:
        logger.error("Failed to create or verify repository: %s", e)
        return False

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Uploading... this may take several minutes. (Attempt %d/%d)",
                attempt,
                max_retries,
            )
            api.upload_folder(
                folder_path=str(adapter_dir),
                repo_id=repo_id,
                commit_message=commit_message,
            )
            logger.info("Adapter uploaded to %s", repo_id)
            return True
        except Exception as e:
            logger.warning("Upload attempt %d/%d failed: %s", attempt, max_retries, e)
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info("Retrying in %d seconds...", wait)
                time.sleep(wait)
            else:
                logger.error("Upload failed after %d attempts: %s", max_retries, e)
                return False

    return False


def main() -> int:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

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
