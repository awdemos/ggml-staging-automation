#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Download and verify the wikitext-2 test split used by llama-perplexity."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Same source as llama.cpp/scripts/get-wikitext-2.sh.
CORPUS_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
CORPUS_ZIP_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
CORPUS_ZIP_MEMBER = "wikitext-2-raw/wiki.test.raw"
CORPUS_FILENAME = "wiki.test.raw"
CORPUS_SHA256 = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 120.0


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be downloaded and verified."""


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as corpus_file:
        for chunk in iter(lambda: corpus_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive() -> bytes:
    """Fetch the pinned archive, retrying transient network failures."""
    request = urllib.request.Request(
        CORPUS_URL, headers={"User-Agent": "hrx-benchmark/1"}
    )
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            log(f"Downloading {CORPUS_URL} (attempt {attempt}/{DOWNLOAD_ATTEMPTS})")
            with urllib.request.urlopen(  # nosec B310 - fixed https URL
                request, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                archive = response.read()
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                log(f"Download attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed: {exc}")
                time.sleep(float(attempt))
            continue
        archive_sha256 = hashlib.sha256(archive).hexdigest()
        if archive_sha256 != CORPUS_ZIP_SHA256:
            raise CorpusError(
                f"Archive sha256 mismatch: expected {CORPUS_ZIP_SHA256}, "
                f"got {archive_sha256}"
            )
        return archive
    raise CorpusError(
        f"Could not download {CORPUS_URL} after {DOWNLOAD_ATTEMPTS} attempt(s): "
        f"{last_error}"
    )


def extract_corpus(archive: bytes) -> bytes:
    """Return the verified test split from the archive."""
    with zipfile.ZipFile(io.BytesIO(archive)) as archive_file:
        corpus = archive_file.read(CORPUS_ZIP_MEMBER)
    corpus_sha256 = hashlib.sha256(corpus).hexdigest()
    if corpus_sha256 != CORPUS_SHA256:
        raise CorpusError(
            f"Corpus sha256 mismatch: expected {CORPUS_SHA256}, got {corpus_sha256}"
        )
    return corpus


def fetch_corpus(output_dir: Path) -> Path:
    """Publish the verified corpus atomically, reusing a verified copy."""
    corpus_path = output_dir / CORPUS_FILENAME
    corpus_exists = corpus_path.is_file()
    corpus_is_verified = corpus_exists and sha256_file(corpus_path) == CORPUS_SHA256
    if corpus_is_verified:
        log(f"Reusing verified corpus {corpus_path}")
        return corpus_path
    if corpus_exists:
        log(f"Replacing corpus {corpus_path}: sha256 does not match the pin")

    corpus = extract_corpus(download_archive())
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / f".{CORPUS_FILENAME}.tmp"
    temporary_path.write_bytes(corpus)
    os.replace(temporary_path, corpus_path)
    log(f"Downloaded and verified corpus {corpus_path} ({len(corpus)} bytes)")
    return corpus_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        corpus_path = fetch_corpus(args.output_dir)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Corpus download failed: {exc}", file=sys.stderr)
        return 1
    print(corpus_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
