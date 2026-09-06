#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run HRX and Vulkan release perplexity measurements with llama-perplexity.

Every selected model is measured once per backend and recorded in a stable JSON
artifact, including failed measurements. Vulkan and unflagged HRX measurements
are mandatory. The batched driver supplies the selected models carrying the
HRX XFAIL policy as runtime names. A failed HRX measurement for one of those
models is an XFAIL; a successful one is an XPASS and fails the worker so the
stale expectation cannot pass unnoticed.

Only completed measurement attempts are classified this way. Invalid inputs,
process invocation errors, result-writing errors, and cleanup failures remain
fatal and retain their normal exception path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from benchmark_output import (
    append_batch_log,
    atomic_write_json,
    merge_benchmark_output,
)
from run_batched_benchmark import ModelSpec, load_manifest

FINAL_ESTIMATE_PATTERN = re.compile(
    r"Final estimate: PPL = (?P<value>[0-9.]+) \+/- (?P<uncertainty>[0-9.]+)"
)


class PerplexityBenchmarkError(RuntimeError):
    """Raised when the perplexity benchmark cannot run as requested."""


@dataclass
class PerplexityPhase:
    name: str
    backend: str
    device: str
    output: Path
    log: Path


@dataclass(frozen=True)
class ResolvedModel:
    spec: ModelSpec
    path: Path


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_models(
    manifest_path: Path, models_dir: Path, names: list[str]
) -> list[ResolvedModel]:
    """Map Lemonade model names to the staged GGUF files they refer to."""
    manifest = load_manifest(manifest_path)
    specs_by_name = {spec.name: spec for spec in manifest.models}
    unknown_names = [name for name in names if name not in specs_by_name]
    if unknown_names:
        raise PerplexityBenchmarkError(
            f"Models are not in {manifest_path}: {unknown_names!r}"
        )
    resolved: list[ResolvedModel] = []
    for name in names:
        spec = specs_by_name[name]
        path = models_dir / spec.directory / spec.filename
        if not path.is_file():
            raise PerplexityBenchmarkError(f"Model {name} is not staged at {path}")
        resolved.append(ResolvedModel(spec=spec, path=path))
    return resolved


def log_llama_perplexity_devices(llama_perplexity: Path) -> None:
    """Record the devices llama-perplexity enumerates before measuring."""
    result = subprocess.run(
        [os.fspath(llama_perplexity), "--list-devices"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log(f"llama-perplexity --list-devices (exit {result.returncode}):")
    log(result.stdout.rstrip())


def parse_final_estimate(text: str) -> dict[str, float] | None:
    match = FINAL_ESTIMATE_PATTERN.search(text)
    if match is None:
        return None
    return {
        "value": float(match.group("value")),
        "uncertainty": float(match.group("uncertainty")),
    }


def run_perplexity(
    llama_perplexity: Path,
    model: ResolvedModel,
    corpus_file: Path,
    phase: PerplexityPhase,
    args: argparse.Namespace,
    log_handle: TextIO,
) -> tuple[dict[str, Any], bool]:
    """Measure one model on one device and record the outcome as a row."""
    command = [
        os.fspath(llama_perplexity),
        "-m",
        os.fspath(model.path),
        "-f",
        os.fspath(corpus_file),
        "-c",
        str(args.ctx),
        "--chunks",
        str(args.chunks),
        "-b",
        str(args.batch),
        "--device",
        phase.device,
        *args.perplexity_arg,
    ]
    log("++ " + shlex.join(command))
    log_handle.write(f"===== {model.spec.name} on {phase.device} =====\n")
    log_handle.write("++ " + shlex.join(command) + "\n")
    log_handle.flush()

    started = time.monotonic()
    exit_code: int | None = None
    timed_out = False
    try:
        result = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.run_timeout_seconds,
        )
        output = result.stdout
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        timed_out = True
    duration_s = time.monotonic() - started
    log_handle.write(output)
    if not output.endswith("\n"):
        log_handle.write("\n")
    log_handle.flush()

    ppl = parse_final_estimate(output)
    exited_with_error = exit_code != 0
    missing_estimate = ppl is None
    if timed_out:
        error: str | None = f"timed out after {args.run_timeout_seconds} seconds"
    elif exited_with_error:
        error = f"llama-perplexity exited with status {exit_code}"
    elif missing_estimate:
        error = "llama-perplexity did not print a final PPL estimate"
    else:
        error = None
    succeeded = error is None

    summary = f"{model.spec.name} on {phase.device}: "
    if succeeded:
        assert ppl is not None
        summary += f"PPL = {ppl['value']} +/- {ppl['uncertainty']}"
    else:
        summary += f"failed: {error}"
    log(f"{summary} in {duration_s:.1f}s")

    row = {
        "model": model.spec.name,
        "file": model.spec.filename,
        "batch": args.batch_number,
        "status": "ok" if succeeded else "failed",
        "error": error,
        "exit_code": exit_code,
        "duration_s": round(duration_s, 3),
        "ppl": ppl if succeeded else None,
        "log": phase.log.name,
        "command": command,
    }
    return row, succeeded


def run_phase(
    phase: PerplexityPhase,
    active_log: Path,
    models: list[ResolvedModel],
    llama_perplexity: Path,
    corpus_file: Path,
    corpus_sha256: str,
    hrx_xfail_models: set[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool]:
    """Measure every staged model on one device and build the batch document."""
    rows: list[dict[str, Any]] = []
    has_unexpected_outcomes = False
    phase_is_hrx = phase.backend == "hrx"
    active_log.parent.mkdir(parents=True, exist_ok=True)
    with active_log.open("a", encoding="utf-8") as log_handle:
        for model in models:
            row, succeeded = run_perplexity(
                llama_perplexity,
                model,
                corpus_file,
                phase,
                args,
                log_handle,
            )
            rows.append(row)
            model_is_flagged = model.spec.name in hrx_xfail_models
            xfail_applies = phase_is_hrx and model_is_flagged
            model_label = f"{model.spec.id} ({model.spec.name})"
            if succeeded:
                if xfail_applies:
                    log(
                        f"XPASS: {phase.name} perplexity for {model_label} "
                        "completed successfully; hrx.xfail is still set"
                    )
                    has_unexpected_outcomes = True
            elif xfail_applies:
                log(
                    f"XFAIL: {phase.name} perplexity for {model_label}: "
                    f"{row['error']}"
                )
            else:
                log(
                    f"FAIL: {phase.name} perplexity for {model_label}: "
                    f"{row['error']}"
                )
                has_unexpected_outcomes = True
    batch_data = {
        "schema_version": 1,
        "backend": phase.backend,
        "device": phase.device,
        "llama_perplexity": os.fspath(llama_perplexity),
        "settings": {
            "ctx": args.ctx,
            "chunks": args.chunks,
            "batch": args.batch,
            "extra_args": list(args.perplexity_arg),
        },
        "corpus": {
            "name": corpus_file.name,
            "sha256": corpus_sha256,
            "bytes": corpus_file.stat().st_size,
        },
        "models": rows,
    }
    return batch_data, has_unexpected_outcomes


def run(args: argparse.Namespace) -> int:
    hrx_xfail_models = set(args.hrx_xfail_models)
    llama_perplexity = args.llama_perplexity.resolve()
    models_dir = args.models_dir.resolve()
    corpus_file = args.corpus_file.resolve()
    if not corpus_file.is_file():
        raise PerplexityBenchmarkError(f"Corpus file is missing: {corpus_file}")
    corpus_sha256 = sha256_file(corpus_file)
    models = resolve_models(args.model_manifest, models_dir, args.models)
    phases = (
        PerplexityPhase(
            name="HRX",
            backend="hrx",
            device=args.hrx_device,
            output=args.hrx_output,
            log=args.hrx_log,
        ),
        PerplexityPhase(
            name="Vulkan",
            backend="vulkan",
            device=args.vulkan_device,
            output=args.vulkan_output,
            log=args.vulkan_log,
        ),
    )
    has_unexpected_outcomes = False

    # Precreate debug files so early failures still leave uploadable artifacts.
    for phase in phases:
        phase.log.parent.mkdir(parents=True, exist_ok=True)
        if args.batched:
            phase.log.touch(exist_ok=True)
        else:
            phase.log.write_text("", encoding="utf-8")

    log_llama_perplexity_devices(llama_perplexity)

    with tempfile.TemporaryDirectory(
        dir=models_dir.parent,
        prefix="perplexity-state-",
    ) as state_root_name:
        state_root = Path(state_root_name)
        for phase in phases:
            log(f"Starting {phase.name} perplexity phase on {phase.device}")
            active_log = phase.log
            if args.batched:
                active_log = state_root / phase.backend / "perplexity.log"
            try:
                batch_data, phase_has_unexpected_outcomes = run_phase(
                    phase,
                    active_log,
                    models,
                    llama_perplexity,
                    corpus_file,
                    corpus_sha256,
                    hrx_xfail_models,
                    args,
                )
            finally:
                if args.batched:
                    assert args.batch_number is not None
                    try:
                        append_batch_log(phase.log, active_log, args.batch_number)
                    except OSError as exc:
                        log(
                            "Warning: could not append "
                            f"{phase.backend} batch {args.batch_number} "
                            f"log to {phase.log}: {exc}"
                        )

            if args.batched:
                merged_count = merge_benchmark_output(phase.output, batch_data)
                log(
                    f"Merged {merged_count} {phase.backend} model(s) from "
                    f"batch {args.batch_number} into {phase.output}"
                )
            else:
                atomic_write_json(phase.output, batch_data)
                log(f"Wrote {phase.output}")
            has_unexpected_outcomes = (
                has_unexpected_outcomes or phase_has_unexpected_outcomes
            )

    if has_unexpected_outcomes:
        log(
            "Perplexity benchmark completed with unexpected FAIL or XPASS "
            "outcomes"
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch-number", type=int)
    parser.add_argument("--hrx-output", type=Path, required=True)
    parser.add_argument("--vulkan-output", type=Path, required=True)
    parser.add_argument("--hrx-log", type=Path, required=True)
    parser.add_argument("--vulkan-log", type=Path, required=True)
    parser.add_argument("--hrx-device", default="HRX0")
    parser.add_argument("--vulkan-device", default="Vulkan0")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--perplexity-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to llama-perplexity (repeatable).",
    )
    parser.add_argument("--hrx-xfail-models", nargs="*", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    if args.batched and args.batch_number is None:
        parser.error("--batch-number is required with --batched")
    if args.batch_number is not None and not args.batched:
        parser.error("--batch-number requires --batched")
    if args.batch_number is not None and args.batch_number < 1:
        parser.error("--batch-number must be a positive integer")
    try:
        return run(args)
    except (
        OSError,
        RuntimeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Perplexity benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
