#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Write the HRX/Vulkan perplexity comparison report as Markdown.

CI runs ``llama-perplexity`` over the same corpus once per backend
(``run_perplexity_benchmark.py``), leaving ``perplexity-hrx.json`` and
``perplexity-vulkan.json``. This script renders the pair, and optionally each
side against the latest ``main`` artifact, into the GitHub step summary so a
reviewer can see whether HRX reproduces Vulkan's estimates. Perplexity deltas
are informational: they never fail the job, only the report itself can.

Terms:

- A *run* is one model's measurement on one backend, keyed by model name — the
  comparison key here is just the model. A run has ``status`` ``ok`` (with a
  ``ppl`` estimate and uncertainty) or ``failed`` (with an ``error``, and the
  ``log`` and batch to inspect).
- The *settings line* is the corpus and chunking recipe every estimate in one
  artifact was measured with. Two artifacts are only comparable when their
  ``settings`` and corpus ``sha256`` are equal; ``check_comparable`` refuses
  otherwise, because a PPL delta across corpora means nothing.

The artifact shape, reduced to the fields this report reads::

    {"settings": {"ctx": 512, "chunks": 8, "batch": 512, "extra_args": []},
     "corpus": {"name": "wikitext-2-raw/wiki.test.raw", "sha256": "..."},
     "models": [
       {"model": "llama-3.1-8b", "status": "ok", "duration_s": 41.2,
        "ppl": {"value": 6.1234, "uncertainty": 0.0312}},
       {"model": "qwen3-8b", "status": "failed", "duration_s": 3.0,
        "error": "exit code 1", "log": "perplexity-hrx.log", "batch": 2}]}

The command line, the exit-code contract (current pair fatal, ``main``
baseline best-effort), and the matching rule live in ``benchmark_report`` and
are shared with the Lemonade report; this file supplies ``kind="perplexity"``
and the table builder. Under that rule a model is compared only when both
artifacts contain it; a model present on one side only is skipped, and a pair
with no model in common renders a one-line note instead of a table.

Output order on stdout: the partial-failure note, the settings line and
HRX/Vulkan table, then either the two current-versus-main tables or the "no
usable main perplexity artifact" note. Rows follow the current (left) artifact's
model order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from benchmark_report import (
    UNAVAILABLE_MEASUREMENT,
    ReportError,
    format_code,
    format_table_cell,
    match_indexed,
    run_report_cli,
)

Perplexity = dict[str, Any]
Run = dict[str, Any]
ComparisonMatch = tuple[str, Run, Run]
PARTIAL_FAILURE_NOTE = (
    "`—` marks an unavailable measurement; failed runs are described in the "
    "Status column with the log file and batch to inspect. Perplexity deltas "
    "are informational and never fail the job."
)


def run_succeeded(run: Run) -> bool:
    """Return whether one run produced a usable estimate."""
    status = run["status"]
    if status not in ("ok", "failed"):
        raise ReportError(f"Run status must be ok or failed, got {status!r}")
    return status == "ok"


def run_metric(run: Run, *fields: str) -> int | float:
    """Return one finite, non-negative measurement."""
    value: Any = run
    for field in fields:
        value = value[field]
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ReportError(
            f"Run {'.'.join(fields)} must be a finite non-negative number"
        )
    return value


def validate_run(run: Run) -> None:
    """Validate the run fields consumed by the report."""
    run_metric(run, "duration_s")
    if run_succeeded(run):
        run_metric(run, "ppl", "value")
        run_metric(run, "ppl", "uncertainty")
        return
    error = run["error"]
    error_is_string = isinstance(error, str)
    error_is_present = error_is_string and bool(error)
    if not error_is_present:
        raise ReportError("Failed run must describe its error")


def index_runs(perplexity: Perplexity, backend_label: str) -> dict[str, Run]:
    """Index runs by model and reject ambiguous comparison identities."""
    runs: dict[str, Run] = {}
    for run in perplexity["models"]:
        validate_run(run)
        model = run["model"]
        if model in runs:
            raise ReportError(
                f"Duplicate {backend_label} model: {model!r}"
            )
        runs[model] = run
    return runs


def check_comparable(
    left: Perplexity,
    right: Perplexity,
    *,
    left_label: str,
    right_label: str,
) -> None:
    """Refuse to compare estimates that were not measured the same way."""
    same_settings = left["settings"] == right["settings"]
    same_corpus = left["corpus"]["sha256"] == right["corpus"]["sha256"]
    if not same_settings:
        raise ReportError(
            f"{left_label} and {right_label} used different perplexity settings: "
            f"{left['settings']!r} vs {right['settings']!r}"
        )
    if not same_corpus:
        raise ReportError(
            f"{left_label} and {right_label} used different corpora"
        )


def match_runs(
    left: Perplexity,
    right: Perplexity,
    *,
    left_label: str = "HRX",
    right_label: str = "Vulkan",
) -> list[ComparisonMatch]:
    """Pair the runs of every model both sides measured, in left order."""
    check_comparable(left, right, left_label=left_label, right_label=right_label)
    return match_indexed(
        index_runs(left, left_label),
        index_runs(right, right_label),
    )


def format_ppl(run: Run) -> str:
    if not run_succeeded(run):
        return UNAVAILABLE_MEASUREMENT
    return f"{run['ppl']['value']:.4f} ± {run['ppl']['uncertainty']:.4f}"


def format_delta(left: Run, right: Run) -> str:
    """Express the left estimate minus the right estimate."""
    both_succeeded = run_succeeded(left) and run_succeeded(right)
    if not both_succeeded:
        return UNAVAILABLE_MEASUREMENT
    return f"{left['ppl']['value'] - right['ppl']['value']:+.4f}"


def format_ratio(left: Run, right: Run) -> str:
    """Express the left estimate as a multiple of the right estimate."""
    both_succeeded = run_succeeded(left) and run_succeeded(right)
    if not both_succeeded:
        return UNAVAILABLE_MEASUREMENT
    right_value = right["ppl"]["value"]
    if right_value == 0:
        return "N/A"
    return f"{left['ppl']['value'] / right_value:.4f}"


def format_duration(run: Run) -> str:
    return f"{run['duration_s']:.0f}"


def format_run_status(run: Run, label: str) -> str:
    """Describe one failed run with the evidence needed to inspect it."""
    location = format_code(run["log"])
    if run.get("batch") is not None:
        location += f", batch {run['batch']}"
    return f"{label} failed: {format_table_cell(run['error'])} (see {location})"


def format_pair_status(
    left: Run,
    right: Run,
    left_label: str,
    right_label: str,
) -> str:
    """Summarize failures across two backends."""
    statuses = [
        format_run_status(run, label)
        for run, label in ((left, left_label), (right, right_label))
        if not run_succeeded(run)
    ]
    return "; ".join(statuses) if statuses else "OK"


def format_settings_line(perplexity: Perplexity) -> str:
    """Describe how the estimates were measured."""
    settings = perplexity["settings"]
    corpus = perplexity["corpus"]
    extra_args = settings.get("extra_args") or []
    arguments = (
        " ".join(format_code(argument) for argument in extra_args)
        if extra_args
        else "_(none)_"
    )
    return (
        f"**Corpus:** {format_code(corpus['name'])} "
        f"(sha256 {format_code(corpus['sha256'][:12])}) · "
        f"**Chunks:** {format_code(settings['chunks'])} × "
        f"{format_code(settings['ctx'])} tokens · "
        f"**Batch:** {format_code(settings['batch'])} · "
        f"**Extra llama-perplexity arguments:** {arguments}"
    )


def format_comparison_table(
    matches: Sequence[ComparisonMatch],
    settings_line: str,
    *,
    title: str = "Perplexity HRX/Vulkan comparison",
    left_label: str = "HRX",
    right_label: str = "Vulkan",
) -> str:
    """Format already-matched runs without performing I/O or validation."""
    lines = [
        f"## {title}",
        "",
        settings_line,
        "",
        f"Δ PPL is {left_label} PPL minus {right_label} PPL; ratio is "
        f"{left_label} PPL divided by {right_label} PPL.",
        "",
    ]
    if not matches:
        lines.append(
            f"No model was benchmarked by both {left_label} and {right_label}."
        )
        return "\n".join(lines)
    lines.extend(
        (
            f"| Model | Status | {left_label} PPL | {right_label} PPL | Δ PPL | "
            f"Ratio | {left_label} time (s) | {right_label} time (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for model, left_run, right_run in matches:
        lines.append(
            f"| {format_code(model)} | "
            f"{format_pair_status(left_run, right_run, left_label, right_label)} | "
            f"{format_ppl(left_run)} | "
            f"{format_ppl(right_run)} | "
            f"{format_delta(left_run, right_run)} | "
            f"{format_ratio(left_run, right_run)} | "
            f"{format_duration(left_run)} | "
            f"{format_duration(right_run)} |"
        )
    return "\n".join(lines).rstrip()


def format_main_comparisons(
    current_hrx: Perplexity,
    current_vulkan: Perplexity,
    main_hrx: Perplexity,
    main_vulkan: Perplexity,
    baseline_run_url: str | None = None,
) -> str:
    """Format both current-versus-main comparisons atomically."""
    hrx_matches = match_runs(
        current_hrx,
        main_hrx,
        left_label="Current HRX",
        right_label="Main HRX",
    )
    vulkan_matches = match_runs(
        current_vulkan,
        main_vulkan,
        left_label="Current Vulkan",
        right_label="Main Vulkan",
    )

    sections = []
    if baseline_run_url:
        sections.append(
            f"Main perplexity baseline: [CI run on `main`]({baseline_run_url})"
        )
    sections.extend(
        (
            format_comparison_table(
                hrx_matches,
                format_settings_line(current_hrx),
                title="Perplexity current HRX/main HRX comparison",
                left_label="Current HRX",
                right_label="Main HRX",
            ),
            format_comparison_table(
                vulkan_matches,
                format_settings_line(current_vulkan),
                title="Perplexity current Vulkan/main Vulkan comparison",
                left_label="Current Vulkan",
                right_label="Main Vulkan",
            ),
        )
    )
    return "\n\n".join(sections)


def format_report(hrx: Perplexity, vulkan: Perplexity) -> str:
    """Match perplexity data, then format the comparison section."""
    matches = match_runs(hrx, vulkan)
    return "\n\n".join(
        (
            PARTIAL_FAILURE_NOTE,
            format_comparison_table(matches, format_settings_line(hrx)),
        )
    )


def main() -> int:
    return run_report_cli(
        kind="perplexity",
        report_label="perplexity",
        description=__doc__,
        format_report=format_report,
        format_main_comparisons=format_main_comparisons,
    )


if __name__ == "__main__":
    raise SystemExit(main())
