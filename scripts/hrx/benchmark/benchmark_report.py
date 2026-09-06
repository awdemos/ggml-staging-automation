# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared CLI, matching rule, and Markdown helpers for the HRX report scripts.

Two scripts turn CI benchmark artifacts into Markdown for the GitHub step
summary: ``write_lemonade_benchmark_report.py`` (Lemonade throughput/TTFT) and
``write_perplexity_report.py`` (llama-perplexity). Each compares an HRX artifact
against a Vulkan artifact from the same run, and optionally each of those
against the latest artifact from ``main``. The two scripts differ in what a
measurement *is* and how a table row is drawn; they agree on everything
around that — the command line the workflow calls, which failures are fatal,
and the rule for deciding which measurements may be compared. This module
owns the agreed part so the rule is written once.

Terms:

- *Comparison key*: the identity of one measurement inside an artifact. For
  perplexity it is the model name; for Lemonade it is
  ``(model, recipe, ctx_size, backend_args, scenario)``. The backend is never
  part of the key because it is exactly what differs between the two files.
- *Index*: an insertion-ordered ``dict`` from comparison key to the measurement
  it identifies. Each script builds its own indexes and validates entries; this
  module only consumes them.
- *Current pair* / *baseline pair*: the HRX and Vulkan artifacts of this run,
  and the HRX and Vulkan artifacts downloaded from the latest ``main`` run.
- *Kind*: the noun the CLI uses for the artifact, ``benchmark`` or
  ``perplexity``. It appears in the argument names and diagnostics.

Matching (``match_indexed``) pairs every comparison key present in both
indexes and returns ``[(key, left_item, right_item), ...]`` in left order; a
key found on one side only is skipped, never an error. Everything that decides
whether two measurements may sit in one row is therefore in the key, and the
scripts render a pair's failures (a backend with no successful run) rather than
refusing to pair. The two files routinely disagree on what they contain: a
batch's HRX and Vulkan phases run one after the other and merge per phase, so a
Vulkan failure leaves the HRX artifact with extra models; a pull request
benchmarks the ``smoke`` tier while the ``main`` baseline was built from
``full``; and a baseline from an older commit may carry different scenarios.
An earlier rule required identical key sets (with the baseline allowed to be a
superset) and lost the whole comparison, or the whole baseline block, on any
of those::

    >>> left = {("A", "p1"): 1, ("A", "p2"): 2, ("B", "p1"): 3}
    >>> right = {("A", "p1"): 10, ("C", "p1"): 30, ("A", "p2"): 20}
    >>> match_indexed(left, right)
    [(('A', 'p1'), 1, 10), (('A', 'p2'), 2, 20)]

The CLI (``run_report_cli``) is what the workflow invokes, once per script::

    write_<script>.py <hrx>.json <vulkan>.json \\
        --baseline-hrx-<kind> main/<hrx>.json \\
        --baseline-vulkan-<kind> main/<vulkan>.json \\
        --baseline-run-url https://github.com/.../actions/runs/123 \\
        >> "$GITHUB_STEP_SUMMARY"

Its exit-code contract is deliberately asymmetric:

- The current pair must load and compare, otherwise the exit status is 1, a
  one-line diagnostic goes to stderr, and *nothing* is written to stdout: a
  wrong comparison in the summary is worse than a missing one.
- The baseline is best-effort. If either baseline path is omitted, a file is
  missing or malformed, or the baseline cannot be matched, stderr gets a
  diagnostic and the report ends with ``No usable main <kind> artifact is
  available for comparison.`` — exit status 0 either way. A missing baseline
  is routine (first run on a branch, expired artifact) and must not hide the
  current results.

The artifacts are produced by other jobs and downloaded, so they sit outside
this repository's trust boundary. Every load-or-compare step is therefore
guarded by ``REPORT_INPUT_ERRORS``: any shape problem — a missing key, a
string where a number was expected, an unreadable file — becomes the
diagnostic above rather than a traceback that would fail the step and drop
the whole summary. Scripts raise ``ReportError`` for the problems they detect
themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

UNAVAILABLE_MEASUREMENT = "—"

K = TypeVar("K")
V = TypeVar("V")


class ReportError(RuntimeError):
    """Raised when two report inputs cannot be compared safely."""


REPORT_INPUT_ERRORS = (
    OSError,
    json.JSONDecodeError,
    ReportError,
    IndexError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
)


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON artifact without coupling I/O to report formatting."""
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def format_code(value: object) -> str:
    """Format trusted metadata as an inline Markdown code span."""
    text = str(value).replace("\n", " ")
    fence = "``" if "`" in text else "`"
    return f"{fence}{text}{fence}"


def format_table_cell(value: object) -> str:
    """Escape labels for a Markdown table cell."""
    return str(value).replace("\n", " ").replace("|", "\\|")


def match_indexed(
    left: Mapping[K, V],
    right: Mapping[K, V],
) -> list[tuple[K, V, V]]:
    """Pair every key present in both indexes, in left order."""
    return [(key, left[key], right[key]) for key in left if key in right]


def run_report_cli(
    *,
    kind: str,
    report_label: str,
    description: str | None,
    format_report: Callable[[dict[str, Any], dict[str, Any]], str],
    format_main_comparisons: Callable[
        [
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            str | None,
        ],
        str,
    ],
) -> int:
    """Parse the shared command line, build the report, return the exit code."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(f"hrx_{kind}", type=Path)
    parser.add_argument(f"vulkan_{kind}", type=Path)
    parser.add_argument(f"--baseline-hrx-{kind}", type=Path)
    parser.add_argument(f"--baseline-vulkan-{kind}", type=Path)
    parser.add_argument("--baseline-run-url")
    args = parser.parse_args()
    baseline_unavailable_message = (
        f"No usable main {kind} artifact is available for comparison."
    )

    try:
        hrx = load_json(getattr(args, f"hrx_{kind}"))
        vulkan = load_json(getattr(args, f"vulkan_{kind}"))
        report = format_report(hrx, vulkan)
    except REPORT_INPUT_ERRORS as exc:
        print(f"Could not write {report_label} report: {exc}", file=sys.stderr)
        return 1

    baseline_hrx_path = getattr(args, f"baseline_hrx_{kind}")
    baseline_vulkan_path = getattr(args, f"baseline_vulkan_{kind}")
    baseline_paths = (baseline_hrx_path, baseline_vulkan_path)
    if all(path is not None for path in baseline_paths):
        try:
            main_hrx = load_json(baseline_hrx_path)
            main_vulkan = load_json(baseline_vulkan_path)
            main_comparisons = format_main_comparisons(
                hrx,
                vulkan,
                main_hrx,
                main_vulkan,
                args.baseline_run_url,
            )
        except REPORT_INPUT_ERRORS as exc:
            print(
                f"Could not compare with main {kind}: {exc}",
                file=sys.stderr,
            )
            report = f"{report}\n\n{baseline_unavailable_message}"
        else:
            report = f"{report}\n\n{main_comparisons}"
    else:
        if any(path is not None for path in baseline_paths):
            print(
                f"Could not compare with main {kind}: both baseline {kind} "
                "paths are required",
                file=sys.stderr,
            )
        report = f"{report}\n\n{baseline_unavailable_message}"

    print(report)
    return 0
