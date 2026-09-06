#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Find the newest main CI run that uploaded the benchmark artifact.

The release benchmark job uploads its results artifact even when benchmarks
fail (the upload step runs ``if: always()``), and the report scripts compare
whatever models both sides contain. So a red ``main`` run is still a usable
baseline; what disqualifies a run is only having no artifact at all — it
failed before the upload, or the artifact has expired. This script therefore
walks the most recent completed runs on the branch, newest first, and emits
the first one whose artifact exists, regardless of the run's conclusion.

Outputs ``run_id``, ``run_url``, and ``artifact_id`` as step outputs when a
run is found; prints a note and sets nothing when none of the recent runs
qualifies (the workflow then skips the baseline download and the reports
degrade to their "no usable main artifact" line). Exit status is 1 only when
the GitHub API cannot be queried.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode

from github_actions_api import GitHubAPIError, gha_send_request, gha_set_output

JsonRequest = Callable[[str], object]
RECENT_RUNS_TO_CHECK = 20


@dataclass(frozen=True)
class MainBenchmarkArtifact:
    run_id: int
    run_url: str
    artifact_id: int


def discover_main_benchmark(
    repository: str,
    workflow: str,
    branch: str,
    artifact_name: str,
    *,
    send_request: JsonRequest = gha_send_request,
) -> MainBenchmarkArtifact | None:
    """Return the artifact from the newest completed run that uploaded one."""
    runs_query = urlencode(
        {
            "branch": branch,
            "status": "completed",
            "per_page": RECENT_RUNS_TO_CHECK,
        }
    )
    runs_response = send_request(
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{workflow}/runs?{runs_query}"
    )
    artifacts_query = urlencode({"name": artifact_name, "per_page": 100})
    for run in runs_response["workflow_runs"]:
        artifacts_response = send_request(
            f"https://api.github.com/repos/{repository}/actions/runs/"
            f"{run['id']}/artifacts?{artifacts_query}"
        )
        artifact = next(
            (
                candidate
                for candidate in artifacts_response["artifacts"]
                if candidate["name"] == artifact_name
                and not candidate["expired"]
            ),
            None,
        )
        if artifact is not None:
            return MainBenchmarkArtifact(
                run_id=run["id"],
                run_url=run["html_url"],
                artifact_id=artifact["id"],
            )
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)

    try:
        artifact = discover_main_benchmark(
            args.repository,
            args.workflow,
            args.branch,
            args.artifact,
        )
    except (GitHubAPIError, KeyError, TypeError) as exc:
        print(f"Could not discover main benchmark artifact: {exc}", file=sys.stderr)
        return 1

    if artifact is None:
        print(
            f"None of the {RECENT_RUNS_TO_CHECK} most recent completed main CI "
            "runs uploaded a usable benchmark artifact."
        )
        return 0

    gha_set_output(
        {
            "run_id": str(artifact.run_id),
            "run_url": artifact.run_url,
            "artifact_id": str(artifact.artifact_id),
        }
    )
    print(
        f"Using benchmark artifact {artifact.artifact_id} from "
        f"{artifact.run_url}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
