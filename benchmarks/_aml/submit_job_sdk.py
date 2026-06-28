"""Submit an AML command job to WebXT Shopping Singularity via the Python SDK.

Fallback / alternative to ``az ml job create``. Defaults to the M0 A100 smoke test.

    python benchmarks/_aml/submit_job_sdk.py                       # submit hello-a100-1gpu.yml
    python benchmarks/_aml/submit_job_sdk.py <other-job>.yml --stream
"""
from __future__ import annotations

import argparse
from pathlib import Path

from azure.ai.ml import MLClient, load_job
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "<SUBSCRIPTION_ID>"
RESOURCE_GROUP = "<RESOURCE_GROUP>"
WORKSPACE_NAME = "<WORKSPACE>"

DEFAULT_JOB = Path(__file__).with_name("hello-a100-1gpu.yml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_file", nargs="?", default=str(DEFAULT_JOB), help="path to a command-job YAML")
    ap.add_argument("--stream", action="store_true", help="stream job logs after submit")
    args = ap.parse_args()

    client = MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    job = load_job(args.job_file)
    created = client.jobs.create_or_update(job)

    print(f"Job name:   {created.name}")
    print(f"Status:     {created.status}")
    print(f"Studio URL: {created.studio_url}")

    if args.stream:
        client.jobs.stream(created.name)


if __name__ == "__main__":
    main()
