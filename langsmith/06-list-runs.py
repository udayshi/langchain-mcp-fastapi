"""List recent root traces and their run IDs for LangSmith feedback."""

from __future__ import annotations

import os

from langsmith import Client


def _project_name() -> str:
    """Return the configured LangSmith project name or fail clearly."""
    project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    if not project_name:
        raise RuntimeError("Set LANGSMITH_PROJECT before listing runs.")
    return project_name


def main() -> None:
    """Print the ten most recent root runs in the configured project."""
    client = Client()
    project_name = _project_name()
    runs = list(client.list_runs(project_name=project_name, is_root=True, limit=10))
    if not runs:
        print(f"No root traces found in project {project_name!r}.")
        return
    for run in runs:
        run_url = client.get_run_url(run=run, project_name=project_name)
        print(f"run_id={run.id}\nname={run.name}\nurl={run_url}\n")


if __name__ == "__main__":
    main()
