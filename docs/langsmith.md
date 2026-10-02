# LangSmith walkthrough: basic to advanced with Python and uv

LangSmith is an observability and evaluation platform for LLM applications. It records the steps that produced an
answer—prompts, model calls, tool calls, inputs, outputs, latency, errors, and evaluation scores—so you can inspect
what happened instead of guessing from a final response.

This guide is for readers new to LangSmith. It uses Python and `uv`; examples work with ordinary Python functions and
then show how LangChain applications are traced. LangSmith does not replace LangChain or LangGraph:

```text
your Python / LangChain / LangGraph application
                    ↓ traces and evaluation data
                LangSmith project
                    ↓
debugging, monitoring, feedback, datasets, and experiments
```

## 1. Why use LangSmith?

An LLM application can fail in ways an ordinary stack trace does not explain:

- The final answer is wrong, but you need to know whether retrieval, a prompt, a tool, or the model caused it.
- A prompt or model change makes a previously good workflow worse.
- An agent calls an unexpected tool or passes malformed arguments.
- Production latency or token use suddenly increases.
- You need repeatable evidence that a change improves quality before deployment.

LangSmith helps by organizing execution records into **projects**, displaying nested operations as **traces**, accepting
human or automated **feedback**, and running versions of an application against curated **datasets**. It is most useful
when traces, input handling, and evaluation are designed before an incident occurs.

## 2. Create a project and install the SDK

Create an account and API key in the [LangSmith UI](https://smith.langchain.com/). Then create an independent Python
project:

```bash
mkdir langsmith-playground
cd langsmith-playground
uv init --package langsmith-playground
uv add langsmith
```

The runnable application examples in this guide use local Ollama. Install the LangChain integration and download the
model:

```bash
uv add langchain langchain-ollama
ollama pull llama3.2:3b
```

Start Ollama before running an example that invokes a model:

```bash
ollama serve
```

### Choose an Ollama model for the job

All model examples in this guide use Ollama. `llama3.2:3b` remains a useful fast baseline for tracing and inexpensive
smoke tests. Use LangSmith experiments to compare it with the models available on your machine instead of
assuming a larger model is always better.

| Model | Good role in this guide | Trade-off |
| --- | --- | --- |
| `llama3.2:3b` | Fast local baseline and frequent regression checks | Lower capacity for difficult reasoning or tool selection |
| `gemma3:4b` | Default candidate for answer-quality comparisons | More local memory and latency than the baseline |
| `nimble:9b` | Candidate for harder classification or response experiments | Higher memory and evaluation time |
| `gpt-oss:20b` | Candidate judge or high-quality comparison model when hardware allows | Largest local memory and latency cost |

Pull a candidate before an experiment:

```bash
ollama pull gemma3:4b
ollama pull nimble:9b
ollama pull gpt-oss:20b
```

To switch an example, replace only its `model` value. For a useful evaluation, keep the prompt, dataset, and evaluator fixed; change only the model name. Record that name
in the LangSmith project, experiment name, or trace metadata so results can be compared honestly.

Set credentials in your shell, never in committed source code:

```bash
# Compatible with the project's installed langsmith==0.14.1 and newer SDKs.
export LANGSMITH_TRACING_V2=true
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="langsmith-playground"
```

`LANGSMITH_PROJECT` is the project name that groups related traces in the UI. Use separate names for development,
staging, and production, for example `support-dev` and `support-production`. Add secrets to your deployment's secret
manager or CI secret store; do not place them in a `.env` file that will be committed.

### Verify tracing before calling Ollama

The version locked by this repository is `langsmith==0.14.1`. In that version, `LANGSMITH_TRACING=true` alone does
**not** enable the `@traceable` decorator: it requires the `*_TRACING_V2` setting above. Run this diagnostic in the
same terminal that will run the examples:

```bash
uv run python langsmith/00-check-config.py
```

It must print `Tracing enabled: true`, the expected project, and `API key: configured`. If it does not, do not run the
trace example yet. Common causes of a successful Python program but an empty LangSmith console are:

- The exports were made in a different terminal, IDE run configuration, Docker container, or shell session.
- The project filter in the LangSmith UI is not the exact `LANGSMITH_PROJECT` value.
- The API key belongs to a different LangSmith workspace than the console currently open.
- Network, proxy, DNS, or firewall rules prevent access to `https://api.smith.langchain.com`.

Newer LangSmith documentation may use `LANGSMITH_TRACING=true`; retain `LANGSMITH_TRACING_V2=true` here until this
project's lockfile is upgraded and verified.

### Runnable example index

Every Python example below exists in `langsmith/`; no documentation-only Python snippets are used.

| Program | Purpose | Command |
| --- | --- | --- |
| [`01-trace.py`](../langsmith/01-trace.py) | Trace one Python function and Ollama call | `uv run python langsmith/01-trace.py` |
| [`02-trace.py`](../langsmith/02-trace.py) | Automatically trace a LangChain chain | `uv run python langsmith/02-trace.py` |
| [`03-create-dataset.py`](../langsmith/03-create-dataset.py) | Create the evaluation dataset | `uv run python langsmith/03-create-dataset.py` |
| [`04-evaluate.py`](../langsmith/04-evaluate.py) | Run the dataset evaluation | `uv run python langsmith/04-evaluate.py` |
| [`05-named-trace.py`](../langsmith/05-named-trace.py) | Add a useful trace name and metadata | `uv run python langsmith/05-named-trace.py` |
| [`06-list-runs.py`](../langsmith/06-list-runs.py) | Print recent trace IDs and trace URLs | `uv run python langsmith/06-list-runs.py` |
| [`07-feedback.py`](../langsmith/07-feedback.py) | Submit feedback for a trace ID | `uv run python langsmith/07-feedback.py "<run-id>" --helpful` |

## 3. Trace your first Python function

Tracing does not require LangChain. Decorate an important boundary with `@traceable`:

```python
"""Create one simple LangSmith trace."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langsmith import Client, traceable


CLIENT = Client()


@traceable(name="ollama-greeting", run_type="chain", client=CLIENT)
def answer_greeting(name: str) -> str:
    """Ask local Ollama to create a short greeting."""
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("name must not be blank")
    response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(
        [
            SystemMessage(content="Return a warm greeting in one short sentence."),
            HumanMessage(content=f"Greet {cleaned_name}."),
        ]
    )
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("model must return non-empty text")
    return response.content


def main() -> None:
    """Run the traced function."""
    try:
        print(answer_greeting("Uday Shiwakoti"))
    finally:
        CLIENT.flush()


if __name__ == "__main__":
    main()
```

The same complete program is committed as [`langsmith/01-trace.py`](../langsmith/01-trace.py). It explicitly flushes
the SDK upload before the short-lived process exits. Run it from the repository root:

```bash
uv run python langsmith/01-trace.py
```

With the three `LANGSMITH_*` variables set, a trace appears in the `langsmith-playground` project. The decorator
creates a run named `ollama-greeting`; its input, output, duration, nested model call, and any error are visible in
the trace detail.

### Flow

```text
answer_greeting("Uday Shiwakoti")
    → @traceable records input
    → function validates name and calls local Ollama
    → @traceable records output or error
    → LangSmith project displays the run
```

## 4. Trace a LangChain model call

When LangSmith tracing is enabled, LangChain automatically traces compatible chains and model calls. You normally do
not need to decorate every LangChain runnable yourself.

```python
"""Trace a local Ollama LangChain call with LangSmith."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def main() -> None:
    """Invoke a traced LangChain chain."""
    prompt = ChatPromptTemplate.from_template("Answer in one sentence: {question}")
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"question": "What is a Python tuple?"}))


if __name__ == "__main__":
    main()
```

The complete runnable program is [`langsmith/02-trace.py`](../langsmith/02-trace.py). Run it after starting Ollama:

```bash
ollama serve
uv run python langsmith/02-trace.py
```

The trace should show a parent chain run and nested prompt, model, and parser runs. This nesting is the practical
reason to use tracing: if the answer is poor, inspect the exact prompt rendered for the model, the model response, and
the time spent in each step.

## 5. Name and organize traces deliberately

Useful project names and runnable names make filtering possible. Name a trace after a business action, not an
implementation accident:

```python
"""Create a deliberately named LangSmith trace with safe metadata."""

from __future__ import annotations

from langsmith import traceable


@traceable(
    name="support-answer",
    run_type="chain",
    metadata={"environment": "development", "release": "local-example"},
)
def answer_support_question(question: str) -> str:
    """Return a small, traceable support-answer example."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question must not be blank")
    return f"We received your question: {cleaned_question}"


def main() -> None:
    """Run the named trace."""
    print(answer_support_question("Where can I update my email address?"))


if __name__ == "__main__":
    main()
```

Run the committed version with:

```bash
uv run python langsmith/05-named-trace.py
```

Good names include `support-answer`, `retrieve-policy`, `payment-tool`, and `human-approval`. Avoid names such as
`run_1` or `test`. In the UI, filter traces by project, name, execution time, errors, and metadata to investigate a
specific workflow or deployment version.

Use metadata for stable operational facts such as `environment`, `release`, `model_name`, or `tenant_plan`. Do not use
metadata for API keys, raw authorization headers, full user profiles, or other secrets. Keep high-cardinality values
such as request IDs useful but bounded, and make sure they do not expose personal data.

## 6. Debug traces systematically

When a user reports a bad answer, inspect the trace from the outside in:

1. Find the parent run in the correct project and time range.
2. Check whether it failed, timed out, or completed with an unexpected output.
3. Open child runs in order: routing, retrieval, prompt creation, model call, tool call, and output parsing.
4. Compare each child run's actual input and output with the intended contract.
5. Record the failure as feedback or turn it into a dataset example before changing the application.

For an agent, verify the tool name, arguments, return value, and subsequent model message. For RAG, inspect the
retrieved documents before rewriting the answer prompt. For LangGraph, use the graph node traces and state updates to
identify the node that selected an incorrect branch.

To get a `run_id` without manually finding it in the console, run the complete helper program:

```bash
uv run python langsmith/06-list-runs.py
```

It prints the ten newest root traces in `LANGSMITH_PROJECT`, including lines such as `run_id=...` and a direct
LangSmith URL. Copy only the UUID after `run_id=` into the feedback command below.

The source below is exactly [`langsmith/06-list-runs.py`](../langsmith/06-list-runs.py):

```python
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
```

## 7. Add feedback to traces

Feedback links a human or automated judgement to a run. A simple thumbs-up/down score can identify traces worth
reviewing; a numeric score or categorical label supports dashboards and experiments.

The complete CLI is [`langsmith/07-feedback.py`](../langsmith/07-feedback.py). It validates the run ID and accepts
either a positive score or a negative score. First list the IDs, then copy one of them:

```bash
uv run python langsmith/06-list-runs.py
uv run python langsmith/07-feedback.py "<run-id-from-trace-details>" --helpful
uv run python langsmith/07-feedback.py "<run-id-from-trace-details>"
```

You can also copy the UUID from a trace URL in the LangSmith UI; it is the value after `/runs/`. Keep feedback keys stable—for example `helpfulness`,
`groundedness`, or `tool_success`—so scores can be compared over time. Do not treat a small sample of voluntary user
ratings as a complete quality measure; it can be biased toward unusually positive or negative experiences.

The source below is exactly [`langsmith/07-feedback.py`](../langsmith/07-feedback.py):

```python
"""Record a helpfulness score for a LangSmith run ID."""

from __future__ import annotations

import argparse
from uuid import UUID

from langsmith import Client


def record_feedback(run_id: UUID, was_helpful: bool) -> None:
    """Store a binary helpfulness score for one trace run."""
    Client().create_feedback(
        run_id=run_id,
        key="helpfulness",
        score=1 if was_helpful else 0,
        comment="User feedback",
    )


def main() -> None:
    """Parse command-line feedback and submit it."""
    parser = argparse.ArgumentParser(description="Submit LangSmith helpfulness feedback.")
    parser.add_argument("run_id", type=UUID, help="Run ID shown in LangSmith trace details.")
    parser.add_argument("--helpful", action="store_true", help="Record a helpful score; omit for not helpful.")
    arguments = parser.parse_args()
    record_feedback(arguments.run_id, arguments.helpful)
    print("Feedback recorded.")


if __name__ == "__main__":
    main()
```

## 8. Build a dataset from known examples

A dataset is a named collection of inputs and expected outputs. It lets you run the same representative cases against
multiple prompt, model, or code versions.

The source below is exactly [`langsmith/03-create-dataset.py`](../langsmith/03-create-dataset.py):

```python
"""Create the small LangSmith dataset used by the evaluation example."""

from __future__ import annotations

from langsmith import Client


DATASET_NAME = "tuple-answers"


def main() -> None:
    """Create the dataset and its two reference examples once."""
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset {DATASET_NAME!r} already exists; no examples were added.")
        return
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Basic Python tuple questions",
    )
    client.create_example(
        inputs={"question": "What is a Python tuple?"},
        outputs={"required_term": "immutable"},
        dataset_id=dataset.id,
    )
    client.create_example(
        inputs={"question": "Can a tuple be changed after creation?"},
        outputs={"required_term": "immutable"},
        dataset_id=dataset.id,
    )
    print(f"Created dataset {DATASET_NAME!r} with two examples.")


if __name__ == "__main__":
    main()
```

The complete, repeat-safe program is [`langsmith/03-create-dataset.py`](../langsmith/03-create-dataset.py). Run it:

```bash
uv run python langsmith/03-create-dataset.py
```

Choose examples that represent real tasks, known failures, edge cases, invalid input, and safety-critical behaviour.
Avoid adding secrets or sensitive customer content unless your data policy and LangSmith configuration explicitly allow
it. Treat dataset changes as application changes: review them and keep expected outputs trustworthy.

## 9. Run an offline evaluation

Offline evaluation runs an application function against each dataset example and scores its outputs. Start with a
deterministic evaluator before using an LLM-as-a-judge evaluator.

The source below is exactly [`langsmith/04-evaluate.py`](../langsmith/04-evaluate.py):

```python
"""Evaluate a local Ollama answer function against a LangSmith dataset."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langsmith.evaluation import evaluate


DATASET_NAME = "tuple-answers"


def target(inputs: dict[str, str]) -> dict[str, str]:
    """Ask local Ollama for the candidate answer to one dataset input."""
    response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(
        [
            SystemMessage(content="Answer in one sentence and include immutable when relevant."),
            HumanMessage(content=inputs["question"]),
        ]
    )
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("model must return non-empty text")
    return {"answer": response.content}


def contains_required_term(
    inputs: dict[str, str], outputs: dict[str, str], reference_outputs: dict[str, str]
) -> dict[str, float | str]:
    """Score whether the answer contains the required reference term."""
    del inputs
    required_term = reference_outputs["required_term"].casefold()
    score = float(required_term in outputs["answer"].casefold())
    return {"key": "required_term", "score": score}


def main() -> None:
    """Create an evaluation experiment from the named dataset."""
    evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[contains_required_term],
        experiment_prefix="tuple-baseline",
    )


if __name__ == "__main__":
    main()
```

The complete program is [`langsmith/04-evaluate.py`](../langsmith/04-evaluate.py). It evaluates the dataset created
in section 8, so run `03-create-dataset.py` first.

```bash
uv run python langsmith/04-evaluate.py
```

The evaluation creates an experiment in LangSmith. Compare experiments when changing a model, prompt, retriever,
tool policy, or routing rule. Exact matching is appropriate only for deterministic outputs. For Ollama-generated
natural-language answers, use carefully designed semantic or LLM-as-a-judge evaluation alongside deterministic checks
such as schema validity, required citations, required terms, and tool-call constraints.

## 10. Production practices

Before tracing production traffic, decide what may leave your application boundary. Prompts and model outputs can
contain personal data, credentials, internal documents, and payment information.

- Send the minimum trace data needed for debugging and evaluation.
- Redact or avoid logging secrets, authorization headers, tokens, and sensitive fields before calls are traced.
- Use distinct LangSmith projects and API keys for development, staging, and production.
- Attach a release or prompt version to traces so regressions can be tied to a change.
- Sample high-volume, low-risk traffic if tracing every request is unnecessary or too costly.
- Add timeouts, bounded retries, and explicit error handling in the application; tracing observes failures but does not
  fix them.
- Convert validated production failures into sanitized dataset examples, then run them in CI before future releases.
- Add online evaluators gradually and review false positives before making automated decisions from their scores.

## 11. A practical workflow

```text
1. Trace a local development request.
2. Inspect the trace and fix obvious prompt, retrieval, tool, or parsing issues.
3. Save representative failures as sanitized dataset examples.
4. Run an offline evaluation before changing models or prompts.
5. Deploy with a separate production project and safe trace data.
6. Use production feedback and anomalies to improve the dataset.
```

LangSmith is most valuable as this feedback loop, not just as a trace viewer. Start with one meaningful traced boundary
and a tiny dataset, then expand only when the team has a concrete debugging or quality question to answer.

## 12. Turn LangSmith into a serious test system

No dataset, tracing system, or evaluator can make an LLM application **100% correct**. Models can be non-deterministic,
the world changes, retrieval data becomes stale, and test data can miss real user behaviour. The goal is measurable,
repeatable confidence—not a false guarantee.

Use several layers together:

| Layer | What it checks | Where it runs |
| --- | --- | --- |
| Unit tests | Input validation, parsing, routing, tool arguments, access checks | Local machine and CI |
| Dataset evaluation | Known answers, edge cases, regressions, prompt/model comparisons | Before deployment |
| Trace review | Unexpected prompts, retrieval, tool calls, errors, and latency | Development and production |
| Human feedback | Whether a real user found an answer useful | Production |
| Online evaluation | Quality trends, anomalies, and selected policy checks | Production, sampled |

### Build datasets that test the application, not just the model

For a LangChain application, include examples for each important boundary:

- Valid questions with trusted reference answers.
- Questions that must be refused or escalated.
- Requests that should call a specific tool, and requests that should call no tool.
- RAG questions with answerable, unanswerable, stale, and conflicting source documents.
- Structured-output cases with valid values, missing fields, out-of-range values, and adversarial text.
- Past production failures after removing secrets and personal information.

Give every example a purpose in its metadata, such as `"category": "tool-routing"` or `"severity": "high"`. When a
bug is fixed, add a regression example before closing the incident. That makes the dataset a durable record of expected
behaviour rather than a one-time demo.

### Use deterministic checks first

An LLM-as-a-judge can assess tone or semantic similarity, but it should not replace checks your own code can prove.
Then use a LangSmith dataset evaluation to test the end-to-end workflow, including prompt rendering and model/tool
behaviour. A release should fail when a required deterministic check fails; use evaluation-score thresholds as a
review gate until the dataset is mature and representative.

### Compare changes instead of inspecting one run in isolation

For every material change—model, prompt, retriever, tool description, routing rule, or MCP server—run the same
dataset before and after the change. Compare:

- Per-example failures, especially high-severity cases.
- Aggregate quality scores and evaluator disagreements.
- Tool-call accuracy and unwanted tool calls.
- Latency, token usage, and error rate.

Promote a change only when it improves or preserves the required cases. A higher average score does not justify
breaking a safety-critical example.

## 13. One LangSmith operating loop

```text
production trace or user feedback
        ↓
sanitize and add a representative dataset item
        ↓
run baseline and candidate application versions
        ↓
score deterministic requirements and quality metrics
        ↓
review regressions, latency, cost, and safety failures
        ↓
deploy only when the required cases pass
```

This loop shows the practical power of observability platforms: traces explain *why* a result happened, while datasets
and experiments show whether a proposed change makes the application better or worse. Use the smallest process that
answers your current quality question, then deepen coverage as real failures reveal new risks.
