# LangGraph walkthrough: basic to advanced

Each practical section maps directly to one runnable file in [`langgraph/`](../langgraph). The examples use Python,
local Ollama, and `llama3.2:3b` where an LLM is required; no example depends on an earlier example. LangGraph adds
explicit state, routing, persistence, streaming, and human approval around ordinary Python and LangChain model calls.

New to the ecosystem? **LangChain** provides the building blocks for model prompts, messages, tools, and model calls.
**LangGraph** decides the order in which those building blocks run and retains their shared state. Start with
[the LangChain walkthrough](langchain.md) if terms such as `HumanMessage`, tools, and models are unfamiliar.

## 0. Setup

Run Ollama in one terminal:

```bash
ollama serve
```

In another terminal, create a project and install the packages used below:

```bash
mkdir langgraph-playground
cd langgraph-playground
uv init --package langgraph-playground
uv add langgraph langchain-mcp-adapters langchain-ollama
uv add --dev mypy
ollama pull llama3.2:3b
mkdir langgraph
```

Use strict type checking in `pyproject.toml`:

```toml
[tool.mypy]
strict = true
files = ["langgraph"]
```

Run every example as `uv run python langgraph/<file>.py`.

## How to read the examples

Every graph follows the same small lifecycle:

```text
input state → START → node(s) → optional route or loop → END → final state
```

- **State** is a typed dictionary containing the data the workflow needs.
- A **node** is a Python function that reads state and returns only the fields it wants to update.
- An **edge** is an allowed transition between nodes.
- `START` and `END` are LangGraph's built-in entry and exit markers.
- `compile()` validates the declared graph and turns its builder into a runnable graph.
- `invoke()` runs that graph once and returns its final state.

The explanations below use the source line numbers shown in the matching `langgraph/*.py` file. Blank lines are omitted
because they affect readability, not execution.

## 1. Basic: a stateful graph

A graph has typed shared state, nodes that return state updates, and edges that define execution order. Save as
`langgraph/01_basic_graph.py`:

```python
"""Run a minimal LangGraph workflow."""

from __future__ import annotations

# Import TypedDict so the graph state has named, type-checked fields.
from typing import TypedDict

# Import the graph builder and the built-in entry and exit markers.
from langgraph.graph import END, START, StateGraph


# Describe every value that can move through this workflow.
class GreetingState(TypedDict):
    """State carried through the greeting workflow."""

    name: str
    greeting: str


def create_greeting(state: GreetingState) -> dict[str, str]:
    """Create a greeting from a non-empty name."""
    # Read and normalise the input value without mutating shared graph state.
    name = state["name"].strip()
    # Stop immediately when the graph receives invalid input.
    if not name:
        raise ValueError("name must not be blank")
    # Return only the state field this node changes; LangGraph merges this update.
    return {"greeting": f"Hello, {name}!"}


def main() -> None:
    """Run the compiled workflow and print its final greeting."""
    # Create a builder that uses GreetingState as its shared state contract.
    builder = StateGraph(GreetingState)
    # Register the Python function under the node name used by graph edges.
    builder.add_node("create_greeting", create_greeting)
    # Define the flow: begin at START, run the node, then finish at END.
    builder.add_edge(START, "create_greeting")
    builder.add_edge("create_greeting", END)
    # Validate and compile the graph declaration into a runnable object.
    graph = builder.compile()
    # Supply initial state and execute the only path through the graph.
    result = graph.invoke({"name": "Uday Shiwakoti", "greeting": "Hi world"})

    # Read the node's merged update from the final state.
    print(result["greeting"])


# Run main only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
```

```bash
uv run python langgraph/01_basic_graph.py
```

### Code flow

```text
{"name": "Uday Shiwakoti", "greeting": "Hi world"}
    → START → create_greeting → {"greeting": "Hello, Uday Shiwakoti!"}
    → END → print the final greeting
```

The `#` comments in the source now explain the next execution step in reading order, from imports through graph
execution. They are kept directly beside the code so source and guide cannot drift apart.

Nodes should return only fields they change. LangGraph merges those updates into the state; a node should not mutate
the input state in place.

## 2. Conditional routing: choose the next node

Conditional edges let ordinary Python logic choose the next step. Save as `langgraph/02_conditional_routing.py`:

```python
"""Route a support request by its urgency."""

from __future__ import annotations

# Literal limits routing values; TypedDict describes the workflow state.
from typing import Literal, TypedDict

# Import the graph builder and its start and end markers.
from langgraph.graph import END, START, StateGraph


# Define the input, classification result, and routing result for one ticket.
class TicketState(TypedDict):
    """State used to triage one support ticket."""

    message: str
    priority: Literal["normal", "urgent"]
    queue: str


def classify(state: TicketState) -> dict[str, Literal["normal", "urgent"]]:
    """Classify an urgent ticket using an intentionally simple local rule."""
    # Apply a deterministic rule so routing is easy to inspect and test.
    priority: Literal["normal", "urgent"] = "urgent" if "outage" in state["message"].lower() else "normal"
    # Return the priority update for LangGraph to merge into state.
    return {"priority": priority}


def choose_queue(state: TicketState) -> Literal["urgent_queue", "normal_queue"]:
    """Select the queue node for the classified ticket."""
    # Return the registered node name that LangGraph should run next.
    return "urgent_queue" if state["priority"] == "urgent" else "normal_queue"


def assign_urgent(_: TicketState) -> dict[str, str]:
    """Assign an urgent ticket to the incident queue."""
    # This branch has a fixed state update, so it does not need to read state.
    return {"queue": "incident-response"}


def assign_normal(_: TicketState) -> dict[str, str]:
    """Assign a normal ticket to the standard queue."""
    # This branch also returns only the field it changes.
    return {"queue": "support"}


def main() -> None:
    """Build and run the routing workflow."""
    # Create a graph whose state must conform to TicketState.
    builder = StateGraph(TicketState)
    # Register the classifier and both possible destination nodes.
    builder.add_node("classify", classify)
    builder.add_node("urgent_queue", assign_urgent)
    builder.add_node("normal_queue", assign_normal)
    # Always classify first, then let choose_queue select one branch.
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", choose_queue)
    # Both queue nodes are terminal paths.
    builder.add_edge("urgent_queue", END)
    builder.add_edge("normal_queue", END)

    # Compile the declared nodes and edges before executing them.
    graph = builder.compile()
    # Invoke with complete initial state; classify will overwrite priority.
    result = graph.invoke({"message": "There is an outage in the payment service.", "priority": "normal", "queue": ""})
    # Print the queue written by the branch that ran.
    print(result["queue"])


# Run this demonstration only when the file is executed directly.
if __name__ == "__main__":
    main()
```

```bash
uv run python langgraph/02_conditional_routing.py
```

Keep routing functions deterministic and small. Validate data at the graph boundary before it influences an external
action.

### Code flow

```text
START → classify
          ├─ priority == "urgent" → urgent_queue → END
          └─ priority == "normal" → normal_queue → END
```

Read the `#` comments from top to bottom: they first define the state, then classify it, choose a node name, and run
the selected queue node.

## 3. LLM node: make a controlled model call

LangGraph does not replace your model integration; a node can call any LangChain runnable. Save as
`langgraph/03_llm_node.py`:

```python
"""Use local Ollama from a LangGraph node."""

from __future__ import annotations

# TypedDict declares the values passed between graph nodes.
from typing import TypedDict

# Import LangChain message types for a structured model conversation.
from langchain_core.messages import HumanMessage, SystemMessage
# Import the local Ollama chat-model integration.
from langchain_ollama import ChatOllama
# Import the graph builder and execution markers.
from langgraph.graph import END, START, StateGraph


# Keep the user's question and the validated model answer in one state object.
class AnswerState(TypedDict):
    """State for one question and answer."""

    question: str
    answer: str


def answer_question(state: AnswerState) -> dict[str, str]:
    """Ask Ollama for a concise text answer."""
    # Normalise external input before it reaches the model.
    question = state["question"].strip()
    # Avoid a needless model call for a blank question.
    if not question:
        raise ValueError("question must not be blank")
    # Configure a local model; zero temperature makes replies less variable.
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    # Send the application instruction before the user's question.
    response = model.invoke(
        [
            SystemMessage(content="Answer accurately in at most two sentences."),
            HumanMessage(content=question),
        ]
    )
    # Accept only non-empty text before putting model output in graph state.
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("model must return non-empty text")
    # Return the state update rather than changing state in place.
    return {"answer": response.content}


def main() -> None:
    """Run the question-answering graph."""
    # Create a graph whose nodes exchange AnswerState values.
    builder = StateGraph(AnswerState)
    # Register the node and define its complete start-to-end path.
    builder.add_node("answer_question", answer_question)
    builder.add_edge(START, "answer_question")
    builder.add_edge("answer_question", END)
    # Compile and invoke the graph with an empty answer for the node to fill.
    result = builder.compile().invoke({"question": "What is a Python tuple?", "answer": ""})
    # Print the validated answer from final graph state.
    print(result["answer"])


# Run main only for direct execution.
if __name__ == "__main__":
    main()
```

```bash
uv run python langgraph/03_llm_node.py
```

Model output is external input. Validate it before using it as a route, a tool argument, or a write to another system.

### Code flow

```text
question → START → answer_question
                     → validate question
                     → ChatOllama.invoke(messages)
                     → validate text response → END → print answer
```

The comments mark three boundaries a newcomer should notice: validating user input, calling the LangChain model, and
validating the model response before returning it to LangGraph state.

## 4. Tool loop: let the model request bounded actions

`ToolNode` runs tool calls from the latest AI message; `tools_condition` routes either to that node or to the end of
the graph. Save as `langgraph/04_tools.py`:

```python
"""Build a local calculator agent with an explicit tool loop."""

from __future__ import annotations

# Literal constrains the routing function to valid graph destinations.
from typing import Literal

# Import message types, the tool decorator, and the local model integration.
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
# MessagesState appends messages; ToolNode executes requested, registered tools.
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# Expose this narrow, typed function as the only tool the model may request.
@tool
def multiply(left: float, right: float) -> float:
    """Multiply two numbers."""
    # Keep tool logic deterministic and independently testable.
    return left * right


def call_model(state: MessagesState) -> dict[str, list[AnyMessage]]:
    """Ask a tool-capable local model for the next message."""
    # Bind only multiply so the model cannot select an unrestricted action.
    model = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools([multiply])
    # Give the model its rule followed by all conversation history.
    response = model.invoke(
        [
            SystemMessage(content="Use the multiply tool for arithmetic. Do not calculate yourself."),
            *state["messages"],
        ]
    )
    # Append the AI response, including any tool call, to MessagesState.
    return {"messages": [response]}


def route_after_model(state: MessagesState) -> Literal["tools", "__end__"]:
    """Route tool calls for execution, otherwise complete the graph."""
    # Route to ToolNode when the latest AI message requests a tool; otherwise end.
    return tools_condition(state)


def main() -> None:
    """Run the graph for one arithmetic question."""
    # Build a graph with the built-in message-accumulating state schema.
    builder = StateGraph(MessagesState)
    # Register the model node and a ToolNode restricted to multiply.
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode([multiply]))
    # Start with the model, then route based on whether it requested a tool.
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model)
    # Return tool results to the model so it can write the final user-facing answer.
    builder.add_edge("tools", "model")

    # Run the loop from one user message and capture the accumulated messages.
    result = builder.compile().invoke({"messages": [HumanMessage(content="What is 21 multiplied by 2?")]})
    # The last message is the model's final answer after any tool execution.
    print(result["messages"][-1].content)


# Run the demonstration only when this source file is executed directly.
if __name__ == "__main__":
    main()
```

```bash
uv run python langgraph/04_tools.py
```

Use tools with narrow schemas, authorization checks, timeouts, and audit logging. Treat tool arguments as untrusted,
even when the model generated them. A model must support tool calling for this example to work reliably.

### Code flow

```text
HumanMessage → START → model ── no tool call ──→ END
                            │
                            └── tool call → tools (multiply) → model → END
```

The loop is intentional: after a tool returns its result, the model receives that result and produces the final
natural-language response.

The comments make the safety boundary explicit: the model can request only `multiply`, `ToolNode` executes it, and
the graph returns the result to the model rather than letting a model directly perform arbitrary actions.

## 5. Persistence: checkpoint a conversation

A checkpointer saves graph state by `thread_id`, allowing later invocations to load prior messages. Save as
`langgraph/05_persistence.py`:

```python
"""Persist a short local conversation in memory."""

from __future__ import annotations

# Import the message type returned by the model node.
from langchain_core.messages import AnyMessage
# Use Ollama for the local conversational model.
from langchain_ollama import ChatOllama
# InMemorySaver checkpoints state by thread ID for this demonstration.
from langgraph.checkpoint.memory import InMemorySaver
# MessagesState preserves the growing conversation; START begins each run.
from langgraph.graph import START, MessagesState, StateGraph


def respond(state: MessagesState) -> dict[str, list[AnyMessage]]:
    """Return one model response for the accumulated conversation."""
    # Send all messages restored from the checkpoint plus the latest user message.
    response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(state["messages"])
    # MessagesState appends this one new model response to the history.
    return {"messages": [response]}


def main() -> None:
    """Invoke one persistent conversation twice."""
    # Build a graph that uses the built-in conversation state schema.
    builder = StateGraph(MessagesState)
    # Register the response node and make it the first node in every invocation.
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    # Attach a checkpointer so state is saved after each graph invocation.
    graph = builder.compile(checkpointer=InMemorySaver())
    # Reuse this identifier to reload the same conversation on the next invocation.
    config = {"configurable": {"thread_id": "demo-conversation"}}

    # Store the first user message and the model's response under this thread ID.
    graph.invoke({"messages": [("user", "My name is Uday Shiwakoti.")]}, config=config)
    # Resume that thread; LangGraph restores the earlier messages before responding.
    result = graph.invoke({"messages": [("user", "What is my name?")]}, config=config)
    # Print the latest model response from the persisted conversation.
    print(result["messages"][-1].content)


# Run this demo only when this source file is executed directly.
if __name__ == "__main__":
    main()
```

```bash
uv run python langgraph/05_persistence.py
```

`InMemorySaver` is for tests and demos only: it loses state when the process exits. Use a durable, access-controlled
checkpointer such as PostgreSQL for production, and choose non-guessable thread IDs that are authorized to the caller.

### Code flow

```text
first invoke, thread_id "demo-conversation" → respond → checkpoint messages
second invoke, same thread_id                → restore messages → respond → print answer
```

The comments distinguish the two invocations: the first creates a checkpoint, while the second uses the same
`thread_id` to restore that conversation before it calls the model.

## 6. Human approval: pause and resume work

`interrupt()` pauses a node and returns a value only after a subsequent call resumes it. A checkpointer and a stable
`thread_id` are required. Save as `langgraph/06_human_approval.py`:

```python
"""Require a person to approve an action before completing it."""

from __future__ import annotations

# Literal limits approval status; TypedDict describes the graph state.
from typing import Literal, TypedDict

# Import checkpointing, graph markers, and pause/resume primitives.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# Carry the action awaiting a decision and its final approval status.
class ApprovalState(TypedDict):
    """State for a single approval request."""

    action: str
    status: Literal["pending", "approved", "rejected"]


def request_approval(state: ApprovalState) -> dict[str, Literal["approved", "rejected"]]:
    """Pause until a person approves or rejects the action."""
    # Stop graph execution and return this context to the external caller.
    approved = interrupt({"question": "Approve this action?", "action": state["action"]})
    # When resumed, treat only literal True as approval for a safe default.
    return {"status": "approved" if approved is True else "rejected"}


def main() -> None:
    """Pause a run and resume it with an approval decision."""
    # Create a graph whose shared data follows ApprovalState.
    builder = StateGraph(ApprovalState)
    # Register the pausing node and define its start-to-end path.
    builder.add_node("request_approval", request_approval)
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", END)
    # A checkpointer is required to resume an interrupted graph run.
    graph = builder.compile(checkpointer=InMemorySaver())
    # Keep this stable ID so the resume command addresses the paused run.
    config = {"configurable": {"thread_id": "approval-demo"}}

    # Run until interrupt() pauses the node and returns an interrupt payload.
    paused = graph.invoke({"action": "Send the monthly report", "status": "pending"}, config=config)
    # Display the requested action so a person can make an informed decision.
    print(paused["__interrupt__"])
    # Read a local human response; a production UI would provide this decision.
    user_input=input(f"Should I approve {paused['__interrupt__']}? (y/n) ?")
    # Start with rejection, so unknown input cannot approve the action.
    do_action:bool=False
    # Approve only explicit affirmative answers.
    if user_input.lower() in ["y","yes","ok"]:
        do_action=True

    # Resume the same paused graph run with the verified Boolean decision.
    complete = graph.invoke(Command(resume=do_action), config=config)
    # Print the final state written by request_approval after it resumes.
    print(complete["status"])


# Run this interactive demo only when this file is executed directly.
if __name__ == "__main__":
    main()
```

```bash
uv run python langgraph/06_human_approval.py
```

LangGraph re-runs a node from its start after it is resumed. Put side effects after the interrupt or make them
idempotent. Never treat model output as sufficient approval for high-impact actions.

### Code flow

```text
initial state → START → request_approval → interrupt payload returned to caller
caller reads y/n → Command(resume=True or False) → request_approval continues → END
```

The comments call out the safe approval sequence: pause, show the action, default to rejection, accept only explicit
approval, and resume the same checkpointed thread with that Boolean decision.

## 7. Next steps and production checklist

The eight runnable programs above are the complete `langgraph/*.py` set in this repository. Add streaming only after
you understand the state updates produced by `invoke()`; it changes *how results are delivered*, not the graph's
nodes, state, or control flow.

For production workflows, validate external input before the graph begins; set model and tool timeouts; use bounded
retries and explicit error routes; checkpoint durable state; authorize every `thread_id`; redact secrets from logs and
traces; and require human approval for externally visible or irreversible actions. Add unit tests for each node and
route using deterministic fake dependencies before testing the compiled graph end to end.

## 8. Use MCP tools from LangGraph

The local MCP calculator server accepts Streamable HTTP requests at:

```text
http://127.0.0.1:3890/mcp
```

It requires the same `MCP_AUTH_TOKEN` from both the server process and every client request. Start the server in one
terminal from the repository root:

```bash
cd mcp
export MCP_AUTH_TOKEN="$(openssl rand -hex 32)"
uv run python server.py
```

Keep that terminal running. In a second terminal, export the **same token value** and run the LangGraph MCP example:

```bash
export MCP_AUTH_TOKEN="the-same-token-used-by-server.py"
uv run python langgraph/07_mcp_tools.py
```

Save the following complete program as `langgraph/07_mcp_tools.py`:

```python
"""Use an authenticated HTTP MCP tool from a LangGraph workflow."""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def load_token() -> str:
    """Load the bearer token required by the local MCP server."""
    # Read the secret only from the environment, never from source code.
    token = os.getenv("MCP_AUTH_TOKEN", "")
    # Reject missing or implausibly short tokens before attempting a connection.
    if len(token) < 32:
        raise RuntimeError("Set MCP_AUTH_TOKEN to the local MCP server token.")
    return token


async def main() -> None:
    """Ask a LangGraph tool loop to use the local MCP calculator."""
    # Configure the adapter to send the token with every request to the local endpoint.
    client = MultiServerMCPClient(
        {
            "local_calculator": {
                "transport": "http",
                "url": "http://127.0.0.1:3890/mcp",
                "headers": {"Authorization": f"Bearer {load_token()}"},
            }
        }
    )
    # Discover MCP tools and convert them into standard LangChain tools.
    tools = await client.get_tools()
    # Bind the discovered tools so the model can request only advertised MCP actions.
    model = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Ask the model whether it needs an MCP tool or can answer directly."""
        # Put the tool-use policy before the accumulated conversation history.
        response = model.invoke(
            [
                SystemMessage(content="Use the calculator tool for arithmetic. Do not calculate yourself."),
                *state["messages"],
            ]
        )
        # Append the model response so tools_condition can inspect its tool calls.
        return {"messages": [response]}

    def route_after_model(state: MessagesState) -> Literal["tools", "__end__"]:
        """Route model-requested MCP tools or complete the graph."""
        # Select ToolNode for a tool call; otherwise LangGraph ends the workflow.
        return tools_condition(state)

    # Build the same model → tools → model loop used for local LangChain tools.
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model)
    builder.add_edge("tools", "model")

    # Run the async graph and print the final model response after the MCP call.
    result = await builder.compile().ainvoke(
        {"messages": [HumanMessage(content="What is 21 multiplied by 2?")]}
    )
    print(result["messages"][-1].content)


# Run the asynchronous entry point only when this file is executed directly.
if __name__ == "__main__":
    asyncio.run(main())
```

### Code flow

```text
MCP_AUTH_TOKEN → MultiServerMCPClient → discover add/multiply tools
                                          ↓
HumanMessage → model → tools_condition → ToolNode (authenticated HTTP MCP call) → model → final answer
```

`MultiServerMCPClient` converts the server's MCP tool definitions into LangChain tools. `ToolNode` then executes an
MCP tool only when the model explicitly requests it. The `Authorization` header is generated from `MCP_AUTH_TOKEN` for
every connection; do not put the token in source code, prompts, or graph state. The model must support tool calling.

For the full independent MCP setup, see [the MCP guide](mcp.md).

## 9. Use multiple MCP servers from one graph

When each MCP server has a different responsibility and Bearer token, declare each server separately in one
`MultiServerMCPClient`. This example assumes:

| Server | Endpoint | Tool | Environment variable |
| --- | --- | --- | --- |
| Add server | `http://127.0.0.1:3890/mcp` | `add` | `MCP_AUTH_TOKEN_1` |
| Multiply server | `http://127.0.0.1:3891/mcp` | `multiply` | `MCP_AUTH_TOKEN_2` |

Start both servers first. In a third terminal, export both matching tokens and run:

```bash
export MCP_AUTH_TOKEN_1="the-token-for-port-3890"
export MCP_AUTH_TOKEN_2="the-token-for-port-3891"
uv run python langgraph/08_multiple_mcp_tools.py
```

Save the following complete program as `langgraph/08_multiple_mcp_tools.py`:

```python
"""Use tools from two authenticated HTTP MCP servers in one LangGraph workflow."""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def load_token(variable_name: str) -> str:
    """Load one sufficiently long MCP bearer token from the environment."""
    # Read each server secret independently so credentials never cross server boundaries.
    token = os.getenv(variable_name, "")
    # Fail before connecting when the required token was not configured.
    if len(token) < 32:
        raise RuntimeError(f"Set {variable_name} to the matching MCP server token.")
    return token


async def main() -> None:
    """Ask one LangGraph tool loop to use separate add and multiply MCP servers."""
    # Register each endpoint under a unique name and send its matching bearer token.
    client = MultiServerMCPClient(
        {
            "adder": {
                "transport": "http",
                "url": "http://127.0.0.1:3890/mcp",
                "headers": {"Authorization": f"Bearer {load_token('MCP_AUTH_TOKEN_1')}"},
            },
            "multiplier": {
                "transport": "http",
                "url": "http://127.0.0.1:3891/mcp",
                "headers": {"Authorization": f"Bearer {load_token('MCP_AUTH_TOKEN_2')}"},
            },
        },
        # Prefix tool names with their server names to prevent future name collisions.
        tool_name_prefix=True,
    )
    # Discover both servers' tools as one list of LangChain-compatible tools.
    tools = await client.get_tools()
    # Bind exactly those discovered tools to the local model.
    model = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Ask the model whether to call one of the discovered MCP tools."""
        # Require tool use for arithmetic rather than trusting model-calculated results.
        response = model.invoke(
            [
                SystemMessage(content="Use the available MCP tools for all arithmetic. Do not calculate yourself."),
                *state["messages"],
            ]
        )
        # Append the AI message so LangGraph can inspect any requested tool calls.
        return {"messages": [response]}

    def route_after_model(state: MessagesState) -> Literal["tools", "__end__"]:
        """Run requested MCP tools or finish when the model has answered."""
        # ToolNode runs only when the latest model message contains a tool request.
        return tools_condition(state)

    # Build the model → tools → model loop for both server tool sets.
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model)
    builder.add_edge("tools", "model")

    # Let the model select the correct server tools for both arithmetic operations.
    result = await builder.compile().ainvoke(
        {"messages": [HumanMessage(content="What are 2 plus 3 and 4 multiplied by 5?")]}
    )
    print(result["messages"][-1].content)


# Run the asynchronous graph only when this file is executed directly.
if __name__ == "__main__":
    asyncio.run(main())
```

### Code flow

```text
MCP_AUTH_TOKEN_1 → adder      → adder_add tool
MCP_AUTH_TOKEN_2 → multiplier → multiplier_multiply tool
                                  ↓
HumanMessage → model → ToolNode → correct authenticated MCP endpoint → model → final answer
```

`tool_name_prefix=True` turns the discovered names into `adder_add` and `multiplier_multiply`. This avoids ambiguity
if two servers later expose a tool with the same name. The model chooses the tool, but `ToolNode` can invoke only the
two tools discovered from the configured local endpoints. Keep the two tokens separate and never reuse a token for a
server that does not need it.

## 10. Route intent before constructing the MCP tool loop

For ten MCP servers, build the LangGraph tool loop with only the tools relevant to the user's validated intent. Do
not give all ten servers' tool descriptions to the model, and do not ask the model itself to decide which credentials
or server connections it may use.

```text
user input → trusted intent router → selected server names → tool discovery
           → bind selected tools + focused system prompt → model → ToolNode → selected server only
```

The router can begin as a small deterministic function. Later, it may use a separate classifier, but its final result
must still be validated against an allowlist before tool discovery.

```python
from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def select_servers(user_input: str) -> tuple[str, ...]:
    """Return only MCP server names permitted for one recognised intent."""
    request = user_input.casefold()
    if "add" in request or "plus" in request:
        return ("adder",)
    if "multiply" in request or "times" in request:
        return ("multiplier",)
    return ()


def system_prompt_for(server_names: Sequence[str]) -> str:
    """Return a scoped instruction matching the selected server subset."""
    if server_names == ("adder",):
        return "You handle addition. Use only the available tool for arithmetic."
    if server_names == ("multiplier",):
        return "You handle multiplication. Use only the available tool for arithmetic."
    if server_names == ("adder", "multiplier"):
        return "You handle addition and multiplication. Use only the available tools for arithmetic."
    raise ValueError("No unambiguous tool scope is available for this request.")


async def selected_tools(
    client: MultiServerMCPClient, user_input: str
) -> tuple[list[BaseTool], str]:
    """Discover only tools from servers approved by the request intent."""
    server_names = select_servers(user_input)
    if not server_names:
        raise ValueError("Ask the user whether they need addition or multiplication.")

    tools: list[BaseTool] = []
    for server_name in server_names:
        tools.extend(await client.get_tools(server_name=server_name))
    return tools, system_prompt_for(server_names)
```

Use the returned values before adding `ToolNode` to the graph:

```python
question = "What is 21 times 2?"
tools, system_prompt = await selected_tools(client, question)
model = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools)

# call_model uses system_prompt and model; ToolNode receives the same selected tools.
builder.add_node("tools", ToolNode(tools))
```

The focused system prompt is helpful documentation for the model, but the smaller `tools` list is the enforcement
mechanism: `ToolNode` cannot call a tool it was not given. For a request that really requires both operations, make
the router explicitly return `("adder", "multiplier")`. For ambiguous or unsupported input, ask a clarifying question
instead of loading every MCP server.

## 11. Use an LLM to classify the MCP server subset

When ten server capabilities cannot be reliably described by a few keywords, use a separate LLM classification step.
Give that classifier a short, trusted server catalogue; require structured output; validate its selected names; then
construct the graph's `ToolNode` with only the approved tools. The classifier is not an authorization system—the
allowlist validation remains mandatory.

```text
user input + trusted MCP catalogue → LLM classifier → validate server names
→ discover selected tools → build ToolNode(selected tools) → run graph
```

```python
from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


SERVER_CATALOG: dict[str, str] = {
    "adder": "Adds two numbers.",
    "multiplier": "Multiplies two numbers.",
    # Add the remaining trusted server names and one short capability description here.
}


class MCPSelection(BaseModel):
    """LLM-proposed MCP server scope for one user request."""

    server_names: list[str] = Field(max_length=2, description="Names selected only from the supplied catalogue")
    needs_clarification: bool = Field(description="True when no safe server subset can be selected")
    clarification_question: str | None = Field(default=None, description="Question to ask when clarification is needed")


def validate_selection(selection: MCPSelection, allowed_names: Sequence[str]) -> tuple[str, ...]:
    """Return an allowlisted, de-duplicated server selection or raise a safe error."""
    selected = tuple(dict.fromkeys(selection.server_names))
    unknown_names = set(selected).difference(allowed_names)
    if unknown_names:
        raise ValueError(f"Classifier returned unknown MCP servers: {sorted(unknown_names)}")
    if selection.needs_clarification or not selected:
        question = selection.clarification_question or "Which capability do you need?"
        raise ValueError(question)
    return selected


async def selected_tools_from_llm(
    client: MultiServerMCPClient, user_input: str
) -> tuple[list[BaseTool], str]:
    """Classify, validate, and discover only the MCP tools needed for one graph run."""
    catalogue = "\n".join(f"- {name}: {description}" for name, description in SERVER_CATALOG.items())
    classifier = ChatOllama(model="llama3.2:3b", temperature=0).with_structured_output(MCPSelection)
    result = classifier.invoke(
        [
            SystemMessage(
                content=(
                    "Choose zero, one, or two MCP server names only from this catalogue. "
                    "Set needs_clarification=true when the request is ambiguous or unsupported.\n\n"
                    f"Available MCP servers:\n{catalogue}"
                )
            ),
            HumanMessage(content=user_input),
        ]
    )
    if not isinstance(result, MCPSelection):
        raise TypeError("Classifier did not return the requested MCPSelection schema.")

    selected_names = validate_selection(result, tuple(SERVER_CATALOG))
    tools: list[BaseTool] = []
    for server_name in selected_names:
        tools.extend(await client.get_tools(server_name=server_name))
    system_prompt = f"Use only the selected MCP capabilities: {', '.join(selected_names)}."
    return tools, system_prompt
```

Use the result before the graph is built for a request:

```python
tools, system_prompt = await selected_tools_from_llm(client, user_input)
model = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools)

# call_model closes over model and system_prompt; ToolNode receives the same tools.
builder.add_node("tools", ToolNode(tools))
```

For a clarification result, catch the `ValueError` at your API or user-interface boundary and show its safe question
instead of constructing a graph. This prevents an ambiguous request from loading tools from all ten servers.

### Requests that need more than one MCP server

Yes—combined requests are supported. For `"Add 2 and 3, then multiply the result by 4"`, the classifier should return:

```json
{
  "server_names": ["adder", "multiplier"],
  "needs_clarification": false,
  "clarification_question": null
}
```

`validate_selection()` accepts both names after checking the allowlist. `selected_tools_from_llm()` then loads both
tool sets, and `ToolNode(tools)` receives only those tools. The graph can therefore call `add`, route the result back
to the model, then call `multiply` without exposing the other eight MCP servers. Increase `max_length=2` only when
you intentionally support workflows that need more than two services.

For current API details and deployment options, see the official [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview).
