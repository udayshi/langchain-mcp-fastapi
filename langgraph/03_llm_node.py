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
