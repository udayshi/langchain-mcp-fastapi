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
