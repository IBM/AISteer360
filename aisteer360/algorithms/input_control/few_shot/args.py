from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class FewShotArgs(BaseArgs):
    """Arguments for few-shot input control."""

    selector_name: str | None = field(
        default=None,
        metadata={"help": "Name of the example selector to use. If None, uses random selection."}
    )

    template: str | None = field(
        default=None,
        metadata={
            "help": "Custom template for the system prompt. Use {example_blocks} and {directive} as placeholders."}
    )

    directive: str | None = field(
        default=None,
        metadata={"help": "Directive statement at the beginning of the system prompt."}
    )

    positive_example_pool: list[dict] | None = field(
        default=None,
        metadata={"help": "Pool of positive examples to sample from at runtime."}
    )

    negative_example_pool: list[dict] | None = field(
        default=None,
        metadata={"help": "Pool of negative examples to sample from at runtime."}
    )

    k_positive: int | None = field(
        default=None,
        metadata={"help": "Number of positive examples to sample from the pool."}
    )

    k_negative: int | None = field(
        default=None,
        metadata={"help": "Number of negative examples to sample from the pool."}
    )

    def __post_init__(self):
        if self.selector_name and self.selector_name not in ["random"]:
            raise ValueError(f"Unknown selector: {self.selector_name}")

        if self.positive_example_pool is not None or self.negative_example_pool is not None:
            if self.k_positive is None and self.positive_example_pool:
                raise ValueError("k_positive must be specified when positive_example_pool is provided.")
            if self.k_negative is None and self.negative_example_pool:
                raise ValueError("k_negative must be specified when negative_example_pool is provided.")
