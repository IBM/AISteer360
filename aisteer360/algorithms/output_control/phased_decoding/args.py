from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class PhasedDecodingArgs(BaseArgs):
    """Arguments for the `PhasedDecoding` generic (the phase-shape driver).

    The `plan` is a list of phase dicts, each with exactly one of two keys:

        - `{"fixed": <str | callable>, "replace": bool = False, "add_special_tokens": bool = False}`
          — splice text (a `str` literal, or a `(prompt_text, params) -> str` callable).
        - `{"generate": {"until": str | None = None, "budget": int | None = None}}` — generate until
          a boundary; `{"generate": {}}` is unbounded (bounded by the call's own kwargs/criteria).

    Plans whose `fixed` values are all strings are fully JSON-serializable (Benchmark-sweepable and
    log-friendly). Grammar validation happens in the control's `_configure()`.
    """

    plan: list = field(
        default=None,
        metadata={"help": "List of phase dicts (each with exactly one of 'fixed' or 'generate')."},
    )
    extract_after: str | None = field(
        default=None,
        metadata={"help": "Output rule: keep the original prompt prefix + the remainder after this "
                          "marker (ThinkingIntervention's tail extraction). None keeps the full stream."},
    )

    def __post_init__(self) -> None:
        if self.plan is None:
            raise ValueError("'plan' is required.")
        if not isinstance(self.plan, (list, tuple)) or len(self.plan) == 0:
            raise ValueError("'plan' must be a non-empty list of phase dicts.")
