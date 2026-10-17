from __future__ import annotations

import warnings

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.common.drivers.phased import Fixed, Generated, PhasedDriver
from steerability.algorithms.output_control.phased_decoding.args import PhasedDecodingArgs

_FIXED_KEYS = {"fixed", "replace", "add_special_tokens"}
_GENERATE_SUBKEYS = {"until", "until_token_ids", "budget"}


def _parse_phase(entry: dict):
    """Parse one plan entry into a `Fixed` / `Generated`, validating the declarative grammar."""
    if not isinstance(entry, dict):
        raise ValueError(f"Each plan entry must be a dict, got {type(entry).__name__}.")
    has_fixed = "fixed" in entry
    has_generate = "generate" in entry
    if has_fixed == has_generate:
        raise ValueError(
            f"Each plan entry must have exactly one of 'fixed' or 'generate', got keys {sorted(entry)}."
        )

    if has_fixed:
        unknown = set(entry) - _FIXED_KEYS
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in a 'fixed' phase.")
        text = entry["fixed"]
        if not (isinstance(text, str) or callable(text)):
            raise ValueError("'fixed' must be a str or a (prompt_text, params) -> str callable.")
        return Fixed(
            text,
            replace=bool(entry.get("replace", False)),
            add_special_tokens=bool(entry.get("add_special_tokens", False)),
        )

    gen = entry["generate"]
    if not isinstance(gen, dict):
        raise ValueError("'generate' must map to a dict with optional 'until' / 'budget' keys.")
    unknown = set(gen) - _GENERATE_SUBKEYS
    if unknown:
        raise ValueError(f"Unknown subkey(s) {sorted(unknown)} in a 'generate' phase.")
    budget = gen.get("budget")
    if budget is not None and (not isinstance(budget, int) or budget <= 0):
        raise ValueError(f"'generate' budget must be a positive integer when set, got {budget!r}.")
    until = gen.get("until")
    if until is not None and not isinstance(until, str):
        raise ValueError(f"'generate' until must be a string when set, got {type(until).__name__}.")
    until_token_ids = gen.get("until_token_ids") or ()
    if isinstance(until_token_ids, (str, bytes)) or not isinstance(until_token_ids, (list, tuple)):
        raise ValueError(
            f"'generate' until_token_ids must be a sequence of ints when set, got "
            f"{type(until_token_ids).__name__}."
        )
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in until_token_ids):
        raise ValueError("'generate' until_token_ids must contain only ints.")
    return Generated(until=until, until_token_ids=tuple(until_token_ids), budget=budget)


class PhasedDecoding(PhasedDriver):
    """Decoding driver that runs a configured plan of fixed and generated phases.

    The `plan` argument is a list of phase dicts. Each dict has exactly one of two keys:

    - `{"fixed": text, "replace": False, "add_special_tokens": False}` appends `text` (a string, or a
      callable `(prompt_text, params) -> str`) without generating. With `"replace": True`, the text
      replaces the stream instead.
    - `{"generate": {"until": None, "until_token_ids": (), "budget": None}}` generates until the
      `until` text, a token in `until_token_ids`, or `budget` tokens, whichever comes first.
      `{"generate": {}}` has no boundary of its own and is limited only by the call's generation
      parameters.

    The plan is validated at construction, and an invalid entry raises `ValueError`. A plan without a
    `"generate"` phase emits a `UserWarning`. The `extract_after` argument (default None) sets a
    marker, e.g., `"</think>"`. When a candidate's decoded continuation contains the marker, only the
    text after its last occurrence is returned. Methods from the literature correspond to the
    following plans:

    - Budget forcing (s1): a thinking phase with a budget, a fixed `"Wait"`, a second thinking phase,
      a fixed closing tag, and an answer phase without a budget.
    - Response prefill: `plan=[{"fixed": "Sure, here is the answer:\\n"}, {"generate": {}}]`.
    - Scaffolded output: alternating `{"fixed": <header>}` and `{"generate": {"until": "\\n\\n"}}`
      entries.
    - Thinking intervention: a replacing `"fixed"` phase that rewrites the prompt with the
      intervention text, then a `"generate"` phase, with `extract_after="</think>"`.

    Every `"generate"` phase applies the composed logits processors and stopping criteria, and a
    step-level control in the pipeline steers each generated phase. The caller's `max_new_tokens`
    limits the tokens appended to each candidate across all phases.

    The following `runtime_kwargs` are accepted:

    - `"params"`: A mapping passed as `params` to callable `"fixed"` texts. A list or tuple value
      contains one entry per row, and any other value applies to every row.

    Reference:

    - "s1: Simple test-time scaling"
      Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi,
      Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto
      [https://arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393)

    - "Effectively Controlling Reasoning Models through Thinking Intervention"
      Tong Wu, Chong Xiang, Jiachen T. Wang, G. Edward Suh, Prateek Mittal
      [https://arxiv.org/abs/2503.24370](https://arxiv.org/abs/2503.24370)
    """

    Args = PhasedDecodingArgs

    tokenizer: PreTrainedTokenizerBase | None = None

    def _configure(self) -> None:
        """Parse and validate the declarative plan into `Fixed` and `Generated` phases.

        The constructor copies the `plan` argument onto the instance, where it would shadow the
        `plan()` method that the driver calls. The parsed phases are stored in `_parsed_plan`, and
        the instance attribute is removed so that `plan()` is visible again.

        Raises:
            ValueError: If a plan entry is malformed.

        Warns:
            UserWarning: If the plan contains no `generate` phase.
        """
        raw_plan = self.plan
        self._parsed_plan = [_parse_phase(entry) for entry in raw_plan]
        del self.__dict__["plan"]  # unshadow the plan() method mirrored over by __init__
        if not any(isinstance(phase, Generated) for phase in self._parsed_plan):
            warnings.warn(
                "PhasedDecoding plan contains no 'generate' phase; the model will not produce any "
                "tokens beyond the spliced fixed text.",
                UserWarning,
            )
        self.tokenizer = None
        # self.extract_after is already mirrored from PhasedDecodingArgs

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **_) -> PreTrainedModel:
        """Lightweight preparation; attach the tokenizer used to splice phase boundaries."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def plan(self, prompt_text: str, params: dict) -> list:
        """Return the phase plan parsed from the `plan` argument.

        The same plan is returned for every row. Callable `Fixed` texts are evaluated per row when
        the plan runs.

        Args:
            prompt_text: The row's decoded prompt (unused).
            params: The row's plan parameters (unused).

        Returns:
            The list of `Fixed` and `Generated` phases.
        """
        return self._parsed_plan

    def max_rollouts_per_query(self) -> int:
        """Return the number of `Generated` phases in the plan.

        Each `Generated` phase requests one rollout per candidate.

        Returns:
            The bound on the rollouts for one candidate of one row.
        """
        return sum(isinstance(phase, Generated) for phase in self._parsed_plan)
