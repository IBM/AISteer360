from typing import Any

from datasets import Dataset

_PROMPT_KEYS = ["prompt", "question", "query", "input"]
_CHOSEN_KEYS = ["chosen", "chosen_response", "preferred", "pos", "accepted", "answer_chosen"]
_REJECTED_KEYS = ["rejected", "rejected_response", "dispreferred", "neg", "answer_rejected"]
_MESSAGES_KEYS = ["messages", "conversations"]


def _first_present(d: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in d:
            return d[key]
    return None


def _to_plain_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _strip_or_none(s: str | None) -> str | None:
    if s is None:
        return None
    return s.strip()


def standardize_preference_dataset(
    dataset: Dataset,
    drop_unknown_columns: bool = True,
) -> Dataset:
    """
    Return a dataset with exactly {'prompt','chosen','rejected'} as plain strings.
    If the dataset already has those keys, we coerce values to str and drop extras.
    If 'messages' (or similar) coexists with them, we drop 'messages' (pick one schema).
    This function does NOT attempt to synthesize 'chosen'/'rejected' from 'messages'.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("standardize_preference_dataset expects a datasets.Dataset")

    column_names = set(dataset.column_names)

    has_prompt = any(k in column_names for k in _PROMPT_KEYS)
    has_chosen = any(k in column_names for k in _CHOSEN_KEYS)
    has_rejected = any(k in column_names for k in _REJECTED_KEYS)
    has_messages = any(k in column_names for k in _MESSAGES_KEYS)

    if not (has_prompt and has_chosen and has_rejected):
        missing = []
        if not has_prompt: missing.append("prompt")
        if not has_chosen: missing.append("chosen")
        if not has_rejected: missing.append("rejected")
        raise ValueError(
            f"Preference dataset is missing required fields: {missing}. "
            "Provide columns equivalent to prompt/chosen/rejected, or map them before calling this function."
        )

    # build a canonical mapping function
    def to_preference_row(example: dict[str, Any]) -> dict[str, str]:
        prompt_val = _first_present(example, _PROMPT_KEYS)
        chosen_val = _first_present(example, _CHOSEN_KEYS)
        rejected_val = _first_present(example, _REJECTED_KEYS)

        if prompt_val is None or chosen_val is None or rejected_val is None:
            raise ValueError("Example lacks one of prompt/chosen/rejected after key resolution.")

        prompt = _strip_or_none(_to_plain_string(prompt_val))
        chosen = _strip_or_none(_to_plain_string(chosen_val))
        rejected = _strip_or_none(_to_plain_string(rejected_val))

        if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
            raise TypeError("prompt/chosen/rejected must be strings after standardization.")

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    # apply mapping
    standardized = dataset.map(
        to_preference_row,
        remove_columns=[c for c in dataset.column_names if c not in (_PROMPT_KEYS + _CHOSEN_KEYS + _REJECTED_KEYS)],
    )

    # optionally drop stray 'messages' / 'conversations' columns; TRL rejects mixed schemas
    if drop_unknown_columns:
        columns_to_drop = [c for c in standardized.column_names if c not in {"prompt", "chosen", "rejected"}]
        if columns_to_drop:
            standardized = standardized.remove_columns(columns_to_drop)

    return standardized
