"""MTA (Medical Triage Alignment) dataset utilities.

Download, convert, and augment the MTA dataset from the NAACL 2024 paper:
https://aclanthology.org/2024.naacl-industry.18.pdf

The raw data is hosted in the ITM-Kitware/align-system repository on GitHub.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

# Pinned commit for reproducibility
_GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/ITM-Kitware/align-system/"
    "6e97e2dcfb46556d0ce33cf07cd6e314bbea1ea1/"
    "example_data/input_output_files/NAACL24_dataset_split"
)

# Map attribute name → raw filename on GitHub
_RAW_FILENAMES = {
    "continuing_care": "oracle_continuing_care_training_input_output.json",
    "fairness": "oracle_fairness_training_input_output.json",
    "moral_desert": "oracle_moral_desert_training_input_output.json",
    "protocol_focus": "oracle_protocol_focus_training_input_output.json",
    "risk_aversion": "oracle_risk_aversion_training_input_output.json",
    "utilitarianism": "oracle_utilitarianism_training_input_output.json",
}

# Reasoning templates for enriching semantic completions
_REASONING_TEMPLATES = {
    "continuing_care": {
        "positive_reason": (
            "continuity of care requires maintaining ongoing treatment "
            "and not abandoning a patient mid-course"
        ),
        "negative_reason": (
            "this prioritizes immediate need over the existing "
            "treatment relationship"
        ),
    },
    "fairness": {
        "positive_reason": (
            "fairness requires equitable resource allocation regardless "
            "of personal connections or status"
        ),
        "negative_reason": (
            "this is influenced by personal relationships or non-medical "
            "factors that compromise impartial treatment"
        ),
    },
    "moral_desert": {
        "positive_reason": (
            "those who are victims or acted innocently deserve priority "
            "over those who caused harm"
        ),
        "negative_reason": (
            "this patient bears moral responsibility for the situation"
        ),
    },
    "protocol_focus": {
        "positive_reason": (
            "adherence to established medical protocols ensures safety "
            "and accountability"
        ),
        "negative_reason": (
            "this deviates from protocol, relying on personal judgment "
            "over established guidelines"
        ),
    },
    "risk_aversion": {
        "positive_reason": (
            "this is the safer, more predictable option that minimizes "
            "worst-case outcomes"
        ),
        "negative_reason": (
            "this option carries greater uncertainty and risk of "
            "catastrophic failure despite higher potential upside"
        ),
    },
    "utilitarianism": {
        "positive_reason": (
            "utilitarian reasoning prioritizes the greatest total good "
            "across all affected parties"
        ),
        "negative_reason": (
            "this benefits fewer people or prioritizes individual cases "
            "over aggregate welfare"
        ),
    },
}

_SCENARIO_DELIMITER = "\n\nScenario:"
_APPLY_CRITERIA_MARKER = "[APPLY_CRITERIA]"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_raw(attribute: str, dest_path: str | Path) -> Path:
    """Download raw NAACL24 MTA data from GitHub.

    Args:
        attribute: Attribute name (e.g. ``"moral_desert"``).
        dest_path: Local path to save the downloaded JSON file.

    Returns:
        Path to the downloaded file.
    """
    if attribute not in _RAW_FILENAMES:
        raise ValueError(
            f"Unknown attribute '{attribute}'. "
            f"Available: {', '.join(_RAW_FILENAMES)}"
        )
    filename = _RAW_FILENAMES[attribute]
    url = f"{_GITHUB_RAW_BASE}/{filename}"
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_oracle_agreement(scenario: dict, kdma_key: str) -> bool:
    """Check if the oracle's choice agrees with the high-KDMA direction."""
    output = scenario.get("output", [])
    if not output:
        return True

    chosen_idx = output[0].get("choice", -1)
    choices = scenario.get("input", {}).get("choices", [])

    if not choices or chosen_idx < 0 or chosen_idx >= len(choices):
        return True

    scores = [
        (j, c.get("kdma_association", {}).get(kdma_key, 0))
        for j, c in enumerate(choices)
    ]
    max_idx = max(scores, key=lambda x: x[1])[0]
    return chosen_idx == max_idx


def _select_binary_choices(
    choices: list[dict], kdma_key: str
) -> tuple[dict, dict, int, int] | None:
    """Select the 2 most contrasting choices by KDMA score."""
    scored = []
    for i, choice in enumerate(choices):
        kdma = choice.get("kdma_association", {})
        if kdma and kdma_key in kdma:
            scored.append((i, kdma[kdma_key]))

    if len(scored) < 2:
        return None

    scored.sort(key=lambda x: x[1])
    low_idx, low_score = scored[0]
    high_idx, high_score = scored[-1]

    if high_score == low_score:
        return None

    return choices[high_idx], choices[low_idx], high_idx, low_idx


def _enrich_completion(
    choice_desc: str, attribute: str, is_positive: bool
) -> str:
    """Enrich a bare choice description with reasoning context."""
    templates = _REASONING_TEMPLATES.get(attribute)
    if not templates:
        return f" {choice_desc}"
    reason = (
        templates["positive_reason"]
        if is_positive
        else templates["negative_reason"]
    )
    desc_lower = (
        choice_desc[0].lower() + choice_desc[1:]
        if choice_desc
        else choice_desc
    )
    return f" {desc_lower}, because {reason}"


def _anonymize_patient_labels(text: str) -> str:
    """Replace Patient A/B with neutral references."""
    text = re.sub(r"\bPatient A\b", "the first patient", text)
    text = re.sub(r"\bPatient B\b", "the second patient", text)
    return text


def _split_prompt(flat_text: str) -> tuple[str, str]:
    """Split a flat prompt into (system, user) at the Scenario: delimiter."""
    idx = flat_text.find(_SCENARIO_DELIMITER)
    if idx == -1:
        raise ValueError(
            f"Could not find '{_SCENARIO_DELIMITER}' in prompt text"
        )
    system = flat_text[:idx].strip()
    # Skip the 2 leading newline characters so user portion starts with "Scenario:"
    user = flat_text[idx + 2:].strip()
    return system, user


def _get_binary_choices(
    scenario: dict, kdma_key: str
) -> tuple[dict, dict] | None:
    """Extract (high_choice, low_choice) from a scenario."""
    inp = scenario.get("input", {})
    choices = inp.get("choices", [])

    if len(choices) == 2:
        kdma_0 = choices[0].get("kdma_association", {}).get(kdma_key, 0)
        kdma_1 = choices[1].get("kdma_association", {}).get(kdma_key, 0)
        if kdma_0 == kdma_1:
            return None
        if kdma_0 >= kdma_1:
            return choices[0], choices[1]
        return choices[1], choices[0]
    elif len(choices) > 2:
        result = _select_binary_choices(choices, kdma_key)
        if result is None:
            return None
        return result[0], result[1]
    return None


# ---------------------------------------------------------------------------
# Conversion: raw → eval format
# ---------------------------------------------------------------------------


def convert_to_eval(
    raw_data: list[dict], attribute: str, kdma_key: str
) -> list[dict]:
    """Convert raw NAACL24 data to MTA evaluation format.

    Ground truth is the choice with the highest ``kdma_association[kdma_key]``
    score (i.e. ``attribute: 10.0``), regardless of the oracle's choice.

    Args:
        raw_data: Raw scenario list from the NAACL24 dataset.
        attribute: Attribute name (e.g. ``"moral_desert"``).
        kdma_key: KDMA key used for scoring (e.g. ``"moral_deservingness"``).

    Returns:
        List of evaluation scenario dicts with ``input`` and ``output`` keys.
    """
    converted = []
    for idx, scenario in enumerate(raw_data):
        inp = scenario.get("input", {})
        choices = inp.get("choices", [])

        result = _get_binary_choices(scenario, kdma_key)
        if result is None:
            continue
        high_choice, low_choice = result

        if len(choices) == 2:
            binary_choices = choices
            high_idx = choices.index(high_choice)
        else:
            binary_choices = [
                {**high_choice, "action_id": f"{idx}.action_0"},
                {**low_choice, "action_id": f"{idx}.action_1"},
            ]
            high_idx = 0

        correct_action_id = binary_choices[high_idx]["action_id"]
        full_state = inp.get("full_state", {})

        converted.append({
            "input": {
                "scenario_id": attribute,
                "full_state": full_state,
                "state": full_state.get("unstructured", ""),
                "choices": [
                    {
                        "action_id": c.get("action_id", f"{idx}.action_{i}"),
                        "action_type": c.get("action_type", "SITREP"),
                        "unstructured": c.get("unstructured", ""),
                        "kdma_association": c.get("kdma_association", {}),
                    }
                    for i, c in enumerate(binary_choices)
                ],
            },
            "output": correct_action_id,
        })

    return converted


# ---------------------------------------------------------------------------
# Conversion: raw → base semantic training pairs
# ---------------------------------------------------------------------------


def convert_to_semantic_pairs(
    raw_data: list[dict], attribute: str, config: dict
) -> dict:
    """Convert raw NAACL24 data to base semantic training pairs.

    Args:
        raw_data: Raw scenario list from the NAACL24 dataset.
        attribute: Attribute name (e.g. ``"moral_desert"``).
        config: Attribute configuration dict with keys
            ``primary_system_full``, ``question_phrasing``, ``kdma_key``.

    Returns:
        Dict with ``"train"`` key containing list of contrastive pair dicts.
    """
    kdma_key = config["kdma_key"]
    system_prompt = config["primary_system_full"]
    question = config["question_phrasing"]
    examples = []

    for idx, scenario in enumerate(raw_data):
        if not _check_oracle_agreement(scenario, kdma_key):
            continue

        full_state = scenario.get("input", {}).get("full_state", {})
        scenario_text = full_state.get("unstructured", "")
        if not scenario_text:
            continue

        result = _get_binary_choices(scenario, kdma_key)
        if result is None:
            continue
        high_choice, low_choice = result

        high_desc = high_choice.get("unstructured", "").strip()
        low_desc = low_choice.get("unstructured", "").strip()
        if not high_desc or not low_desc:
            continue

        anon_scenario = _anonymize_patient_labels(scenario_text)

        prompt = (
            f"{system_prompt}"
            f"\n\nScenario: {anon_scenario}"
            f"\n\n{question}"
            f"\n\nAnswer: I would choose"
        )

        positive = _enrich_completion(high_desc, attribute, is_positive=True)
        negative = _enrich_completion(low_desc, attribute, is_positive=False)

        examples.append({
            "prompt": prompt,
            "positive": positive,
            "negative": negative,
            "type": "semantic",
            "id": f"semantic_scene{idx}",
        })

    return {"train": examples}


# ---------------------------------------------------------------------------
# Augmentation: base semantic → chat-format augmented
# ---------------------------------------------------------------------------


def augment_semantic(
    base_examples: list[dict], attribute: str, config: dict
) -> dict:
    """Augment base semantic pairs into chat-format training data.

    For each base example, creates 2 orderings x 7 augmentation types = 14
    variants. The 7 types are: original, minimal, marked, emphasis,
    no_system, rephrased, negated.

    Args:
        base_examples: List of base contrastive pair dicts from
            :func:`convert_to_semantic_pairs`.
        attribute: Attribute name.
        config: Attribute configuration dict with keys
            ``primary_system_full``, ``primary_system_minimal``,
            ``primary_system_negated``, ``emphasis_prompts``.

    Returns:
        Dict with ``"train"`` key containing augmented examples and
        ``"metadata"`` with counts.
    """
    primary_full = config["primary_system_full"]
    primary_minimal = config["primary_system_minimal"]
    primary_marked = f"{primary_full}\n\n{_APPLY_CRITERIA_MARKER}"
    primary_negated = config["primary_system_negated"]
    emphasis_prompts = config["emphasis_prompts"]

    augmented = []
    base_idx = 0

    for example in base_examples:
        flat_prompt = example["prompt"]
        positive = example["positive"]
        negative = example["negative"]
        base_id = example["id"]

        _system, user_order1 = _split_prompt(flat_prompt)
        # order2: same scenario, swapped completions (debiases completion order)
        user_order2 = user_order1

        for order, user in [("order1", user_order1), ("order2", user_order2)]:
            id_prefix = f"{base_id}_{order}_chat"

            if order == "order2":
                cur_pos, cur_neg = negative, positive
            else:
                cur_pos, cur_neg = positive, negative

            aug_specs = [
                ("original", primary_full),
                ("minimal", primary_minimal),
                ("marked", primary_marked),
                ("emphasis", emphasis_prompts[base_idx % len(emphasis_prompts)]),
                ("no_system", ""),
                ("rephrased", primary_full),
                ("negated", primary_negated),
            ]

            for type_name, system in aug_specs:
                augmented.append({
                    "system": system,
                    "prompt": user,
                    "positive": cur_pos,
                    "negative": cur_neg,
                    "format": "chat",
                    "type": type_name,
                    "id": f"{id_prefix}_{type_name}",
                })

            base_idx += 1

    type_counts = dict(Counter(ex["type"] for ex in augmented))

    return {
        "metadata": {
            "format": "chat",
            "data_format": "behavior",
            "attribute": attribute,
            "total_examples": len(augmented),
            "types": type_counts,
        },
        "train": augmented,
    }
