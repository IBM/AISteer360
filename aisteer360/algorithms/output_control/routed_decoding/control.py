"""RoutedDecoding control: route each row to a response strategy based on probe decisions."""
from __future__ import annotations

import warnings
from dataclasses import replace

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from aisteer360.algorithms.core.internals.probes import ProbeSetFit
from aisteer360.algorithms.output_control._common.drivers.phased import (
    Fixed,
    PhasedDriver,
)
from aisteer360.algorithms.output_control.base import OutputControl

from .actions import Generate, Prefix, Respond
from .args import RoutedDecodingArgs


class RoutedDecoding(PhasedDriver):
    """Decoding driver that routes each prompt to a response strategy via probe decisions.

    `RoutedDecoding` pairs a `ProbeSet` (named calibrated probes scored in one read-only
    forward) with a `RoutingRules` set (ordered boolean rules over the probe names, first match
    wins, evaluated per row). Decoding proceeds in three steps:

    1. **Probe pass**: one read-only forward of the prompt, issued by `ProbeSet.read()`, yields
       each probe's per-row signed score and decision (`score >= 0`).
    2. **Routing**: the decisions feed `rules.route()`, matching each row to its first
       satisfied rule (or the default).
    3. **Execution**: each row's action is lowered to a phase plan and executed by the
       inherited plan runner. `Respond(text)` splices the canned tokens with no generation;
       `Prefix(text)` splices the prefix then generates; `Generate()` delegates the row to
       `model.generate`. A raw `list[Fixed | Generated]` is also accepted as an action. The
       composed logits processors and stopping criteria apply in every generated phase, so the
       driver contract is preserved.

    The probes arrive fitted (a `ProbeSet`) or as a deferred recipe (a `ProbeSetFit`) that
    `steer()` fits on the model the pipeline provides, so pipelines whose structural controls
    produce the final weights fit on those weights. A fitted set whose recorded model
    fingerprints differ from the pipeline's model raises at `steer()` unless
    `allow_model_mismatch=True`.

    The read wraps its forward in `auxiliary_pass(aligned=True)`, the standard marking for
    same-model forwards issued during decoding (`same_model_forwards = True`). Its capture
    hooks are closures registered for the duration of that single forward and are removed
    before decoding begins, so the re-processing of the prompt inside generated phases is
    never re-scored.

    Costs per row: a routed canned response costs one prompt forward and zero decode steps; a
    pass-through row costs one extra prompt forward compared to the default driver.

    When state controls carrying behavior transforms share the pipeline, their hooks are live
    during the probe pass. The pass is trajectory-aligned, so `"all"`-scope transforms, and
    position-scoped ones covering prompt positions, apply to it, and probe scores are measured
    under that steering; `token_scope="after_prompt"` transforms are inert (the pass contains
    only prompt positions). Those controls' condition scorers, gates, and position counters
    ignore the pass entirely, since it is auxiliary.

    Padding is handled per row: pad positions are stripped (via the attention mask) before plan
    execution, and each returned row is the original padded prompt plus its continuation, so
    the pipeline's prompt-length slicing stays exact.

    The most recent routing outcome is retained on `latest_routes` (one rule name per row,
    `"default"` for unmatched rows); per-probe decisions and scores are available on the probe
    set's `latest` readout.

    `runtime_kwargs`:

    - `"canned_responses"`: dict mapping rule names to replacement text, overriding the
      `Respond`/`Prefix` text of matching rules for this call only. Keys that do not name a
      `Respond`/`Prefix` rule are ignored with a warning.
    - `"base_generate"`: replacement for `model.generate` inside generated phases.
    """

    Args = RoutedDecodingArgs

    supports_batching: bool = True
    same_model_forwards: bool = True

    RUNTIME_KWARGS_SCHEMA = [
        {
            "name": "canned_responses",
            "type": "dict[str, str]",
            "description": "Per-call override of Respond/Prefix text, keyed by rule name.",
        },
    ]

    tokenizer: PreTrainedTokenizerBase | None = None

    def __init__(self, *args, **kwargs):
        # route through OutputControl (validate RoutedDecodingArgs, mirror fields, _configure)
        OutputControl.__init__(self, *args, **kwargs)

    def _configure(self) -> None:
        """Initialize the `PhasedDriver` fields the plan runner reads."""
        self.extract_after = None
        self.tokenizer = None
        self.latest_routes: list[str] = []

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Attach the tokenizer, resolve the probes on the pipeline's model, and validate.

        A `ProbeSetFit` is fitted here, on the model the pipeline provides (its `StatsSpec`,
        when present, is estimated on that model first). A fitted `ProbeSet` is checked
        against the model instead: any probe whose recorded `model_fingerprint` differs
        raises unless `allow_model_mismatch=True`, and probes with no recorded fingerprint
        are exempt.

        Args:
            model: The pipeline's model.
            tokenizer: Tokenizer used for splicing and padding. If None, attempts to retrieve
                from model attributes.

        Returns:
            The input model, unchanged.

        Raises:
            ValueError: If a fitted set's recorded fingerprints differ from the model's, the
                set's `model_type` does not match the model, or a rule references a probe name
                the set does not define.
        """
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)

        if isinstance(self.probes, ProbeSetFit):
            self.probes = self.probes.fit(model, self.tokenizer)
        elif not self.allow_model_mismatch:
            live_fingerprint = model_fingerprint(model)
            mismatched = [
                name for name, probe in self.probes.probes.items()
                if probe.meta.get("model_fingerprint") not in (None, live_fingerprint)
            ]
            if mismatched:
                raise ValueError(
                    "ProbeSet was fitted on a different model than this pipeline produced. "
                    "Pass a ProbeSetFit for steer-time fitting on the pipeline's final model, "
                    "or set allow_model_mismatch=True."
                )

        live_model_type = getattr(model.config, "model_type", "unknown")
        if self.probes.model_type != live_model_type:
            raise ValueError(
                f"ProbeSet was fitted on model_type {self.probes.model_type!r} but the "
                f"pipeline's model is {live_model_type!r}."
            )

        self.rules.validate_names(set(self.probes.names))
        return model

    def decode(self, input_ids, attention_mask, model: PreTrainedModel, logits_processors,
               stopping_criteria, runtime_kwargs, **gen_kwargs) -> torch.Tensor:
        """Read the probes on the prompt, route each row, and execute the routed phase plans.

        Args:
            input_ids: Prompt token ids `[B, T]` (a 1-D tensor is treated as one row).
            attention_mask: Prompt attention mask matching `input_ids`, or None.
            model: The pipeline's model.
            logits_processors: Composed logits-processor stack, applied in every generated
                phase.
            stopping_criteria: Composed stopping-criteria stack, applied in every generated
                phase.
            runtime_kwargs: Per-call parameters (see the class docstring).
            **gen_kwargs: Generation parameters forwarded to every generated phase.

        Returns:
            Full sequence ids `[B, L]` (prompt + continuation), padded per row.

        Raises:
            RuntimeError: If no tokenizer is attached; `steer()` must run first.
            TypeError: If a matched action cannot be lowered to a phase plan.
            ValueError: If a matched plan contains a prompt-replacing `Fixed` phase
                (`replace=True`).

        Warns:
            UserWarning: If `canned_responses` carries keys that do not name a
                `Respond`/`Prefix` rule.
        """
        if self.tokenizer is None:
            raise RuntimeError("RoutedDecoding requires a tokenizer; steer() must run first.")

        runtime_kwargs = runtime_kwargs or {}
        base_generate = runtime_kwargs.get("base_generate") or (model.generate if model is not None else None)
        overrides = runtime_kwargs.get("canned_responses") or {}

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        batch_size = input_ids.size(0)

        readout = self.probes.read(model, input_ids, attention_mask)
        matched = self.rules.route(readout.decisions)
        self.latest_routes = [rule.name if rule is not None else "default" for rule in matched]

        if overrides:
            rules_by_name = {rule.name: rule for rule in self.rules.rules}
            unusable = [
                key for key in overrides
                if key not in rules_by_name
                or not isinstance(rules_by_name[key].action, (Respond, Prefix))
            ]
            if unusable:
                warnings.warn(
                    f"canned_responses keys {unusable} do not name a Respond/Prefix rule and "
                    "are ignored.",
                    UserWarning,
                )

        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        final_sequences: list[torch.Tensor] = []
        for i in range(batch_size):
            rule = matched[i]
            if rule is not None:
                action = rule.action
                if rule.name in overrides and isinstance(action, (Respond, Prefix)):
                    action = replace(action, text=overrides[rule.name])
            else:
                action = self.rules.default_action if self.rules.default_action is not None else Generate()
            plan = self._lower(action)

            row_full = input_ids[i:i + 1]
            if attention_mask is not None:
                row_ids = row_full[:, attention_mask[i].bool()]  # strip pad positions
            else:
                row_ids = row_full
            full = self._run_plan(
                plan, row_ids, prompts[i], {}, model, base_generate,
                logits_processors, stopping_criteria, gen_kwargs,
            )
            continuation = full[0][row_ids.size(1):]
            final_sequences.append(torch.cat([row_full[0], continuation]))

        # right padding keeps each prompt at offsets [0, T), which the pipeline's
        # prompt-length slicing requires regardless of the tokenizer's configured side
        padded = self.tokenizer.pad(
            {"input_ids": [seq.tolist() for seq in final_sequences]},
            padding=True,
            padding_side="right",
            return_tensors="pt",
        ).to(input_ids.device)
        return padded["input_ids"]

    @staticmethod
    def _lower(action) -> list:
        """Lower an action to a phase plan.

        Args:
            action: A `Respond`/`Prefix`/`Generate` (or any object with a `plan()` method), or
                a raw list/tuple of `Fixed`/`Generated` phases.

        Returns:
            The phase plan list.

        Raises:
            TypeError: If the action is neither plan-bearing nor a phase sequence.
            ValueError: If the plan contains a prompt-replacing `Fixed` phase (`replace=True`);
                routed actions append to the row's prompt, which the returned-continuation
                accounting relies on.
        """
        if hasattr(action, "plan"):
            plan = action.plan()
        elif isinstance(action, (list, tuple)):
            plan = list(action)
        else:
            raise TypeError(
                f"Cannot lower action of type {type(action).__name__} to a phase plan; use "
                "respond()/prefix()/generate() or a list of Fixed/Generated phases."
            )
        if any(isinstance(phase, Fixed) and phase.replace for phase in plan):
            raise ValueError(
                "RoutedDecoding does not support prompt-replacing Fixed phases (replace=True); "
                "routed actions append to the row's prompt."
            )
        return plan
