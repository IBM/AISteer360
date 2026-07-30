"""
Core steering pipeline for composing and applying multiple LLM control methods.
"""
import contextlib
import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, overload

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)

from aisteer360.algorithms.core.utils.controls import (
    merge_controls,
    warn_if_adapt_messages_bypassed,
)
from aisteer360.algorithms.core.utils.generation import (
    apply_adapt_messages_and_tokenize,
    infer_finish_reason,
)
from aisteer360.algorithms.core.types import Output
from aisteer360.utils.tokenization import (
    ensure_pad_token,
    infer_attention_mask_from_ids,
    to_left_pad,
    warn_if_duplicate_bos,
)
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.output_control.base import (
    DecodingDriver,
    HFGenerateDriver,
    OutputControl,
)
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.structural_control.base import StructuralControl

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SteeringPipeline:
    """Main steering pipeline for applying various control methods to Hugging Face causal language models.

    Enables application of structural, state, input, and output controls in a coordinated manner.
    Controls are applied in a fixed bottom-up order during steering, then used together during generation.

    Workflow:

    1. Instantiate with a base model checkpoint and/or control objects.
    2. Call `steer()` once to apply all controls in order (structural, input, state, output).
    3. Use `generate()` for inference with steering applied, accepting str, list[str], chat, or tensor input.

    Args:
        model_name_or_path (str or pathlib.Path, optional): HuggingFace model hub name or local directory.
            Required when `lazy_init=False`. Ignored when `lazy_init=True` and the structural
            control returns a model.
        controls (Sequence[StructuralControl | StateControl | InputControl | OutputControl], optional):
            Controls for the steering pipeline. Every category accepts any number of controls,
            applied in list order. The output category additionally accepts at most one enabled
            `DecodingDriver` (the decode loop does not compose). Omitted input/structural categories
            fall back to no-op controls; an omitted output category uses the default decoding driver
            (`model.generate`).
        tokenizer_name_or_path (str, optional): Tokenizer location. Defaults to `model_name_or_path`.
        device_map (str or dict[str, int], optional): Device map (passed to
            `transformers.AutoModelForCausalLM.from_pretrained`). Defaults to `"auto"`.
            Cannot be used together with `device` parameter.
        device (torch.device, str, optional): Device (passed to model's `.to()` method).
            When specified, `device_map` must remain at its default value of `"auto"`.
        hf_model_kwargs (dict, optional): Extra keyword arguments passed to
            `transformers.AutoModelForCausalLM.from_pretrained`.
        trust_remote_code (bool, optional): Trust remote code when loading the tokenizer. Defaults to
            `False`. To trust remote code for the model, pass `trust_remote_code=True` via `hf_model_kwargs`.
        lazy_init (bool, optional): If `True`, defers loading the base model until `steer()` time.
            Useful when a `StructuralControl` will itself load or create the final weights
            (e.g., MergeKit). When `False`, the model is loaded during `SteeringPipeline`
            construction. Defaults to `False`.

    Raises:
        RuntimeError: If `generate()` is called before `steer()`
        ValueError: If more than one enabled `DecodingDriver` is supplied or required arguments are missing

    Note:

    - Every category accepts multiple controls, applied in list order. Omitted input/structural
        categories use no-op defaults; an omitted output category uses the pipeline's default
        decoding driver.
    - Controls with a `tokenizer` attribute will have it auto-injected if not already set

    For the state category, list order in `controls` defines the composition surface. List order sets
    `steer()` order, hook registration order, and execution order for hooks on the same module. PyTorch
    forward hooks chain, so a later hook receives the previous hook's returned output and pre-hooks chain
    likewise on inputs. Combinations that do not commute, such as ablate after add versus add after
    ablate, therefore produce order-sensitive results. A gated or condition-scoring control placed after
    another observes activations already edited at earlier layers by upstream list entries.

    For the input category, adaptation runs in two phases. On chat input, every control's
    `adapt_messages` runs in list order over the message batch (each non-None return feeds the next
    control); the result is templated and tokenized once, then every control whose `adapt_messages`
    returned None runs its token-level `adapt` in list order over the token stream. On text/tensor
    input there is no message phase; every control's `adapt` runs in list order. List order is
    authoritative within each phase, but the message phase structurally precedes the token phase:
    with `[TokenOnlyControl, MessageLevelControl]` on chat input, the message-level control's effect
    lands first even though it is listed second (tokens do not exist before templating). Place
    semantic rewriters (e.g. `PRewrite`, `CPO`, `GEPA`) before surface formatting (e.g. `FewShot`),
    since a rewriter trained on bare instructions degrades on exemplar-prepended input.

    For the structural category, `steer()` threads the model through the controls in list order:
    each control receives the previous control's returned model (and the possibly mutated tokenizer).
    Nothing implicit happens between stages (no adapter merging, no embedding-resize reconciliation);
    stage compatibility is the caller's responsibility.
    """

    # construction args
    model_name_or_path: str | Path | None = None
    controls: Sequence[StructuralControl | StateControl | InputControl | OutputControl] = ()
    tokenizer_name_or_path: str | None = None
    device_map: str | dict[str, int] | int | torch.device | None = "auto"
    device: torch.device | str | None = None
    hf_model_kwargs: dict = field(default_factory=dict)
    trust_remote_code: bool = False
    lazy_init: bool = False

    # lazy‑filled fields
    model: PreTrainedModel | None = field(init=False, default=None)
    tokenizer: AutoTokenizer | None = field(init=False, default=None)

    structural_controls: list[StructuralControl] = field(init=False)
    input_controls: list[InputControl] = field(init=False)
    state_controls: list[StateControl] = field(init=False)
    output_controls: list[OutputControl] = field(init=False)
    _default_driver: DecodingDriver = field(init=False, repr=False)

    _is_steered: bool = field(default=False, init=False, repr=False)
    _warned_tensor_with_adapt_messages: bool = field(default=False, init=False, repr=False)
    _warned_duplicate_bos: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:

        # sort/validate the supplied steering methods
        controls_merged = merge_controls(self.controls)
        self.structural_controls = controls_merged["structural_controls"]
        self.input_controls = controls_merged["input_controls"]
        self.state_controls = controls_merged["state_controls"]
        self.output_controls = controls_merged["output_controls"]
        self._default_driver = HFGenerateDriver()

        # load HF artifacts
        if not self.lazy_init:
            if self.model_name_or_path is None:
                raise ValueError("`model_name_or_path` must be provided when lazy_init=False")

            if self.device is not None and self.device_map != "auto":
                raise ValueError("Cannot specify both `device` and `device_map`.")

            if self.device is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    **self.hf_model_kwargs,
                )
                self.model = self.model.to(self.device)
                self.device = self.model.device
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=self.device_map,
                    **self.hf_model_kwargs,
                )
                self.device = self.model.device

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path or self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
            self.tokenizer = ensure_pad_token(self.tokenizer)
        else:
            if isinstance(self.tokenizer_name_or_path, (str, Path)):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    trust_remote_code=self.trust_remote_code
                )
                self.tokenizer = ensure_pad_token(self.tokenizer)

        # late‑inject tokenizer into controls that accept it
        controls_iter = (*self.structural_controls, *self.input_controls, *self.state_controls, *self.output_controls)
        for control in controls_iter:
            if hasattr(control, "tokenizer") and getattr(control, "tokenizer") is None:
                setattr(control, "tokenizer", self.tokenizer)

    @property
    def supports_batching(self) -> bool:
        """Return True if all enabled controls in this pipeline are batch-safe.

        The default decoding driver is batch-safe, so an empty `output_controls` list is vacuously
        true and does not constrain batching.
        """
        controls = (
            *self.structural_controls,
            *self.input_controls,
            *self.state_controls,
            *self.output_controls,
        )
        return all(
            getattr(control, "supports_batching", False)
            for control in controls
            if getattr(control, "enabled", True)
        )

    def _warn_on_runtime_kwargs_overlap(self) -> None:
        """Warn (UserWarning, once) when two or more enabled controls declare the same
        `RUNTIME_KWARGS_SCHEMA` variable name.

        All controls read from the single `runtime_kwargs` dict passed to `generate()`, so a shared
        name means one value feeds several controls. Sharing can be intentional, hence a warning
        rather than an error.
        """
        declared: dict[str, list[str]] = {}
        controls = (*self.structural_controls, *self.input_controls, *self.state_controls, *self.output_controls)
        for control in controls:
            if not getattr(control, "enabled", True):
                continue
            for entry in getattr(control, "RUNTIME_KWARGS_SCHEMA", []):
                name = entry.get("name")
                if name:
                    declared.setdefault(name, []).append(type(control).__name__)
        overlaps = {name: owners for name, owners in declared.items() if len(owners) > 1}
        if overlaps:
            details = "; ".join(
                f"'{name}' is declared by {', '.join(owners)}" for name, owners in overlaps.items()
            )
            warnings.warn(
                f"Multiple controls declare the same runtime_kwargs variable: {details}. "
                "All controls read from the one runtime_kwargs dict, so these controls will share "
                "the same value at inference time.",
                UserWarning,
            )

    def steer(self, **steer_kwargs) -> None:
        """Apply all steering controls to the model in place.

        Executes each control's steer() method in a fixed bottom-up order: structural -> input -> state -> output,
        and in list order within each category. This ensures that higher-level controls always see the final
        configured model from lower levels.

        If any control's steer() method returns a PreTrainedModel instance, it replaces the current model for
        subsequent controls, so structural controls thread the model through in list order.

        Args:
            **steer_kwargs: Keyword arguments passed to all control steer() methods

        Warns:
            UserWarning: If two or more enabled controls declare the same `RUNTIME_KWARGS_SCHEMA`
                variable name.

        Raises:
            RuntimeError: If called more than once or no model available after steering
        """
        if self._is_steered:
            return

        self._warn_on_runtime_kwargs_overlap()

        # steer each control (bottom-up order: structural -> input -> state -> output)
        for control in (*self.structural_controls, *self.input_controls, *self.state_controls, *self.output_controls):
            steer_fn = getattr(control, "steer", None)
            if callable(steer_fn):
                maybe_new_model = steer_fn(self.model, tokenizer=self.tokenizer, **steer_kwargs)
                if isinstance(maybe_new_model, nn.Module):
                    self.model = maybe_new_model

        # safety checks
        if self.model is None:
            raise RuntimeError(
                "No model is available after steering. Either provide a base model (lazy_init=False) or ensure a "
                "`StructuralControl` returns one."
            )

        if self.tokenizer is None:
            repo = getattr(self.model, "name_or_path", None)
            source = repo or self._structural_out_path()
            if source is None:
                raise RuntimeError("Failed to resolve tokenizer post‑steer.")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    source,
                    trust_remote_code=self.trust_remote_code,
                )
                self.tokenizer = ensure_pad_token(self.tokenizer)

            except Exception as exception:
                raise RuntimeError("Failed to resolve tokenizer post‑steer.") from exception

        for control in (*self.structural_controls, *self.input_controls, *self.state_controls, *self.output_controls):
            if hasattr(control, "tokenizer") and getattr(control, "tokenizer", None) is None:
                setattr(control, "tokenizer", self.tokenizer)

        # return steered pipeline
        self._is_steered = True

    def _structural_out_path(self) -> Path | None:
        """The last structural control's non-empty `args.out_path`, as a tokenizer-directory fallback.

        Scans `structural_controls` in reverse so the most recently produced artifact wins; controls
        without an `args` attribute or without `out_path` are skipped. When more than one control
        defines `out_path`, logs at info level which path won.
        """
        candidates = [
            (type(control).__name__, out_path)
            for control in self.structural_controls
            if (out_path := getattr(getattr(control, "args", None), "out_path", None))
        ]
        if not candidates:
            return None
        winner_name, winner_path = candidates[-1]
        if len(candidates) > 1:
            logger.info(
                "Multiple structural controls define out_path (%s); using %s from %s.",
                ", ".join(name for name, _ in candidates), winner_path, winner_name,
            )
        return Path(winner_path)

    def _prepare_inputs(
            self,
            input_ids: list[int] | torch.LongTensor,
            attention_mask: torch.Tensor | None,
            runtime_kwargs: dict | None,
            message_handled: frozenset[int] = frozenset(),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the token-level input-control chain and normalize input tensors.

        Runs each input control's `adapt` in list order (each control receives the previous
        control's output), then ensures both input_ids and attention_mask are properly shaped
        tensors on the correct device.

        Args:
            input_ids: Input token IDs as list or tensor [seq_len] or [batch, seq_len]
            attention_mask: Optional attention mask matching input_ids shape
            runtime_kwargs: Per-call parameters for input controls
            message_handled: `id()`s of input controls whose `adapt_messages` already performed the
                adaptation before tokenization for this call; their token-level `adapt` is skipped so
                no control is applied twice to the same prompt.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (steered_input_ids, attention_mask), both as 2D tensors on model device
        """
        runtime_kwargs = runtime_kwargs or {}
        device = self.model.device

        # token-phase chain (controls already handled at message level are skipped)
        steered_input_ids = input_ids
        for control in self.input_controls:
            if id(control) in message_handled:
                continue
            steered_input_ids = control.adapt(
                steered_input_ids,
                runtime_kwargs=runtime_kwargs,
            )

        # normalize input_ids to 2D tensor
        if isinstance(steered_input_ids, list):
            steered_input_ids = torch.tensor(steered_input_ids, dtype=torch.long)
        if steered_input_ids.ndim == 1:
            steered_input_ids = steered_input_ids.unsqueeze(0)
        steered_input_ids = steered_input_ids.to(device)

        # normalize attention_mask
        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            # rebuild if length mismatch after input control transformation
            if attention_mask.shape[-1] != steered_input_ids.shape[-1]:
                attention_mask = None

        if attention_mask is None:
            if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
                attention_mask = infer_attention_mask_from_ids(steered_input_ids, self.tokenizer.pad_token_id)
            else:
                attention_mask = torch.ones_like(steered_input_ids, dtype=torch.long)

        attention_mask = attention_mask.to(dtype=steered_input_ids.dtype, device=device)

        self._warned_duplicate_bos = warn_if_duplicate_bos(
            steered_input_ids, attention_mask, self.tokenizer, self._warned_duplicate_bos
        )

        return steered_input_ids, attention_mask

    def _setup_state_controls(
            self,
            steered_input_ids: torch.Tensor,
            runtime_kwargs: dict | None,
            attention_mask: torch.Tensor | None = None,
            **kwargs,
    ) -> None:
        """Configure every state control's hooks for the current forward/generate call.

        Prepares each state control (in list order) by computing hooks based on the (already
        transformed) input and setting up the model reference for the context manager.

        Args:
            steered_input_ids: Input token IDs after input control transformation
            runtime_kwargs: Per-call parameters for state controls
            attention_mask: The prompt attention mask matching `steered_input_ids`. Forwarded to
                `get_hooks` so controls (e.g. CAST) score conditions on the real prompt tokens rather
                than re-deriving a pad mask by token identity.
            **kwargs: Additional arguments passed to get_hooks()
        """
        for state_control in self.state_controls:
            state_control.reset()  # reset before get_hooks() to clear state from previous generation
            hooks = state_control.get_hooks(
                steered_input_ids, runtime_kwargs, attention_mask=attention_mask, **kwargs
            )
            state_control.set_hooks(hooks)
            state_control._model_ref = self.model

    def _resolve_decoding_driver(self) -> DecodingDriver:
        """The sole enabled DecodingDriver, else the default (model.generate).

        merge_controls guarantees at most one enabled driver at construction; `enabled` is
        re-checked here so a driver disabled afterward falls back cleanly.
        """
        for control in self.output_controls:
            if isinstance(control, DecodingDriver) and getattr(control, "enabled", True):
                return control
        return self._default_driver

    def _collect_processors_and_criteria(
        self, input_ids, runtime_kwargs, attention_mask=None, for_scoring=False, **kwargs,
    ) -> tuple[list, list]:
        """(processors, criteria) from enabled output controls, in controls-list order.

        With `for_scoring=True`, only `include_in_scoring` controls contribute processors and
        criteria are skipped (there is no loop to stop). Each hook result is guarded with `or []`.
        """
        processors, criteria = [], []
        for control in self.output_controls:
            if not getattr(control, "enabled", True):
                continue
            if for_scoring and not getattr(control, "include_in_scoring", True):
                logger.info(
                    "compute_logprobs: skipping %s (include_in_scoring=False); scored logprobs will "
                    "not reflect this control's logits processors.",
                    type(control).__name__,
                )
                continue
            processors.extend(control.get_logits_processors(
                input_ids, runtime_kwargs, attention_mask=attention_mask, **kwargs) or [])
            if not for_scoring:
                criteria.extend(control.get_stopping_criteria(
                    input_ids, runtime_kwargs, attention_mask=attention_mask, **kwargs) or [])
        return processors, criteria

    def _compose_stacks(self, input_ids, runtime_kwargs, attention_mask, gen_kwargs,
                        ) -> tuple[LogitsProcessorList, StoppingCriteriaList]:
        """Compose the controls' processors and criteria, then append caller extras popped from
        `gen_kwargs` (mutates `gen_kwargs`).

        Caller-supplied `logits_processor` / `stopping_criteria` entries append after the
        pipeline's own processors and criteria (per-call extras apply on top of the pipeline's
        standing configuration) and the keys are removed from `gen_kwargs`, so exactly one
        authoritative stack of each kind exists, travelling as an explicit parameter. gen_kwargs
        reaching the driver never contains processor or criteria objects, so drivers that copy or
        serialize their kwargs are safe by construction, and a driver that ignores the stacks
        visibly ignores named parameters.
        """
        processors, criteria = self._collect_processors_and_criteria(
            input_ids, runtime_kwargs, attention_mask=attention_mask, **gen_kwargs
        )
        user_processors = gen_kwargs.pop("logits_processor", None) or []
        user_criteria = gen_kwargs.pop("stopping_criteria", None) or []
        return (
            LogitsProcessorList([*processors, *user_processors]),
            StoppingCriteriaList([*criteria, *user_criteria]),
        )

    def _apply_scoring_processors(self, logits, steered_input_ids, ref_output_ids,
                                  runtime_kwargs, attention_mask, is_encoder_decoder,
                                  **forward_kwargs) -> torch.Tensor:
        """Apply scoring-time logits processors position-by-position (teacher forcing).

        Processors receive the same `(prefix_ids, scores)` view as during generation. For causal
        models the prefix is `input ++ ref[:t]` when scoring `ref[t]`; for encoder-decoder models
        the prefix is the decoder ids `ref[:t+1]` when scoring `ref[t+1]` (matching the existing
        target alignment in both paths).
        """
        processors, _ = self._collect_processors_and_criteria(
            steered_input_ids, runtime_kwargs, attention_mask=attention_mask,
            for_scoring=True, **forward_kwargs,
        )
        if not processors:
            return logits
        stack = LogitsProcessorList(processors)
        with torch.no_grad():
            for t in range(logits.size(1)):
                prefix = (ref_output_ids[:, : t + 1] if is_encoder_decoder
                          else torch.cat([steered_input_ids, ref_output_ids[:, :t]], dim=1))
                logits[:, t, :] = stack(prefix, logits[:, t, :])
        return logits

    def _resolve_generate_source(
            self,
            inputs: Any,
            text: Any,
            messages: Any,
            input_ids: Any,
    ) -> tuple[Literal["text", "messages", "tokens"], Any]:
        """Select the single prompt source and its modality.

        Exactly one of positional `inputs`, `text=`, `messages=`, or `input_ids=` may be provided.
        Positional input is a convenience for text prompts (`str` or a `list` whose every element is
        a `str`) and routes to text; any other positional shape raises (E12). Because the check is a
        total `all(...)` over the list, a mixed list such as `["a", {"role": ...}]` fails here rather
        than downstream.

        Returns:
            tuple[kind, payload] where `kind` is `"text"`, `"messages"`, or `"tokens"` and `payload`
            is the value handed to the matching resolver.

        Raises:
            TypeError: If no source or more than one source is provided (E1/E2), or a positional
                input is neither a `str` nor a `list[str]` (E12).
        """
        provided = [
            name for name, value in (
                ("inputs", inputs), ("text", text), ("messages", messages), ("input_ids", input_ids),
            ) if value is not None
        ]
        if len(provided) == 0:
            raise TypeError(
                "generate() requires a prompt: pass positional text, or exactly one of text=, "
                "messages=, input_ids=."
            )
        if len(provided) > 1:
            names = ", ".join(provided)
            raise TypeError(
                f"generate() received multiple prompt sources ({names}); pass exactly one of "
                "positional inputs, text=, messages=, input_ids=."
            )

        if text is not None:
            return "text", text
        if messages is not None:
            return "messages", messages
        if input_ids is not None:
            return "tokens", input_ids

        # positional inputs: text convenience only
        if isinstance(inputs, str) or (
            isinstance(inputs, list) and all(isinstance(element, str) for element in inputs)
        ):
            return "text", inputs
        raise TypeError(
            "positional input to generate() must be a str or list of str; pass messages=... "
            "for chat or input_ids=... for token input."
        )

    def _resolve_text_prompt(self, text: Any) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
        """Validate and tokenize a text prompt (design §4.3.1).

        Args:
            text: A `str` (single) or a `list`/`tuple` whose elements are all `str` (batch).

        Returns:
            tuple[input_ids, attention_mask, is_single].

        Raises:
            TypeError: If `text` is a sequence containing a non-`str` element (E3).
            ValueError: If `text` is an empty sequence (E4).
        """
        is_single = isinstance(text, str)
        if is_single:
            normalized = [text]
        else:
            normalized = list(text)
            if len(normalized) == 0:
                raise ValueError("text= received an empty sequence.")
            for index, element in enumerate(normalized):
                if not isinstance(element, str):
                    raise TypeError(
                        f"text= must be a str or a sequence of str; element {index} is "
                        f"{type(element).__name__}."
                    )

        self._warned_tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
            self.input_controls, self._warned_tensor_with_adapt_messages
        )
        tokenized = self.tokenizer(normalized, return_tensors="pt", padding=True)
        return tokenized["input_ids"], tokenized.get("attention_mask"), is_single

    def _resolve_messages_prompt(
            self,
            messages: Any,
            runtime_kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor | None, set[int], bool]:
        """Validate a chat prompt, then adapt and chat-template tokenize it (design §4.3.2).

        Accepts one conversation (a sequence of mappings) or a batch (a sequence of sequences of
        mappings). Message elements are validated as `collections.abc.Mapping`; role/content schema
        remains the responsibility of `apply_chat_template`.

        Args:
            messages: One conversation or a batch of conversations.
            runtime_kwargs: Per-call parameters forwarded to `adapt_messages`.

        Returns:
            tuple[input_ids, attention_mask, message_handled, is_single], where `message_handled`
            holds `id()`s of controls that adapted at message level.

        Raises:
            ValueError: If the conversation or batch is empty (E5).
            TypeError: If a batch inner element is not a mapping (E6) or the outer sequence mixes
                element kinds (E7).
        """
        outer = list(messages)
        if len(outer) == 0:
            raise ValueError("messages= received an empty conversation or batch.")

        if all(isinstance(element, Mapping) for element in outer):
            is_single = True
            normalized = [list(outer)]
        elif all(isinstance(element, (list, tuple)) for element in outer):
            is_single = False
            normalized = []
            for i, chat in enumerate(outer):
                chat = list(chat)
                if len(chat) == 0:
                    raise ValueError("messages= received an empty conversation or batch.")
                for j, message in enumerate(chat):
                    if not isinstance(message, Mapping):
                        raise TypeError(
                            f"messages[{i}][{j}] must be a mapping (one chat message); got "
                            f"{type(message).__name__}."
                        )
                normalized.append(chat)
        else:
            raise TypeError(
                "messages= must be one conversation (a sequence of mappings) or a batch (a sequence "
                "of sequences of mappings); got mixed element types at the outer level."
            )

        input_ids, attention_mask, message_handled = apply_adapt_messages_and_tokenize(
            self.input_controls, self.tokenizer, normalized, runtime_kwargs
        )
        return input_ids, attention_mask, message_handled, is_single

    def _resolve_token_prompt(
            self,
            input_ids: Any,
            attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
        """Validate a token prompt (tokens only; design §4.3.3).

        Args:
            input_ids: A 1-D/2-D `torch.Tensor`, a `list[int]`, or a `list[list[int]]`.
            attention_mask: Optional mask, passed through unchanged.

        Returns:
            tuple[input_ids, attention_mask, is_single].

        Raises:
            ValueError: If a tensor is neither 1-D nor 2-D (E8), or nested lists are ragged (E9).
            TypeError: If the value is not a token tensor or integer list (E10).
        """
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 1:
                resolved, is_single = input_ids.unsqueeze(0), True
            elif input_ids.ndim == 2:
                resolved, is_single = input_ids, False
            else:
                raise ValueError(f"input_ids tensor must be 1-D or 2-D; got {input_ids.ndim}-D.")
        elif isinstance(input_ids, list) and input_ids and all(isinstance(x, int) for x in input_ids):
            resolved, is_single = torch.tensor([input_ids], dtype=torch.long), True
        elif (
            isinstance(input_ids, list) and input_ids
            and all(isinstance(row, list) and row and all(isinstance(x, int) for x in row) for row in input_ids)
        ):
            try:
                resolved = torch.tensor(input_ids, dtype=torch.long)
            except ValueError as exception:
                raise ValueError(
                    "input_ids= nested lists must be rectangular (equal-length rows)."
                ) from exception
            is_single = False
        else:
            raise TypeError(
                f"input_ids= accepts a 1-D/2-D integer tensor, list[int], or list[list[int]]; got "
                f"{type(input_ids).__name__}. For text prompts use text= or positional input; for "
                "chat use messages=."
            )

        self._warned_tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
            self.input_controls, self._warned_tensor_with_adapt_messages
        )
        return resolved, attention_mask, is_single

    @overload
    def generate(
            self,
            inputs: str,
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> str: ...
    @overload
    def generate(
            self,
            inputs: list[str],
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> list[str]: ...
    @overload
    def generate(
            self,
            *,
            text: str,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> str: ...
    @overload
    def generate(
            self,
            *,
            text: Sequence[str],
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> list[str]: ...
    @overload
    def generate(
            self,
            *,
            messages: Sequence[Mapping],
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> str: ...
    @overload
    def generate(
            self,
            *,
            messages: Sequence[Sequence[Mapping]],
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> list[str]: ...
    @overload
    def generate(
            self,
            *,
            input_ids: torch.Tensor | list[int] | list[list[int]],
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> torch.Tensor: ...
    @overload
    def generate(
            self,
            inputs: Any = ...,
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            *,
            text: Any = ...,
            messages: Any = ...,
            input_ids: Any = ...,
            return_output: Literal[True],
            **gen_kwargs: Any,
    ) -> Output | list[Output]: ...

    def generate(
            self,
            inputs: str | list[str] | None = None,
            attention_mask: torch.Tensor | None = None,
            runtime_kwargs: dict | None = None,
            return_output: bool = False,
            *,
            text: str | Sequence[str] | None = None,
            messages: Sequence[Mapping] | Sequence[Sequence[Mapping]] | None = None,
            input_ids: torch.Tensor | list[int] | list[list[int]] | None = None,
            **gen_kwargs,
    ) -> str | list[str] | torch.Tensor | Output | list[Output]:
        """Generate with steering across text, chat, and token prompts.

        The prompt source is declared by keyword, with exactly one source per call:

        | Source | Tokenization | Default return type |
        | --- | --- | --- |
        | `text=` (`str`) | plain text | `str` |
        | `text=` (`Sequence[str]`) | batched plain text | `list[str]` |
        | `messages=` (one conversation) | `apply_chat_template` | `str` |
        | `messages=` (batch of conversations) | batched `apply_chat_template` | `list[str]` |
        | `input_ids=` (1-D tokens) | already tokenized; passed through | `torch.Tensor` |
        | `input_ids=` (2-D tokens) | already tokenized; passed through | `torch.Tensor` |

        Positional `str`/`list[str]` is accepted as a convenience for text prompts and behaves like
        `text=`; any other positional shape raises `TypeError`. With `return_output=True`, the return
        is always `Output` (single) or `list[Output]` (batched) regardless of source.

        Unlike `model.generate`, the returned token ids exclude the prompt by default. Do not slice
        the result by prompt length, since that discards generated tokens. Pass
        `return_full_sequence=True` to get HF-style prompt+continuation output.

        `attention_mask` is valid only with token input (`input_ids=`); it is derived automatically
        for `text=` and `messages=`, and passing it with either raises `TypeError`.
        The `adapt_messages` hook fires only on chat input; text and token input go straight to the
        token-level `adapt(input_ids, ...)` chain. For chat input, each input control whose
        `adapt_messages` returns a non-None result is not additionally run at token level, so every
        input control is applied exactly once per call.

        Args:
            inputs: Positional convenience for text prompts (`str` or `list[str]`), behaving like
                `text=`. Any other positional shape raises `TypeError`; use the keywords below.
            attention_mask: Attention mask, valid only with `input_ids=`.
            runtime_kwargs: Per-generation parameters for controls (e.g., `{"substrings": [...]}`).
            return_output: If True, return one or more `Output` objects instead of decoded text /
                token IDs.
            text: Text prompt as a `str` or a sequence of `str`.
            messages: Chat prompt as one conversation (a sequence of mappings) or a batch (a sequence
                of sequences of mappings).
            input_ids: Token prompt as a 1-D/2-D integer tensor, `list[int]`, or `list[list[int]]`.
            **gen_kwargs: Generation parameters passed to `model.generate()`. May include
                `return_full_sequence: bool` to include the prompt in the returned token IDs.

        Returns:
            See dispatch table above.

        Raises:
            RuntimeError: If `steer()` has not yet been called.
            TypeError: If no prompt source or more than one is provided, if a source fails
                validation, or if `attention_mask` is paired with `text=`/`messages=`.
            ValueError: If a token tensor is not 1-D/2-D, nested token lists are ragged, or a text/
                chat sequence is empty.
        """
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.generate()`.")

        runtime_kwargs = runtime_kwargs or {}
        return_full_sequence = bool(gen_kwargs.pop("return_full_sequence", False))

        kind, payload = self._resolve_generate_source(inputs, text, messages, input_ids)

        # attention_mask pairing
        if attention_mask is not None and kind != "tokens":
            raise TypeError(
                "attention_mask is only valid with token input (input_ids=); it is derived "
                "automatically for text= and messages=."
            )

        # resolve the prompt tensors per modality
        message_handled: set[int] = set()
        if kind == "text":
            prompt_input_ids, prompt_attention_mask, is_single = self._resolve_text_prompt(payload)
        elif kind == "messages":
            prompt_input_ids, prompt_attention_mask, message_handled, is_single = (
                self._resolve_messages_prompt(payload, runtime_kwargs)
            )
        else:  # tokens
            prompt_input_ids, prompt_attention_mask, is_single = self._resolve_token_prompt(
                payload, attention_mask
            )

        return self._execute_generation(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            message_handled=message_handled,
            decode_text=(kind != "tokens"),
            is_single=is_single,
            runtime_kwargs=runtime_kwargs,
            return_output=return_output,
            return_full_sequence=return_full_sequence,
            gen_kwargs=gen_kwargs,
        )

    def _execute_generation(
            self,
            prompt_input_ids: torch.Tensor,
            prompt_attention_mask: torch.Tensor | None,
            message_handled: frozenset[int] | set[int],
            *,
            decode_text: bool,
            is_single: bool,
            runtime_kwargs: dict,
            return_output: bool,
            return_full_sequence: bool,
            gen_kwargs: dict,
    ) -> str | list[str] | torch.Tensor | Output | list[Output]:
        """Run the shared generation tail from prompt tensors through the shaped return.

        Applies the token-level input-control chain, configures state hooks, composes the output
        processor and stopping-criteria stacks, drives decoding under the state-control context, and
        shapes the return. The prompt slice is removed by default (`return_full_sequence=False`
        returns continuation tokens only).

        Args:
            prompt_input_ids: Prompt token IDs [seq_len] or [batch, seq_len] before the token-level
                `adapt` chain.
            prompt_attention_mask: Attention mask matching `prompt_input_ids`, or None to derive it.
            message_handled: `id()`s of input controls already handled at message level; their
                token-level `adapt` is skipped.
            decode_text: If True, decode the returned IDs to `str`/`list[str]`; if False, return the
                token tensor.
            is_single: If True, the caller passed a single (non-batched) prompt, so unwrap the batch
                dimension in the return.
            runtime_kwargs: Per-call parameters forwarded to input, state, and output controls.
            return_output: If True, return `Output` (single) or `list[Output]` (batched) regardless
                of `decode_text`.
            return_full_sequence: If True, include the prompt in the returned token IDs.
            gen_kwargs: Generation parameters forwarded to the decoding driver.

        Returns:
            `str` or `list[str]` when `decode_text` is True, `torch.Tensor` when False, or
            `Output`/`list[Output]` when `return_output` is True.
        """
        # input controls (token-level adapt chain) + normalize
        steered_input_ids, steered_attention_mask = self._prepare_inputs(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            runtime_kwargs=runtime_kwargs,
            message_handled=frozenset(message_handled),
        )

        # state controls
        self._setup_state_controls(
            steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask, **gen_kwargs
        )

        # output controls: compose the processor and criteria stacks (list order), then drive the loop
        logits_processors, stopping_criteria = self._compose_stacks(
            steered_input_ids, runtime_kwargs, steered_attention_mask, gen_kwargs
        )
        decoding_driver = self._resolve_decoding_driver()

        with contextlib.ExitStack() as stack:  # hooks live only for duration of decoding
            for state_control in self.state_controls:
                stack.enter_context(state_control)
            full_output_ids = decoding_driver.decode(
                input_ids=steered_input_ids,
                attention_mask=steered_attention_mask,
                model=self.model,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                runtime_kwargs=runtime_kwargs,
                **gen_kwargs,
            )

        prompt_len = steered_input_ids.size(1)
        new_tokens = full_output_ids[:, prompt_len:]
        returned_ids = full_output_ids if return_full_sequence else new_tokens

        # build Output (always — used for return_output=True and to expose adapted_input_ids for introspection)
        finish_reason = infer_finish_reason(new_tokens, gen_kwargs)
        output = Output(
            output_ids=new_tokens,
            adapted_input_ids=steered_input_ids,
            runtime_kwargs=runtime_kwargs or None,
            finish_reason=finish_reason,
            metadata=None,
        )

        # shape return per modality + flag
        if return_output:
            if is_single:
                return output
            return [
                Output(
                    output_ids=output.output_ids[i:i + 1],
                    adapted_input_ids=output.adapted_input_ids[i:i + 1] if output.adapted_input_ids is not None else None,
                    runtime_kwargs=output.runtime_kwargs,
                    finish_reason=output.finish_reason,
                    metadata=output.metadata,
                )
                for i in range(output.output_ids.size(0))
            ]

        if not decode_text:
            return returned_ids

        # text / chat → decode
        decoded = self.tokenizer.batch_decode(
            returned_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if is_single:
            return decoded[0]
        return decoded

    def compute_logprobs(
            self,
            input_ids: list[int] | torch.LongTensor,
            attention_mask: torch.Tensor | None = None,
            ref_output_ids: list[int] | torch.LongTensor = None,
            runtime_kwargs: dict | None = None,
            **forward_kwargs: Any,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities of ref_output_ids with structural, input, state, and output steering
        controls applied.

        Step-level output controls' logits processors are part of the steered next-token distribution, so they are
        applied here position-by-position (teacher-forced), for every enabled output control whose
        `include_in_scoring` is True. The exclusive mechanism has no scoring analogue: decoding drivers are not
        invoked and stopping criteria are not applied. A control opts out of scoring by setting
        `include_in_scoring=False` (e.g. processors too expensive to evaluate per reference position).

        Uses teacher forcing to compute log P(ref_t | steered_input, ref_1, ..., ref_{t-1}) for each
        token in the reference sequence.

        When all pipeline controls support batching, a single batched forward pass is used (inputs are left-padded
        internally for correct positional alignment). Otherwise, falls back to sequential per-item processing.

        Args:
            input_ids: Input token IDs as list or tensor [seq_len] or [batch, seq_len]
            attention_mask: Optional attention mask matching input_ids shape
            ref_output_ids: Reference tokens to score [ref_len] or [batch, ref_len]
            runtime_kwargs: Per-call parameters for controls (e.g., {"substrings": [...]})
            **forward_kwargs: Additional arguments passed to model forward pass

        Returns:
            torch.Tensor: Log probabilities of shape [batch, ref_len] for decoder-only models,
                or [batch, ref_len - 1] for encoder-decoder models (excludes first decoder token)

        Raises:
            RuntimeError: If steer() has not been called
            ValueError: If ref_output_ids is None
        """
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.compute_logprobs()`.")
        if ref_output_ids is None:
            raise ValueError("`ref_output_ids` is required for `compute_logprobs()`.")

        runtime_kwargs = runtime_kwargs or {}
        device = self.model.device

        # normalize ref_output_ids
        if isinstance(ref_output_ids, list):
            ref_output_ids = torch.tensor(ref_output_ids, dtype=torch.long)
        if ref_output_ids.ndim == 1:
            ref_output_ids = ref_output_ids.unsqueeze(0)
        ref_output_ids = ref_output_ids.to(device)
        ref_len = ref_output_ids.size(1)

        is_encoder_decoder = getattr(self.model.config, "is_encoder_decoder", False)

        # batched path (all controls are batch-safe)
        if self.supports_batching:
            # input controls
            steered_input_ids, attention_mask = self._prepare_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                runtime_kwargs=runtime_kwargs,
            )
            batch_size = steered_input_ids.size(0)

            # broadcast single ref sequence across batch
            if ref_output_ids.size(0) == 1 and batch_size > 1:
                ref_output_ids = ref_output_ids.expand(batch_size, -1)

            if ref_len == 0:
                return torch.zeros((batch_size, 0), device=device, dtype=torch.float32)

            # left-pad for correct positional alignment in causal models; with right-padding, pad tokens between the
            # real input and the appended ref tokens corrupt positional encodings and the causal attention chain
            if not is_encoder_decoder:
                steered_input_ids, attention_mask = to_left_pad(steered_input_ids, attention_mask)

            # state controls
            self._setup_state_controls(
                steered_input_ids, runtime_kwargs, attention_mask=attention_mask, **forward_kwargs
            )

            # forward pass under state control context
            with contextlib.ExitStack() as stack:
                for state_control in self.state_controls:
                    stack.enter_context(state_control)
                with torch.no_grad():
                    if is_encoder_decoder:
                        outputs = self.model(
                            input_ids=steered_input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=ref_output_ids,
                            **forward_kwargs,
                        )
                        # predicts ref[t+1] from ref[0:t]; logits[:, t, :] -> ref[t+1]
                        # logits[:, :-1, :] aligns with targets ref[:, 1:]
                        logits = outputs.logits[:, :-1, :]
                        target_ids = ref_output_ids[:, 1:]
                    else:
                        # concatenate input + ref for causal teacher forcing
                        combined_ids = torch.cat([steered_input_ids, ref_output_ids], dim=1)
                        combined_mask = torch.cat([
                            attention_mask,
                            torch.ones(batch_size, ref_len, device=device, dtype=attention_mask.dtype),
                        ], dim=1)

                        outputs = self.model(
                            input_ids=combined_ids,
                            attention_mask=combined_mask,
                            **forward_kwargs,
                        )

                        # logits at [input_len - 1] predicts ref[0]
                        # logits at [input_len + ref_len - 2] predicts ref[ref_len - 1]
                        input_len = steered_input_ids.size(1)
                        logits = outputs.logits[:, input_len - 1: input_len + ref_len - 1, :]
                        target_ids = ref_output_ids

                # apply output-control scoring processors under the steered distribution
                logits = self._apply_scoring_processors(
                    logits, steered_input_ids, ref_output_ids, runtime_kwargs,
                    attention_mask, is_encoder_decoder, **forward_kwargs,
                )

                # compute logprobs
                logprobs = torch.log_softmax(logits, dim=-1)
                return logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        # sequential fallback (one or more controls do not support batching)

        # normalize input_ids to 2D for indexing
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(device)

        num_inputs = input_ids.size(0)

        # broadcast single ref sequence across batch
        if ref_output_ids.size(0) == 1 and num_inputs > 1:
            ref_output_ids = ref_output_ids.expand(num_inputs, -1)

        if ref_len == 0:
            return torch.zeros((num_inputs, 0), device=device, dtype=torch.float32)

        all_logprobs = []

        for i in range(num_inputs):
            single_input_ids = input_ids[i:i + 1]
            single_attention_mask = attention_mask[i:i + 1] if attention_mask is not None else None
            single_ref = ref_output_ids[i:i + 1]

            # input controls
            steered_input_ids, steered_attention_mask = self._prepare_inputs(
                input_ids=single_input_ids,
                attention_mask=single_attention_mask,
                runtime_kwargs=runtime_kwargs,
            )

            # state controls
            self._setup_state_controls(
                steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask, **forward_kwargs
            )

            # forward pass under state control context
            with contextlib.ExitStack() as stack:
                for state_control in self.state_controls:
                    stack.enter_context(state_control)
                with torch.no_grad():
                    if is_encoder_decoder:
                        outputs = self.model(
                            input_ids=steered_input_ids,
                            attention_mask=steered_attention_mask,
                            decoder_input_ids=single_ref,
                            **forward_kwargs,
                        )
                        logits = outputs.logits[:, :-1, :]
                        target_ids = single_ref[:, 1:]
                    else:
                        combined_ids = torch.cat([steered_input_ids, single_ref], dim=1)
                        combined_mask = torch.cat([
                            steered_attention_mask,
                            torch.ones(1, ref_len, device=device, dtype=steered_attention_mask.dtype),
                        ], dim=1)

                        outputs = self.model(
                            input_ids=combined_ids,
                            attention_mask=combined_mask,
                            **forward_kwargs,
                        )

                        input_len = steered_input_ids.size(1)
                        logits = outputs.logits[:, input_len - 1: input_len + ref_len - 1, :]
                        target_ids = single_ref

                # apply output-control scoring processors under the steered distribution
                logits = self._apply_scoring_processors(
                    logits, steered_input_ids, single_ref, runtime_kwargs,
                    steered_attention_mask, is_encoder_decoder, **forward_kwargs,
                )

                # compute logprobs
                logprobs = torch.log_softmax(logits, dim=-1)
                token_logprobs = logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            all_logprobs.append(token_logprobs)

        return torch.cat(all_logprobs, dim=0)
