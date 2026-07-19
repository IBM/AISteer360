"""
Core steering pipeline for composing and applying multiple LLM control methods.
"""
import contextlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, overload

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

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
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.structural_control.base import StructuralControl

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SteeringPipeline:
    """Main steering pipeline for applying various control methods to Hugging Face causal language models.

    Enables application of structural, state, input, and output controls in a coordinated manner.
    Controls are applied in a fixed bottom-up order during steering, then used together during generation.

    Workflow:

    1. Instantiate with a base model checkpoint and/or control objects
    2. Call `steer()` once to apply all controls in order (structural → input → state → output)
    3. Use `generate()` for inference with steering applied (polymorphic across str / list[str] / chat / tensor)

    Args:
        model_name_or_path (str or pathlib.Path, optional): HuggingFace model hub name or local directory.
            Required when `lazy_init=False`. Ignored when `lazy_init=True` and the structural
            control returns a model.
        controls (Sequence[StructuralControl | StateControl | InputControl | OutputControl], optional):
            Controls for the steering pipeline. The state category accepts any number of controls
            (applied in list order); the input, structural, and output categories accept at most one
            each. Omitted categories fall back to no-op controls (see control base classes).
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
        ValueError: If multiple controls provided for same category or required arguments missing

    Note:

    - The state category accepts multiple controls; the other categories accept at most one each.
        Omitted categories use no-op defaults.
    - Controls with a `tokenizer` attribute will have it auto-injected if not already set

    State-control ordering contract:

    For the state category, list order in `controls` is the single, well-defined composition surface:
    list order = `steer()` order = hook registration order = execution order for hooks on the same
    module. PyTorch forward hooks chain (a later hook receives the previous hook's returned output;
    pre-hooks chain likewise on inputs), so "control A then B at layer 12" is well-defined and
    non-commuting combinations (e.g. ablate∘add vs. add∘ablate) are order-sensitive by design. A
    gated/condition-scoring control placed after another observes activations already edited at earlier
    layers by upstream list entries.
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

    structural_control: StructuralControl = field(init=False)
    input_control: InputControl = field(init=False)
    state_controls: list[StateControl] = field(init=False)
    output_control: OutputControl = field(init=False)

    _is_steered: bool = field(default=False, init=False, repr=False)
    _warned_tensor_with_adapt_messages: bool = field(default=False, init=False, repr=False)
    _warned_duplicate_bos: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:

        # sort/validate the supplied steering methods
        controls_merged = merge_controls(self.controls)
        self.structural_control = controls_merged["structural_control"]
        self.input_control = controls_merged["input_control"]
        self.state_controls = controls_merged["state_controls"]
        self.output_control = controls_merged["output_control"]

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
        controls_iter = (self.structural_control, self.input_control, *self.state_controls, self.output_control)
        for control in controls_iter:
            if hasattr(control, "tokenizer") and getattr(control, "tokenizer") is None:
                setattr(control, "tokenizer", self.tokenizer)

    @property
    def state_control(self) -> StateControl:
        """Deprecated: use `state_controls`. Returns the sole state control; raises if multiple."""
        warnings.warn("`state_control` is deprecated; use `state_controls`.", DeprecationWarning, stacklevel=2)
        if len(self.state_controls) != 1:
            raise RuntimeError(
                f"Pipeline has {len(self.state_controls)} state controls; use `state_controls`."
            )
        return self.state_controls[0]

    @property
    def supports_batching(self) -> bool:
        """Return True if all enabled controls in this pipeline are batch-safe.
        """
        controls = (
            self.structural_control,
            self.input_control,
            *self.state_controls,
            self.output_control,
        )
        return all(
            getattr(control, "supports_batching", False)
            for control in controls
            if getattr(control, "enabled", True)
        )

    def steer(self, **steer_kwargs) -> None:
        """Apply all steering controls to the model in place.

        Executes each control's steer() method in a fixed bottom-up order: structural -> input -> state -> output.
        This ensures that higher-level controls always see the final configured model from lower levels.

        If any control's steer() method returns a PreTrainedModel instance, it replaces the current model for subsequent
        controls.

        Args:
            **steer_kwargs: Keyword arguments passed to all control steer() methods

        Raises:
            RuntimeError: If called more than once or no model available after steering
        """
        if self._is_steered:
            return

        # steer each control (bottom-up order: structural -> input -> state -> output)
        for control in (self.structural_control, self.input_control, *self.state_controls, self.output_control):
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
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    repo or Path(getattr(self.structural_control.args, "out_path", "")),
                    trust_remote_code=self.trust_remote_code,
                )
                self.tokenizer = ensure_pad_token(self.tokenizer)

            except Exception as exception:
                raise RuntimeError("Failed to resolve tokenizer post‑steer.") from exception

        for control in (self.structural_control, self.input_control, *self.state_controls, self.output_control):
            if hasattr(control, "tokenizer") and getattr(control, "tokenizer", None) is None:
                setattr(control, "tokenizer", self.tokenizer)

        # return steered pipeline
        self._is_steered = True

    def _prepare_inputs(
            self,
            input_ids: list[int] | torch.LongTensor,
            attention_mask: torch.Tensor | None,
            runtime_kwargs: dict | None,
            skip_adapt: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply input control and normalize input tensors.

        Transforms the prompt via the input control's adapter and ensures both input_ids and attention_mask are
        properly shaped tensors on the correct device.

        Args:
            input_ids: Input token IDs as list or tensor [seq_len] or [batch, seq_len]
            attention_mask: Optional attention mask matching input_ids shape
            runtime_kwargs: Per-call parameters for input control
            skip_adapt: When True, skip the token-level `input_control.adapt()` call (used by the chat path
                when `adapt_messages` already performed the adaptation before tokenization, so the control is
                not applied twice to the same prompt).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (steered_input_ids, attention_mask), both as 2D tensors on model device
        """
        runtime_kwargs = runtime_kwargs or {}
        device = self.model.device

        # apply input control (unless message-level adaptation already ran for this call)
        if skip_adapt:
            steered_input_ids = input_ids
        else:
            steered_input_ids = self.input_control.adapt(
                input_ids,
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

    @staticmethod
    def _classify_inputs(
            inputs: Any,
    ) -> tuple[Literal["text", "chat", "tensor"], bool, Any]:
        """Classify the input modality and normalize to a batched form.

        Returns:
            tuple[modality, is_single, normalized] where:

                - `modality`: one of `"text"`, `"chat"`, or `"tensor"`.
                - `is_single`: True if the caller passed a single (non-batched) input.
                - `normalized`: `list[str]` for text, `list[list[dict]]` for chat, 2-D `torch.Tensor` for tensor.
        """
        if isinstance(inputs, str):
            return "text", True, [inputs]

        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 1:
                return "tensor", True, inputs.unsqueeze(0)
            if inputs.ndim == 2:
                return "tensor", False, inputs
            raise ValueError(f"Tensor input must be 1-D or 2-D; got {inputs.ndim}-D.")

        if isinstance(inputs, list):
            if len(inputs) == 0:
                raise ValueError("Empty input list.")
            first = inputs[0]
            if isinstance(first, str):
                return "text", False, list(inputs)
            if isinstance(first, dict):
                # one chat (list of messages)
                return "chat", True, [list(inputs)]
            if isinstance(first, list) and first and isinstance(first[0], dict):
                # batch of chats (list of list of messages)
                return "chat", False, [list(chat) for chat in inputs]
            if isinstance(first, int):
                # 1-D token id list
                return "tensor", True, torch.tensor([list(inputs)], dtype=torch.long)
            if isinstance(first, list) and first and isinstance(first[0], int):
                # 2-D token id list-of-lists
                return "tensor", False, torch.tensor([list(seq) for seq in inputs], dtype=torch.long)

        raise TypeError(f"Unsupported input type: {type(inputs).__name__}.")

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
            inputs: torch.Tensor,
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[False] = ...,
            **gen_kwargs: Any,
    ) -> torch.Tensor: ...
    @overload
    def generate(
            self,
            inputs: Any,
            attention_mask: torch.Tensor | None = ...,
            runtime_kwargs: dict | None = ...,
            return_output: Literal[True] = ...,
            **gen_kwargs: Any,
    ) -> Output | list[Output]: ...

    def generate(
            self,
            inputs: str | list[str] | list[dict] | list[list[dict]] | torch.Tensor | list[int] | list[list[int]] | None = None,
            attention_mask: torch.Tensor | None = None,
            runtime_kwargs: dict | None = None,
            return_output: bool = False,
            *,
            input_ids: Any = None,
            **gen_kwargs,
    ) -> str | list[str] | torch.Tensor | Output | list[Output]:
        """Polymorphic generation across text, chat, and tensor inputs.

        Dispatch table:

        | Input type | Tokenization | Default return type |
        | --- | --- | --- |
        | `str` | plain text | `str` |
        | `list[str]` | batched plain text | `list[str]` |
        | `list[dict]` (one chat) | `apply_chat_template` | `str` |
        | `list[list[dict]]` (batch of chats) | batched `apply_chat_template` | `list[str]` |
        | `torch.Tensor` | already tokenized; passed through | `torch.Tensor` |

        With `return_output=True`, the return is always `Output` (single) or `list[Output]` (batched), regardless
        of input modality.

        NOTE: unlike `model.generate`, the returned token ids EXCLUDE the prompt by default. Do not
        slice the result by prompt length — that discards generated tokens (pass
        `return_full_sequence=True` if you need HF-style prompt+continuation output).

        `attention_mask` is meaningful only for tensor inputs; it is ignored (with a warning) for text and chat
        inputs. The `adapt_messages` hook fires only on chat input; text and tensor inputs go straight to
        `adapt(input_ids, ...)`. For chat input, when `adapt_messages` returns a non-None result the token-level
        `adapt()` is not called for that generation — the input control is applied exactly once per call.

        Args:
            inputs: One of the modalities above.
            attention_mask: Optional, tensor-only.
            runtime_kwargs: Per-generation parameters for controls (e.g., `{"substrings": [...]}`).
            return_output: If True, return one or more `Output` objects instead of decoded text / token IDs.
            **gen_kwargs: Generation parameters passed to `model.generate()`. May include
                `return_full_sequence: bool` to include the prompt in the returned token IDs.

        Returns:
            See dispatch table above.

        Raises:
            RuntimeError: If `steer()` has not yet been called.
        """
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.generate()`.")

        # keyword alias: callers may pass `input_ids=` as the prompt
        if inputs is None and input_ids is not None:
            inputs = input_ids
        elif inputs is not None and input_ids is not None:
            raise TypeError("Pass either `inputs` or `input_ids`, not both.")
        if inputs is None:
            raise TypeError("`generate()` requires `inputs` (or the legacy `input_ids` keyword).")

        runtime_kwargs = runtime_kwargs or {}
        return_full_sequence = bool(gen_kwargs.pop("return_full_sequence", False))

        modality, is_single, normalized = self._classify_inputs(inputs)

        # resolve the prompt input_ids (and attention_mask) per modality
        message_level_handled = False
        if modality == "chat":
            if attention_mask is not None:
                warnings.warn(
                    "`attention_mask` is ignored for chat input; it is rebuilt after tokenization.",
                    UserWarning,
                )
            prompt_input_ids, prompt_attention_mask, message_level_handled = (
                apply_adapt_messages_and_tokenize(self.input_control, self.tokenizer, normalized, runtime_kwargs)
            )
        elif modality == "text":
            if attention_mask is not None:
                warnings.warn(
                    "`attention_mask` is ignored for text input; it is rebuilt after tokenization.",
                    UserWarning,
                )
            self._warned_tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
                self.input_control, self._warned_tensor_with_adapt_messages
            )
            tokenized = self.tokenizer(
                list(normalized),
                return_tensors="pt",
                padding=True,
            )
            prompt_input_ids = tokenized["input_ids"]
            prompt_attention_mask = tokenized.get("attention_mask")
        else:  # tensor
            self._warned_tensor_with_adapt_messages = warn_if_adapt_messages_bypassed(
                self.input_control, self._warned_tensor_with_adapt_messages
            )
            prompt_input_ids = normalized
            prompt_attention_mask = attention_mask

        # input control (tensor-level adapt) + normalize
        steered_input_ids, steered_attention_mask = self._prepare_inputs(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            runtime_kwargs=runtime_kwargs,
            skip_adapt=message_level_handled,
        )

        # state controls
        self._setup_state_controls(
            steered_input_ids, runtime_kwargs, attention_mask=steered_attention_mask, **gen_kwargs
        )

        # output control
        with contextlib.ExitStack() as stack:  # hooks live only for duration of decoding
            for state_control in self.state_controls:
                stack.enter_context(state_control)
            full_output_ids = self.output_control.generate(
                input_ids=steered_input_ids,
                attention_mask=steered_attention_mask,
                runtime_kwargs=runtime_kwargs,
                model=self.model,
                **gen_kwargs
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

        if modality == "tensor":
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
        """Compute per-token log-probabilities of ref_output_ids with structural, input, and state steering controls
        applied. Note that output controls are *not* applied since they concern scoring, not generation.

        The strategy below uses teacher forcing, computes log P(ref_t | steered_input, ref_1, ..., ref_{t-1}) for each
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
            # input control
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

            # input control
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

            # compute logprobs
            logprobs = torch.log_softmax(logits, dim=-1)
            token_logprobs = logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            all_logprobs.append(token_logprobs)

        return torch.cat(all_logprobs, dim=0)
