"""
Core steering pipeline for composing and applying multiple LLM control methods.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from aisteer360.algorithms.core.steering_utils import ensure_pad_token, merge_controls
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.structural_control.base import StructuralControl


@dataclass(slots=True)
class SteeringPipeline:
    """Main steering pipeline for applying various control methods to Hugging Face causal language models.

    Enables application of structural, state, input, and output controls in a coordinated manner.
    Controls are applied in a fixed bottom-up order during steering, then used together during generation.

    Workflow:

    1. Instantiate with a base model checkpoint and/or control objects
    2. Call `steer()` once to apply all controls in order (structural → input → state → output)
    3. Use `generate()` or `generate_text()` for inference with steering applied

    Args:
        model_name_or_path (str or pathlib.Path, optional): HuggingFace model hub name or local directory.
            Required when `lazy_init=False`. Ignored when `lazy_init=True` and the structural
            control returns a model.
        controls (Sequence[StructuralControl | StateControl | InputControl | OutputControl], optional):
            Controls for the steering pipeline, max one control per category. Omitted categories
            fall back to no-op controls (see control base classes).
        tokenizer_name_or_path (str, optional): Tokenizer location. Defaults to `model_name_or_path`.
        device_map (str or dict[str, int], optional): Device map (passed to
            `transformers.AutoModelForCausalLM.from_pretrained`). Defaults to `"auto"`.
            Cannot be used together with `device` parameter.
        device (torch.device, str, optional): Device (passed to model's `.to()` method).
            When specified, `device_map` must remain at its default value of `"auto"`.
        hf_model_kwargs (dict, optional): Extra keyword arguments passed to
            `transformers.AutoModelForCausalLM.from_pretrained`.
        lazy_init (bool, optional): If `True`, defers loading the base model until `steer()` time.
            Useful when a `StructuralControl` will itself load or create the final weights
            (e.g., MergeKit). When `False`, the model is loaded during `SteeringPipeline`
            construction. Defaults to `False`.

    Raises:
        RuntimeError: If `generate()` is called before `steer()`
        ValueError: If multiple controls provided for same category or required arguments missing

    Note:

    - Maximum one control per category; omitted categories use no-op defaults
    - Controls with a `tokenizer` attribute will have it auto-injected if not already set
    """

    # construction args
    model_name_or_path: str | Path | None = None
    controls: Sequence[StructuralControl | StateControl | InputControl | OutputControl] = ()
    tokenizer_name_or_path: str | None = None
    device_map: str | dict[str, int] | int | torch.device | None = "auto"
    device: torch.device | str | None = None
    hf_model_kwargs: dict = field(default_factory=dict)
    lazy_init: bool = False

    # lazy‑filled fields
    model: PreTrainedModel | None = field(init=False, default=None)
    tokenizer: AutoTokenizer | None = field(init=False, default=None)

    structural_control: StructuralControl = field(init=False)
    input_control: InputControl = field(init=False)
    state_control: StateControl = field(init=False)
    output_control: OutputControl = field(init=False)

    _is_steered: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:

        # sort/validate the supplied steering methods
        controls_merged = merge_controls(self.controls)
        self.structural_control = controls_merged["structural_control"]
        self.input_control = controls_merged["input_control"]
        self.state_control = controls_merged["state_control"]
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
                trust_remote_code=True,
            )
            self.tokenizer = ensure_pad_token(self.tokenizer)
        else:
            if isinstance(self.tokenizer_name_or_path, (str, Path)):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    trust_remote_code=True
                )
                self.tokenizer = ensure_pad_token(self.tokenizer)

        # late‑inject tokenizer into controls that accept it
        controls_iter = (self.structural_control, self.input_control, self.state_control, self.output_control)
        for control in controls_iter:
            if hasattr(control, "tokenizer") and getattr(control, "tokenizer") is None:
                setattr(control, "tokenizer", self.tokenizer)

    @property
    def supports_batching(self) -> bool:
        """Return True if all enabled controls in this pipeline are batch-safe.
        """
        controls = (
            self.structural_control,
            self.input_control,
            self.state_control,
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
        for control in (self.structural_control, self.input_control, self.state_control, self.output_control):
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
                    trust_remote_code=True,
                )
                self.tokenizer = ensure_pad_token(self.tokenizer)

            except Exception as exception:
                raise RuntimeError("Failed to resolve tokenizer post‑steer.") from exception

        for control in (self.structural_control, self.input_control, self.state_control, self.output_control):
            if hasattr(control, "tokenizer") and getattr(control, "tokenizer", None) is None:
                setattr(control, "tokenizer", self.tokenizer)

        # return steered pipeline
        self._is_steered = True

    def _prepare_inputs(
            self,
            input_ids: list[int] | torch.LongTensor,
            attention_mask: torch.Tensor | None,
            runtime_kwargs: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply input control and normalize input tensors.

        Transforms the prompt via the input control's adapter and ensures both input_ids and attention_mask are
        properly shaped tensors on the correct device.

        Args:
            input_ids: Input token IDs as list or tensor [seq_len] or [batch, seq_len]
            attention_mask: Optional attention mask matching input_ids shape
            runtime_kwargs: Per-call parameters for input control

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (steered_input_ids, attention_mask), both as 2D tensors on model device
        """
        runtime_kwargs = runtime_kwargs or {}
        device = self.model.device

        # apply input control adapter
        adapter = self.input_control.get_prompt_adapter(runtime_kwargs)
        steered_input_ids = adapter(input_ids, runtime_kwargs)

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
                attention_mask = (steered_input_ids != self.tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(steered_input_ids, dtype=torch.long)

        attention_mask = attention_mask.to(dtype=steered_input_ids.dtype, device=device)

        return steered_input_ids, attention_mask

    def _setup_state_control(
            self,
            steered_input_ids: torch.Tensor,
            runtime_kwargs: dict | None,
            **kwargs,
    ) -> None:
        """Configure state control hooks for the current forward/generate call.

        Prepares the state control by computing hooks based on the (already transformed) input and setting up the model
        reference for the context manager.

        Args:
            steered_input_ids: Input token IDs after input control transformation
            runtime_kwargs: Per-call parameters for state control
            **kwargs: Additional arguments passed to get_hooks()
        """
        hooks = self.state_control.get_hooks(steered_input_ids, runtime_kwargs, **kwargs)
        self.state_control.set_hooks(hooks)
        self.state_control._model_ref = self.model
        self.state_control.reset()

    def generate(
            self,
            input_ids: list[int] | torch.LongTensor,
            attention_mask: torch.Tensor | None = None,
            runtime_kwargs: dict | None = None,
            **gen_kwargs
    ) -> torch.Tensor:
        """Generate text with all steering controls applied.

        Applies controls in sequence during generation:

        1. Input control adapts the prompt
        2. State control registers hooks for state control (e.g., activation steering)
        3. Output control handles the actual generation

        Args:
            input_ids: Token IDs as list or tensor (shape: [seq_len] or [batch, seq_len])
            attention_mask: Optional attention mask matching input_ids shape
            runtime_kwargs: Per-generation parameters for controls (e.g., {"substrings": [...]})
            **gen_kwargs: Generation parameters passed to `model.generate()`

        Returns:
            Generated token IDs (shape: [batch, generated_len])

        Raises:
            RuntimeError: If steer() has not yet been called
        """
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.generate()`.")

        runtime_kwargs = runtime_kwargs or {}
        return_full_sequence = bool(gen_kwargs.pop("return_full_sequence", False))

        # input control
        steered_input_ids, attention_mask = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            runtime_kwargs=runtime_kwargs,
        )

        # state control
        self._setup_state_control(steered_input_ids, runtime_kwargs, **gen_kwargs)

        # output control
        with self.state_control:  # hooks live only for duration of decoding
            output_ids = self.output_control.generate(
                input_ids=steered_input_ids,
                attention_mask=attention_mask,
                runtime_kwargs=runtime_kwargs,
                model=self.model,
                **gen_kwargs
            )

        if not return_full_sequence:
            output_ids = output_ids[:, steered_input_ids.size(1):]

        return output_ids

    def generate_text(self, *args, **kwargs) -> str | list[str]:
        """Generate text and decode to string(s).

        Convenience wrapper that calls generate() and decodes the output tokens.

        Args:
            *args: Arguments passed to generate()
            **kwargs: Keyword arguments passed to generate()

        Returns:
            Decoded text string (single prompt) or list of strings (batch)
        """
        ids = self.generate(*args, **kwargs)
        if ids.ndim == 1:
            return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

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

        # input control
        steered_input_ids, attention_mask = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            runtime_kwargs=runtime_kwargs,
        )

        # normalize ref_output_ids
        if isinstance(ref_output_ids, list):
            ref_output_ids = torch.tensor(ref_output_ids, dtype=torch.long)
        if ref_output_ids.ndim == 1:
            ref_output_ids = ref_output_ids.unsqueeze(0)
        ref_output_ids = ref_output_ids.to(device)

        batch_size = steered_input_ids.size(0)
        ref_len = ref_output_ids.size(1)

        # broadcast single ref sequence across batch
        if ref_output_ids.size(0) == 1 and batch_size > 1:
            ref_output_ids = ref_output_ids.expand(batch_size, -1)

        if ref_len == 0:
            return torch.zeros((batch_size, 0), device=device, dtype=torch.float32)

        # state control
        self._setup_state_control(steered_input_ids, runtime_kwargs, **forward_kwargs)

        # forward pass under state control context
        is_encoder_decoder = getattr(self.model.config, "is_encoder_decoder", False)

        with self.state_control:
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
        token_logprobs = logprobs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        return token_logprobs
