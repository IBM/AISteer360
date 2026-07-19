import json
import re
import warnings
from typing import Any, Callable, Iterable, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from aisteer360.evaluation.metrics.base import Metric
from aisteer360.utils.rendering import has_chat_template, render_messages


_FORMAT_INSTRUCTIONS = (
    'The output should be a markdown code snippet formatted in the following schema, '
    'including the leading and trailing "```json" and "```":\n\n'
    "```json\n"
    "{{\n"
    '\t"score": float  // A single float between {low} and {high} (inclusive) that rates the prediction.\n'
    "}}\n"
    "```"
)

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text, handling optional markdown code fences.

    Tries to find a fenced code block first; falls back to parsing the raw text.

    Args:
        text: Raw LLM response that should contain a JSON object.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    match = _CODE_BLOCK_RE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from response: {e}")
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object, got {type(result).__name__}")
    return result


def build_structured_parser(scale):
    """Build format instructions and a parsing function for rating predictions.

    Returns a lightweight parser that extracts a ``{"score": <float>}`` JSON object
    from the judge model's response and clamps the value to the given scale.

    Args:
        scale (tuple[float, float]): A ``(low, high)`` tuple specifying the valid inclusive range for the score.

    Returns:
        A tuple of ``(format_instructions: str, parse_fn)`` where *format_instructions*
        is the instruction string to append to the judge prompt and *parse_fn(text, scale)*
        returns a clamped float score.
    """
    low, high = scale
    format_instructions = _FORMAT_INSTRUCTIONS.format(low=low, high=high)

    def parse_fn(text: str, _: tuple[float, float]) -> float:
        """Parse and validate a score from text.

        Args:
            text: Raw text response from the judge model.
            _: Unused (scale is captured from the enclosing scope).

        Returns:
            A float score clamped to [low, high].

        Raises:
            ValueError: If the score cannot be parsed from the text.
        """
        parsed = _extract_json(text)
        if "score" not in parsed:
            raise ValueError(f"JSON missing 'score' key, got keys: {list(parsed.keys())}")
        score = float(parsed["score"])
        return max(low, min(high, score))

    return format_instructions, parse_fn


class LLMJudgeMetric(Metric):
    """Base class for LLM-as-a-judge evaluation metrics.

    Leverages a language model to evaluate the quality of generated text responses according to customized (natural
    language) criteria. The judge model evaluates each response (optionally with respect to an associated prompt and
    context) and returns numerical scores within a specified range. When multiple samples are generated per prompt (via
    ``num_return_sequences`` in ``gen_kwargs``), scores are averaged to improve reliability.

    Subclasses should define their specific evaluation criteria by providing a ``prompt_template`` that instructs the
    judge model how to score responses. The template should use placeholders ``{response}``, ``{lower_bound}``, and
    ``{upper_bound}`` (and optionally ``{prompt}``). Subclasses typically override ``__init__()`` to set their
    specific prompt template and scoring scale (e.g., see ``metrics.generic.relevance``).

    Args:
        model_or_id (str | PreTrainedModel): HuggingFace model ID or loaded model instance to use as the judge.
            If string, the model will be loaded automatically.
        prompt_template (str): Template string for evaluation prompts. Should contain placeholders for ``{response}``,
            ``{lower_bound}``, ``{upper_bound}``, and optionally ``{prompt}``.
            The formatted prompt will be passed to the judge model.
        tokenizer (Any | None): Tokenizer for the judge model. If None, will be loaded from the model ID.
            Required if passing a PreTrainedModel instance without an attached tokenizer hint.
        device (str | None): Device for model inference (e.g. ``'cuda'``, ``'mps'``, ``'cpu'``). Used only when
            loading a fresh model from a string id; ignored (with a warning) when ``model_or_id`` is a pre-loaded
            ``PreTrainedModel``.
        scale (tuple[float, float]): Score range as ``(min, max)`` tuple. Scores outside this range will be clamped.
            Defaults to ``(1, 5)``.
        batch_size (int): Number of prompts to process simultaneously. Defaults to 8.
        max_retries (int): Maximum retry attempts when score parsing fails. Only meaningful when sampling
            (``temperature > 0``). Defaults to 5.
        gen_kwargs (dict[str, Any] | None): Generation parameters forwarded to ``model.generate``.
        structured_output (bool): If True (default), append JSON format instructions to each prompt and parse the
            judge's response with a built-in JSON parser. If False, a custom ``parser`` must be supplied.
        parser (Callable[[str], float] | None): Custom parser mapping the judge's decoded response to a float.
            Required when ``structured_output=False``; forbidden when ``structured_output=True``.
    """

    def __init__(
        self,
        model_or_id: str | PreTrainedModel,
        prompt_template: str,
        tokenizer: Any | None = None,
        device: str | None = None,
        scale: tuple[float, float] = (1, 5),
        batch_size: int = 8,
        max_retries: int = 5,
        gen_kwargs: dict[str, Any] | None = None,
        structured_output: bool = True,
        parser: Callable[[str], float] | None = None,
    ):
        super().__init__()

        if isinstance(model_or_id, str):
            self.model = AutoModelForCausalLM.from_pretrained(model_or_id)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_id)
            self.device = device or (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            self.model.to(self.device).eval()
        else:
            self.model = model_or_id
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_id.config._name_or_path)
            if device is not None:
                warnings.warn(
                    "LLMJudgeMetric received both a pre-loaded model and an explicit `device`; "
                    "ignoring `device` and using the model's existing placement.",
                    UserWarning,
                )
            self.device = next(self.model.parameters()).device
            self.model.eval()

        self.use_chat = has_chat_template(self.tokenizer)

        gen_kwargs = dict(gen_kwargs or {})
        gen_kwargs.setdefault("temperature", 0.0)
        gen_kwargs.setdefault("max_new_tokens", 30)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        self.num_return_sequences: int = int(gen_kwargs.pop("num_return_sequences", 1))
        self.gen_kwargs = gen_kwargs

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if structured_output:
            if parser is not None:
                raise ValueError(
                    "Provide either `structured_output=True` (default) or a custom `parser`, not both. "
                    "When structured_output=True the built-in JSON parser is used."
                )
            self.format_instructions, self.parse_fn = build_structured_parser(scale)
        else:
            if parser is None:
                raise ValueError(
                    "structured_output=False requires a `parser` callable: (text: str) -> float."
                )
            self.format_instructions = ""
            self.parse_fn = lambda text, _scale, _p=parser: float(_p(text))

        temperature = self.gen_kwargs.get("temperature", 0.0)
        if temperature == 0.0 and self.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences > 1 requires temperature > 0; "
                "deterministic decoding produces identical samples."
            )

        self.scale = scale
        self.base_prompt_template = prompt_template.strip()
        self.batch_size = batch_size
        self.max_retries = max_retries

    def _wrap(self, prompt: str) -> str:
        """Wrap prompt with appropriate formatting for the model.

        Applies the chat template (if the model supports it) with the prompt as a user message.
        Otherwise, returns the prompt unchanged.

        Args:
            prompt (str): The user prompt.

        Returns:
            str: The formatted prompt.
        """
        if self.use_chat:
            messages = [{"role": "user", "content": prompt}]
            return render_messages(self.tokenizer, messages, add_generation_prompt=True)
        return prompt

    @staticmethod
    def _batch_chunks(seq: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
        """Split a sequence into chunks of specified size."""
        for i in range(0, len(seq), chunk_size):
            yield seq[i: i + chunk_size]

    def _generate_batch(self, prompts: list[str], num_return_sequences: int = 1) -> list[list[str]]:
        """Batched generation. Returns one list of decoded responses per input prompt."""
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=not self.use_chat,
            ).to(self.model.device)

            gen_kwargs = dict(self.gen_kwargs)
            if num_return_sequences > 1:
                gen_kwargs["num_return_sequences"] = num_return_sequences

            with torch.inference_mode():
                output_ids = self.model.generate(**encoded, **gen_kwargs)

            prompt_len = encoded["input_ids"].size(1)
            new_ids = output_ids[:, prompt_len:]
            decoded = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        finally:
            self.tokenizer.padding_side = original_padding_side

        return [
            decoded[i * num_return_sequences: (i + 1) * num_return_sequences]
            for i in range(len(prompts))
        ]

    def _retry_score(self, prompt: str) -> float:
        """Re-sample on parse failure. Only meaningful when temperature > 0."""
        temperature = self.gen_kwargs.get("temperature", 0.0)
        if temperature == 0.0:
            raise ValueError("Cannot retry under deterministic decoding (temperature=0).")
        for _ in range(self.max_retries):
            [generations] = self._generate_batch([prompt], num_return_sequences=1)
            try:
                return self.parse_fn(generations[0], self.scale)
            except Exception:
                continue
        warnings.warn(
            f"Failed to parse score after {self.max_retries} retries; returning float('nan').",
            UserWarning,
        )
        return float("nan")

    def score_rendered(self, rendered_prompts: list[str]) -> dict[str, Any]:
        """Run the judge LM on already-rendered prompts and parse scores.

        Used internally by ``compute()``. Bypasses prompt-template formatting and format-instructions
        injection — assumes the caller has done both already.

        Args:
            rendered_prompts: Prompts already formatted (template-substituted and chat-wrapped if applicable).

        Returns:
            Score statistics with keys:

                - ``"mean_score"``: Overall average score across all prompts.
                - ``"scores"``: List of mean scores for each prompt (averaged across samples).
                - ``"raw_scores"``: List of lists containing all individual scores per prompt.
        """
        prompt_scores: list[list[float]] = []
        for batch in self._batch_chunks(rendered_prompts, self.batch_size):
            grouped = self._generate_batch(list(batch), self.num_return_sequences)
            for prompt, generations in zip(batch, grouped):
                scores: list[float] = []
                for generation in generations:
                    try:
                        score = self.parse_fn(generation, self.scale)
                    except Exception as e:
                        if self.gen_kwargs.get("temperature", 0.0) == 0.0:
                            raise ValueError(
                                f"Failed to parse score under deterministic decoding. "
                                f"Raw response: {generation!r}. Original error: {e}"
                            ) from e
                        score = self._retry_score(prompt)
                    scores.append(score)
                prompt_scores.append(scores)

        mean_per_prompt = [sum(s) / len(s) for s in prompt_scores]
        corpus_mean = sum(mean_per_prompt) / len(mean_per_prompt) if mean_per_prompt else 0.0
        return {
            "mean_score": corpus_mean,
            "scores": mean_per_prompt,
            "raw_scores": prompt_scores,
        }

    @torch.inference_mode()
    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float | list[float]]:
        """Compute LLM judge scores for a list of responses.

        Evaluates each response using the configured judge model and prompt template. Scores are averaged when multiple
        samples are generated per response (via ``num_return_sequences``).

        Args:
            responses (list[str]): List of text responses to evaluate.
            prompts (list[str] | None): Optional list of prompts corresponding to each response.
                If provided, must be the same length as responses. These prompts can be
                referenced in the prompt_template using the ``{prompt}`` placeholder.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            Score statistics containing:

                - ``"mean_score"``: Overall average score across all responses.
                - ``"scores"``: List of mean scores for each response (averaged across samples).
                - ``"raw_scores"``: List of lists containing all individual scores for each response.

        Raises:
            AssertionError: If prompts is provided but has different length than responses.
        """
        if prompts is not None and len(prompts) != len(responses):
            raise AssertionError("`responses` and `prompts` must be the same length")

        rendered: list[str] = []
        for i in range(len(responses)):
            fields: dict[str, str | float] = {
                "response": responses[i],
                "lower_bound": self.scale[0],
                "upper_bound": self.scale[1],
            }
            if prompts is not None:
                fields["prompt"] = prompts[i]
            prompt_core = self.base_prompt_template.format(**fields)
            suffix = ("\n\n" + self.format_instructions) if self.format_instructions else ""
            rendered.append(self._wrap(prompt_core + suffix))

        return self.score_rendered(rendered)
