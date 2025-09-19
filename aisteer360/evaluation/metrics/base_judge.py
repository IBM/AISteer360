import math
import warnings
from typing import Any, Iterable, Sequence

import torch
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import OutputParserException
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    TextGenerationPipeline,
)

from aisteer360.evaluation.metrics.base import Metric


def build_structured_parser(scale):
    """
    Build a StructuredOutputParser and parsing function for rating predictions.

    Constructs a `StructuredOutputParser` configured with a single `ResponseSchema` that expects a float score within
    the specified scale range. It also returns a parsing function that extracts and validates the score from text,
    ensuring the result is clamped between the provided bounds.

    Args:
        scale (tuple[float, float]): A `(low, high)` tuple specifying the valid inclusive range for the score.
    """
    low, high = scale
    score_schema = ResponseSchema(
        name="score",
        description=f"A single float between {low} and {high} (inclusive) that rates the prediction."
    )
    output_parser = StructuredOutputParser.from_response_schemas([score_schema])

    def parse_fn(text: str, _: tuple[float, float]) -> float:
        """
        Parse and validate a score from text using the structured output parser.

        Returns:
            A tuple with elements:

                - StructuredOutputParser: The parser configured with the score schema.
                - Callable[[str, tuple[float, float]], float]: A function that takes a raw text response and the
                  `(low, high)` scale, extracts the score, converts it to a float, and clamps it within the valid range.

        Raises:
            ValueError: If the score cannot be parsed from the text.
        """
        try:
            score = float(output_parser.parse(text)["score"])
        except OutputParserException as e:
            raise ValueError(f"Could not parse score: {e}")
        return max(low, min(high, score))

    return output_parser, parse_fn


class LLMJudgeMetric(Metric):
    """Base class for LLM-as-a-judge evaluation metrics.

    Leverages a language model to evaluate the quality of generated text responses according to customized (natural
    language) criteria. The judge model evaluates each response (optionally with respect to an associated prompt and
    context) and returns numerical scores within a specified range. When multiple samples are generated per prompt (via
    num_return_sequences), scores are averaged to improve reliability.

    Subclasses should define their specific evaluation criteria by providing a `prompt_template` that instructs the
    judge model how to score responses. The template should use placeholders {response}, {lower_bound}, and
    {upper_bound} (and optionally {prompt} and {context}). Subclasses typically override `__init__()` to set their
    specific prompt template and scoring scale (e.g., see `metrics.generic.relevance`).

    Args:
        model_or_id (str | PreTrainedModel): HuggingFace model ID or loaded model instance to use as the judge.
            If string, the model will be loaded automatically.
        prompt_template (str): Template string for evaluation prompts. Should contain placeholders for {response},
            {lower_bound}, {upper_bound}, and optionally {prompt}, {context}.
            The formatted prompt will be passed to the judge model.
        tokenizer (Any | None): Tokenizer for the judge model. If None, will be loaded from the model ID.
            Required if passing a PreTrainedModel instance.
        device (str | None): Device for model inference ('cuda', 'mps', 'cpu').
            Defaults to GPU if available, otherwise CPU.
        scale (tuple[float, float]): Score range as (min, max) tuple. Scores outside this range will be clamped.
            Defaults to (1, 5).
        batch_size (int): Number of prompts to process simultaneously. Defaults to 8.
        max_retries (int): Maximum retry attempts when score parsing fails. Defaults to 5.
        gen_kwargs (dict[str, Any] | None): Generation parameters passed to the model.
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
    ):
        super().__init__()

        if isinstance(model_or_id, str):
            self.model = AutoModelForCausalLM.from_pretrained(model_or_id)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_id)
        else:  # model
            self.model = model_or_id
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_id.config._name_or_path)

        self.use_chat = hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device).eval()

        gen_kwargs = dict(gen_kwargs or {})
        gen_kwargs.setdefault("temperature", 0.0)
        gen_kwargs.setdefault("max_new_tokens", 30)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        self.num_return_sequences: int = int(gen_kwargs.pop("num_return_sequences", 1))
        self.model.generation_config = GenerationConfig(**gen_kwargs)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.scale = scale
        self.output_parser, self.parse_fn = build_structured_parser(scale)
        self.base_prompt_template = prompt_template.strip()
        self.format_instructions = self.output_parser.get_format_instructions()
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
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    @staticmethod
    def _batch_chunks(seq: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
        """Split a sequence into chunks of specified size.

        Args:
            seq (Sequence[Any]): The sequence to split into chunks.
            chunk_size (int): Maximum size of each chunk.

        Yields:
            Sequence[Any]: Chunks of the input sequence, each with at most chunk_size elements.
        """
        for i in range(0, len(seq), chunk_size):
            yield seq[i: i + chunk_size]

    def _score_with_retries(self, prompt: str) -> float:
        """Generate replies until parsing succeeds or maximum retries reached.

        Attempts to generate a response and parse it (using `parse_fn`) as a score.
        If parsing fails, retries up to `max_retries` times.
        If all attempts fail, raises a warning and returns `float('nan')`.

        Args:
            prompt (str): The formatted prompt to send to the model.

        Returns:
            float: The parsed score from the model's response, or `float('nan')` if parsing fails.
        """
        for attempt in range(self.max_retries + 1):
            reply_text = self.pipeline(
                prompt,
                clean_up_tokenization_spaces=True,
                return_full_text=False
            )[0]["generated_text"]

            try:
                return self.parse_fn(reply_text, self.scale)
            except Exception:
                if attempt == self.max_retries:
                    warnings.warn(
                        f"Failed to parse score after {self.max_retries + 1} attempts. "
                        "Returning float('nan') instead."
                    )
                    return float('nan')

    @torch.inference_mode()
    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float | list[float]]:
        """Compute LLM judge scores for a list of responses.

        Evaluates each response using the configured judge model and prompt template. Scores are averaged when multiple
        samples are generated per response (via `num_return_sequences`).

        Args:
            responses (list[str]): List of text responses to evaluate.
            prompts (list[str] | None): Optional list of prompts corresponding to each response.
                If provided, must be the same length as responses. These prompts can be
                referenced in the prompt_template using the {prompt} placeholder.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            Score statistics containing:

                - "mean_score": Overall average score across all responses
                - "scores": List of mean scores for each response (averaged across samples)
                - "raw_scores": List of lists containing all individual scores for each response

        Raises:
            AssertionError: If prompts is provided but has different length than responses.
        """

        if prompts is not None and len(prompts) != len(responses):
            raise AssertionError("`responses` and `prompts` must be the same length")

        # build prompts
        prompts_list: list[str] = []
        for i in range(len(responses)):
            fields: dict[str, str | float] = {
                "response": responses[i],
                "lower_bound": self.scale[0],
                "upper_bound": self.scale[1],
            }
            if prompts is not None:
                fields["prompt"] = prompts[i]

            prompt_core = self.base_prompt_template.format(**fields)
            prompt_formatted = self._wrap(prompt_core + "\n\n" + self.format_instructions)
            prompts_list.append(prompt_formatted)

        # generate
        prompt_scores: list[list[float]] = []
        for batch in self._batch_chunks(prompts_list, self.batch_size):
            outputs = self.pipeline(
                batch,
                num_return_sequences=self.num_return_sequences,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
            )

            for prompt, generations in zip(batch, outputs):
                generations = generations if isinstance(generations, list) else [generations]
                assert len(generations) == self.num_return_sequences

                scores = []
                for generation in generations:
                    reply_text = generation["generated_text"]
                    try:
                        score = self.parse_fn(reply_text, self.scale)
                    except Exception:
                        score = self._score_with_retries(prompt)
                    scores.append(score)

                prompt_scores.append(scores)

        mean_per_prompt = [sum(prompt_score) / len(prompt_score) for prompt_score in prompt_scores]
        corpus_mean = sum(mean_per_prompt) / len(mean_per_prompt)

        return {
            "mean_score": corpus_mean,  # overall average
            "scores": mean_per_prompt,  # one number per original prompt
            "raw_scores": prompt_scores  # n_samples scores per prompt
        }
