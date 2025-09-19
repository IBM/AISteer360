from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric

_PROMPT = """\
You are a careful fact-checker.

Considering only verifiable facts, rate the response’s factual accuracy with respect to the prompt on a scale from
{lower_bound} (completely incorrect) to {upper_bound} (fully correct).

PROMPT:
{prompt}

RESPONSE:
{response}

What is your score?
"""


class Factuality(LLMJudgeMetric):
    """
    Judge factual correctness of a response to a prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            prompt_template=_PROMPT,
            scale=(1, 5),
            **kwargs,
        )
