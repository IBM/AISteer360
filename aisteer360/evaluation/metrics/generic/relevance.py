from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric

_PROMPT = """\
You are an impartial grader.

Rate, on a scale from {lower_bound} (completely irrelevant) to {upper_bound} (perfectly relevant), how well the response
addresses the information need expressed in the prompt.

PROMPT:
{prompt}

RESPONSE:
{response}

What is your score?
"""


class Relevance(LLMJudgeMetric):
    """
    Judge relevance of a response to a prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            prompt_template=_PROMPT,
            scale=(1, 5),
            **kwargs,
        )
