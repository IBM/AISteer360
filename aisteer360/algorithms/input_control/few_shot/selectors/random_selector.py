import random
from typing import Sequence

from aisteer360.algorithms.input_control.few_shot.selectors.base import Selector


class RandomSelector(Selector):
    """Selects examples uniformly at random from a pool for few-shot prompting."""

    def sample(self, pool: Sequence[dict], k: int, **_) -> list[dict]:
        """Select k examples uniformly at random from the pool.

        Args:
            pool: Available examples to select from
            k: Number of examples to select
            **_: Ignored (for compatibility with other selectors)

        Returns:
            List of randomly selected examples (up to min(k, len(pool)))
        """
        return random.sample(pool, min(k, len(pool)))
