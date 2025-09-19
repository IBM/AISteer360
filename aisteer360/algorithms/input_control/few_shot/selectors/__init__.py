"""
Example selectors for few-shot learning prompt adaptation.

This module provides different strategies for selecting examples from pools during few-shot prompting. Selectors
determine which examples are passed as demonstrations to the model.

Available selectors:

- `RandomSelector`: Randomly samples examples from the pool

"""
from .base import Selector
from .random_selector import RandomSelector

# __all__ = [
#     "Selector",
#     "RandomSelector",
# ]

SELECTOR_REGISTRY = {
    "random": RandomSelector,
}
