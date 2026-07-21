"""Gate utility modules."""
from .scores import (
    aggregate_condition_hidden,
    masked_mean,
    projected_cosine_similarity,
    projected_cosine_similarity_tensor,
    rank_one_projector,
)

__all__ = [
    "aggregate_condition_hidden",
    "masked_mean",
    "projected_cosine_similarity",
    "projected_cosine_similarity_tensor",
    "rank_one_projector",
]
