"""Score functions used by gates to evaluate condition signals."""
import torch

from ...specs import CompMode


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool `[N, T, H]` hidden states over real (non-pad) positions.

    Falls back to an unmasked mean over all positions when no attention mask is available.

    Args:
        hidden: Shape `[N, T, H]`.
        attention_mask: Shape `[N, T]` (1 for real tokens, 0 for pads), or None.

    Returns:
        Pooled tensor of shape `[N, H]`.
    """
    if attention_mask is None:
        return hidden.mean(dim=1)
    m = attention_mask.to(hidden.dtype).unsqueeze(-1)  # [N, T, 1]
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-8)


def rank_one_projector(direction: torch.Tensor) -> torch.Tensor:
    """Build the rank-one projector `cc^T / (c^T c)` for a direction.

    Args:
        direction: Shape `[H]`.

    Returns:
        Projection matrix of shape `[H, H]`.
    """
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D [H]; got shape {tuple(direction.shape)}.")
    c = direction.float()
    return torch.outer(c, c) / (c @ c + 1e-8)


@torch.no_grad()
def projected_cosine_similarity_tensor(
    hidden: torch.Tensor,
    projector: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity between rows of `hidden` and their projections, one score per row.

    Args:
        hidden: Shape `[..., H]`.
        projector: Shape `[H, H]` outer-product projection matrix.

    Returns:
        Scores of shape `[...]` (float32).
    """
    hidden = hidden.float()
    projector = projector.float()
    projected = torch.tanh(hidden @ projector)  # projector is symmetric
    numerator = (hidden * projected).sum(dim=-1)
    denominator = hidden.norm(dim=-1) * projected.norm(dim=-1) + 1e-8
    return numerator / denominator


@torch.no_grad()
def projected_cosine_similarity(
    hidden_state: torch.Tensor,
    projector: torch.Tensor,
) -> float:
    """Compute cosine similarity between a vector and its projection.

    This function projects the hidden state through the condition subspace
    projector, applies tanh, then computes cosine similarity with the original.
    The CAST method uses this scoring function.

    Args:
        hidden_state: Shape [H] - aggregated hidden state.
        projector: Shape [H, H] - outer-product projection matrix.

    Returns:
        Cosine similarity as a float.
    """
    score = projected_cosine_similarity_tensor(hidden_state.unsqueeze(0), projector)[0]
    return float(score.item())


@torch.no_grad()
def aggregate_condition_hidden(
    hidden: torch.Tensor,
    mode: CompMode,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate `[B, T, H]` hidden states to `[B, H]` using non-pad tokens only.

    Args:
        hidden: Shape `[B, T, H]`.
        mode: "mean" pools over all real tokens; "last" selects the last real token per row.
        attention_mask: Shape `[B, T]` (1 for real tokens, 0 for pads), or None. When None, "mean"
            averages all positions and "last" uses the final position.

    Returns:
        Aggregated tensor of shape `[B, H]`.

    Raises:
        ValueError: If a row has no real tokens, or the mode is unsupported.
    """
    if mode == "mean":
        return masked_mean(hidden, attention_mask)

    if mode != "last":
        raise ValueError(f"Unsupported condition comparison mode: {mode!r}.")

    batch_size = hidden.size(0)
    if attention_mask is None:
        return hidden[:, -1, :]

    mask = attention_mask.to(hidden.device).bool()
    if not mask.any(dim=1).all():
        raise ValueError("aggregate_condition_hidden received a row with no real tokens.")

    sequence_length = mask.size(1)
    positions = torch.arange(sequence_length, device=hidden.device).unsqueeze(0)
    last_positions = positions.masked_fill(~mask, -1).max(dim=1).values  # [B]
    return hidden[torch.arange(batch_size, device=hidden.device), last_positions]
