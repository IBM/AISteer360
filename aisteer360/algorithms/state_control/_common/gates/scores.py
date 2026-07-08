"""Score functions used by gates to evaluate condition signals."""
import torch


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
    projected = torch.tanh(projector @ hidden_state)
    sim = torch.dot(hidden_state, projected) / (
        hidden_state.norm() * projected.norm() + 1e-8
    )
    return float(sim.item())
