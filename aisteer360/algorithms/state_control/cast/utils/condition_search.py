from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ...activation_common.estimators import (
    _layerwise_tokenwise_hidden,
    _pool_over_spans,
    _select_spans,
    _tokenize,
)
from ...activation_common.specs import (
    ConditionSearchSpec,
    ContrastivePairs,
    VectorTrainSpec,
)


@torch.no_grad()
def _proj_sim(h: torch.Tensor, c: torch.Tensor) -> float:
    P = torch.outer(c, c) / (c @ c + 1e-8)
    proj = torch.tanh(P @ h)
    return float((h @ proj) / (h.norm() * proj.norm() + 1e-8))

@torch.no_grad()
def find_best_condition_point(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cond_dirs: dict[int, torch.Tensor],
    data: ContrastivePairs,
    fit: VectorTrainSpec,
    search: ConditionSearchSpec,
):
    pos_texts, neg_texts = data.positives, data.negatives
    device = next(model.parameters()).device
    enc_pos = _tokenize(tokenizer, pos_texts, device)
    enc_neg = _tokenize(tokenizer, neg_texts, device)
    hs_pos = _layerwise_tokenwise_hidden(model, enc_pos)
    hs_neg = _layerwise_tokenwise_hidden(model, enc_neg)
    enc_pos_cpu = {k:v.to("cpu") for k,v in enc_pos.items()}
    enc_neg_cpu = {k:v.to("cpu") for k,v in enc_neg.items()}
    spans_pos = _select_spans(enc_pos_cpu, None, fit.accumulate)
    spans_neg = _select_spans(enc_neg_cpu, None, fit.accumulate)

    if search.candidate_layers is not None:
        layers = search.candidate_layers
    else:
        start,end = search.layer_range or (1,len(hs_pos))
        layers = range(start, min(end, len(hs_pos)))

    best = dict(f1=-1.0, layer=None, thr=None, direction=None)
    thr_min, thr_max = search.threshold_range
    steps = max(1, int((thr_max - thr_min) / search.threshold_step))
    grid = torch.linspace(thr_min, thr_max, steps=steps) if steps>1 else torch.tensor([thr_min])

    for lid in layers:
        Hp = _pool_over_spans(hs_pos[int(lid)], spans_pos)
        Hn = _pool_over_spans(hs_neg[int(lid)], spans_neg)
        c = cond_dirs[int(lid)].to(Hp.dtype)

        sims_p = torch.tensor([_proj_sim(h, c) for h in Hp], dtype=torch.float32)
        sims_n = torch.tensor([_proj_sim(h, c) for h in Hn], dtype=torch.float32)

        for cmp in ("larger","smaller"):
            for thr in grid:
                if cmp == "larger":
                    yhat_p = sims_p >= thr; yhat_n = sims_n >= thr
                else:
                    yhat_p = sims_p <= thr; yhat_n = sims_n <= thr
                tp = int(yhat_p.sum().item()); fp = int(yhat_n.sum().item()); fn = len(sims_p) - tp
                prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
                f1 = 0.0 if prec + rec < 1e-8 else 2*prec*rec/(prec+rec)
                if f1 > best["f1"]:
                    best.update(f1=float(f1), layer=int(lid), thr=float(thr), direction=cmp)
    return best["layer"], best["thr"], best["direction"], best["f1"]
