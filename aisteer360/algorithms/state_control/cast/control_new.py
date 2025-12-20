from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.state_control.base import StateControl

from ..activation_common.estimators import fit_contrastive_directions
from ..activation_common.hook_sites import get_model_layer_list
from ..activation_common.specs import (
    Comparator,
    CompMode,
    ConditionSearchSpec,
    ContrastivePairs,
    VectorTrainSpec,
)
from ..activation_common.steering_vector import SteeringVector
from ..activation_common.token_scope import (
    compute_prompt_lens_from_input_ids,
    make_token_mask,
)
from .utils.condition_search import find_best_condition_point  # CAST-specific


@dataclass
class CASTArgs(BaseArgs):
    behavior_data: ContrastivePairs | None = None
    behavior_fit: VectorTrainSpec = field(default_factory=lambda: VectorTrainSpec(method="pca_pairwise", accumulate="suffix-only"))
    behavior_layer_ids: Sequence[int] = tuple(range(15,24))
    behavior_vector_strength: float = 1.0

    condition_data: ContrastivePairs | None = None
    condition_fit: VectorTrainSpec = field(default_factory=lambda: VectorTrainSpec(method="pca_pairwise", accumulate="all"))
    search: ConditionSearchSpec = field(default_factory=ConditionSearchSpec)
    condition_layer_ids: Sequence[int] | None = None
    condition_vector_threshold: float | None = None
    condition_comparator_threshold_is: Comparator = "larger"
    condition_threshold_comparison_mode: CompMode = "mean"

    apply_behavior_on_first_call: bool = True
    use_ooi_preventive_normalization: bool = False
    use_explained_variance: bool = False
    token_scope: str = "all"
    last_k: int | None = None
    compat_mode_ibm_cast: bool = True

class CAST(StateControl):
    Args = CASTArgs

    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    device = None

    _layer_names: list[str] | None = None
    _behavior: SteeringVector | None = None
    _condition: SteeringVector | None = None

    _cond_layers: dict[int,bool] | None = None
    _beh_layers: dict[int,bool] | None = None
    _state: dict[int,dict] | None = None

    _forward_calls: dict[int,int] = defaultdict(int)
    _cond_met: torch.BoolTensor | None = None
    _prompt_lens: torch.LongTensor | None = None

    def reset(self):
        self._forward_calls = defaultdict(int)
        self._cond_met = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **__) -> PreTrainedModel:
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = next(model.parameters()).device

        # train vectors (if not provided)
        self._behavior = self._behavior or (fit_contrastive_directions(model, self.tokenizer, self.behavior_data, self.behavior_fit)
                                            if self.behavior_data is not None else None)
        if (self.condition_layer_ids is not None or self.search.auto_find):
            self._condition = self._condition or (fit_contrastive_directions(model, self.tokenizer, self.condition_data, self.condition_fit)
                                                  if self.condition_data is not None else None)

        # search condition point
        if self._condition is not None and self.search.auto_find:
            lid, thr, direction, _ = find_best_condition_point(model, self.tokenizer, self._condition.directions,
                                                               self.condition_data, self.condition_fit, self.search)
            self.condition_layer_ids = [lid]
            self.condition_vector_threshold = thr
            self.condition_comparator_threshold_is = direction

        # layer bookkeeping
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names
        L = len(layer_names)
        cond = [False]*L; beh = [False]*L
        if self._condition is not None and self.condition_layer_ids is not None:
            for i in self.condition_layer_ids: cond[int(i)] = True
        for i in self.behavior_layer_ids: beh[int(i)] = True
        self._cond_layers = {i:v for i,v in enumerate(cond)}
        self._beh_layers  = {i:v for i,v in enumerate(beh)}

        # per-layer tensors
        self._state = {}
        for lid in range(L):
            st = dict(beh_vec=None, cond_proj=None, thr=self.condition_vector_threshold,
                      cmp=self.condition_comparator_threshold_is, mode=self.condition_threshold_comparison_mode)
            if beh[lid] and self._behavior is not None:
                v = self._behavior.directions.get(int(lid))
                if v is not None:
                    if self.use_explained_variance and self._behavior.explained_variances:
                        v = v * float(self._behavior.explained_variances.get(int(lid), 1.0))
                    st["beh_vec"] = (self.behavior_vector_strength * v).to(self.device, dtype=self.model.dtype)
            if cond[lid] and self._condition is not None:
                c = self._condition.directions[int(lid)].to(self.device, dtype=self.model.dtype)
                st["cond_proj"] = torch.outer(c, c) / (c @ c + 1e-8)
            self._state[int(lid)] = st

        return model

    def get_hooks(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **__) -> dict[str,list]:
        hooks = {"pre": [], "forward": [], "backward": []}
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        self._prompt_lens = compute_prompt_lens_from_input_ids(ids, pad_id)
        for lid, path in enumerate(self._layer_names):
            hooks["pre"].append({"module": path, "hook_func": partial(self._pre_hook, layer_id=int(lid))})
        return hooks

    def _pre_hook(self, module, input_args: Tuple, input_kwargs: dict, layer_id: int):
        h = input_args[0] if input_args else input_kwargs.get("hidden_states")
        if h is None: return input_args, input_kwargs

        self._forward_calls[layer_id] += 1
        B, T, _ = h.shape
        mask = make_token_mask(self.token_scope, prompt_lens=self._prompt_lens.to(h.device), seq_len=T, last_k=self.last_k)

        # condition (first pass at cond layer)
        if self._cond_layers.get(layer_id, False):
            self._eval_condition(h, layer_id)

        # behavior
        if self._beh_layers.get(layer_id, False):
            h = self._apply_behavior(h, layer_id, mask)

        if self.use_ooi_preventive_normalization and self._beh_layers.get(layer_id, False):
            h = self._ooi_norm(input_args, input_kwargs, h)

        if input_args:
            lst = list(input_args); lst[0] = h; return tuple(lst), input_kwargs
        input_kwargs["hidden_states"] = h; return input_args, input_kwargs

    @torch.no_grad()
    def _eval_condition(self, h: torch.Tensor, lid: int):
        if self._cond_met is not None: return
        st = self._state[lid]; P, thr = st.get("cond_proj"), st.get("thr")
        if P is None or thr is None:
            self._cond_met = torch.ones(h.size(0), dtype=torch.bool, device=h.device); return
        agg = h.mean(1) if st.get("mode") == "mean" else h[:, -1, :]
        proj = torch.tanh(torch.matmul(agg, P.T))
        sim = (agg * proj).sum(-1) / (agg.norm(dim=-1) * proj.norm(dim=-1) + 1e-8)
        if self.compat_mode_ibm_cast:
            cond = torch.where(st["cmp"]=="larger", sim < thr, sim >= thr)
        else:
            cond = torch.where(st["cmp"]=="larger", sim >= thr, sim <= thr)
        self._cond_met = cond.bool()

    def _apply_behavior(self, h: torch.Tensor, lid: int, mask: torch.BoolTensor) -> torch.Tensor:
        st = self._state[lid]; v = st.get("beh_vec");
        if v is None: return h
        if self._forward_calls[lid]==1 and not self.apply_behavior_on_first_call: return h
        gate = 1.0
        if self._cond_met is not None:
            gate = self._cond_met.view(-1,1,1).to(h.dtype)
        delta = gate * mask.unsqueeze(-1).to(h.dtype) * v.view(1,1,-1)
        return h + delta

    def _ooi_norm(self, input_args, input_kwargs, h_new: torch.Tensor) -> torch.Tensor:
        h_old = input_args[0] if input_args else input_kwargs.get("hidden_states")
        if h_old is None: return h_new
        on = h_old.norm(dim=-1, keepdim=True); nn = h_new.norm(dim=-1, keepdim=True)
        ratio = (nn/(on+1e-8)).max().item()
        if torch.isnan(h_new).any() or torch.isinf(h_new).any():
            raise ValueError("NaN/Inf during OOI normalization")
        if ratio > 1.0: h_new = h_new * (on / (nn + 1e-8))
        return h_new
