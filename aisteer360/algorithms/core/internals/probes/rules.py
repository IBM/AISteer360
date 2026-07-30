"""Predicates and ordered rules over named probe decisions.

A `ProbePredicate` is a boolean expression over probe names, built from `P(name)` leaves and the
operators `&` (and), `|` (or), and `~` (not), and evaluated per row against a mapping from probe
name to per-row decisions. `RoutingRules` holds an ordered list of `Rule`s with first-match-wins
semantics, evaluated independently per row, so one batched call can route each prompt to a
different rule.

Actions are opaque payloads: a `Rule` carries whatever the consumer interprets (e.g. a decoding
driver lowers actions to phase plans). This module depends only on `torch` and the standard
library.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


def _normalize_decisions(decisions: Mapping[str, torch.Tensor | bool]) -> tuple[dict[str, torch.Tensor], int]:
    """Validate a decisions mapping and normalize every value to a 1-D bool tensor.

    Each value must be a 1-D bool tensor of a common length `num_rows`, or a bare python bool,
    which is accepted only when `num_rows == 1` (the single-row scalar allowance).

    Args:
        decisions: Mapping from probe name to per-row decisions.

    Returns:
        Tuple of (normalized mapping, num_rows).

    Raises:
        ValueError: If `decisions` is empty, a tensor is not 1-D or not bool dtype, tensor
            lengths disagree, or a bare bool is mixed with multi-row tensors.
    """
    if not decisions:
        raise ValueError("decisions mapping is empty; at least one probe decision is required.")

    num_rows: int | None = None
    for name, value in decisions.items():
        if isinstance(value, bool):
            continue
        t = torch.as_tensor(value)
        if t.dtype != torch.bool:
            raise ValueError(
                f"Decision for probe '{name}' has dtype {t.dtype}; decisions must be bool tensors."
            )
        if t.ndim != 1:
            raise ValueError(
                f"Decision for probe '{name}' has shape {tuple(t.shape)}; decisions must be 1-D "
                f"per-row tensors of shape [num_rows]."
            )
        if num_rows is None:
            num_rows = t.numel()
        elif t.numel() != num_rows:
            raise ValueError(
                f"Decision for probe '{name}' has {t.numel()} rows but earlier probes have "
                f"{num_rows}; all probes must describe the same logical batch."
            )

    if num_rows is None:  # every value was a bare bool
        num_rows = 1

    normalized: dict[str, torch.Tensor] = {}
    for name, value in decisions.items():
        if isinstance(value, bool):
            if num_rows != 1:
                raise ValueError(
                    f"Decision for probe '{name}' is a bare bool but the batch has {num_rows} "
                    f"rows; bare bools are accepted only when num_rows == 1."
                )
            normalized[name] = torch.tensor([value], dtype=torch.bool)
        else:
            normalized[name] = torch.as_tensor(value)
    return normalized, num_rows


class ProbePredicate(ABC):
    """Boolean expression over named probe decisions, evaluated per row.

    Leaves are created with `P(name)`; composites are built with the operators `&` (and),
    `|` (or), and `~` (not). `evaluate()` takes a mapping from probe name to per-row decisions
    (1-D bool tensors of a common length, or a bare bool for single-row batches) and returns a
    bool tensor of shape `[num_rows]`. `repr()` renders the expression infix, e.g.
    ``(legal & ~advice)``.
    """

    def evaluate(self, decisions: Mapping[str, torch.Tensor | bool]) -> torch.Tensor:
        """Evaluate the predicate against per-row probe decisions.

        Args:
            decisions: Mapping from probe name to a 1-D bool tensor of shape `[num_rows]`
                (a bare bool is accepted only when `num_rows == 1`).

        Returns:
            Bool tensor of shape `[num_rows]`; True where the predicate holds for that row.

        Raises:
            KeyError: If the predicate references a probe name absent from `decisions`; the
                message lists the available names.
            ValueError: If the decision tensors are malformed (see `evaluate` requirements).
        """
        normalized, _ = _normalize_decisions(decisions)
        return self._eval(normalized)

    @abstractmethod
    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate against an already-normalized decisions mapping."""

    @abstractmethod
    def probe_names(self) -> set[str]:
        """The set of probe names this predicate references."""

    def __and__(self, other: "ProbePredicate") -> "ProbePredicate":
        if not isinstance(other, ProbePredicate):
            return NotImplemented
        return _And(self, other)

    def __or__(self, other: "ProbePredicate") -> "ProbePredicate":
        if not isinstance(other, ProbePredicate):
            return NotImplemented
        return _Or(self, other)

    def __invert__(self) -> "ProbePredicate":
        return _Not(self)


class _Probe(ProbePredicate):
    """Leaf predicate: the named probe's per-row decision."""

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError(f"Probe name must be a non-empty string; got {name!r}.")
        self.name = name

    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.name not in decisions:
            raise KeyError(
                f"Unknown probe name '{self.name}'; available probes: {sorted(decisions)}."
            )
        return decisions[self.name]

    def probe_names(self) -> set[str]:
        return {self.name}

    def __repr__(self) -> str:
        return self.name


class _And(ProbePredicate):
    def __init__(self, left: ProbePredicate, right: ProbePredicate):
        self.left = left
        self.right = right

    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.left._eval(decisions) & self.right._eval(decisions)

    def probe_names(self) -> set[str]:
        return self.left.probe_names() | self.right.probe_names()

    def __repr__(self) -> str:
        return f"({self.left!r} & {self.right!r})"


class _Or(ProbePredicate):
    def __init__(self, left: ProbePredicate, right: ProbePredicate):
        self.left = left
        self.right = right

    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.left._eval(decisions) | self.right._eval(decisions)

    def probe_names(self) -> set[str]:
        return self.left.probe_names() | self.right.probe_names()

    def __repr__(self) -> str:
        return f"({self.left!r} | {self.right!r})"


class _Not(ProbePredicate):
    def __init__(self, operand: ProbePredicate):
        self.operand = operand

    def _eval(self, decisions: dict[str, torch.Tensor]) -> torch.Tensor:
        return ~self.operand._eval(decisions)

    def probe_names(self) -> set[str]:
        return self.operand.probe_names()

    def __repr__(self) -> str:
        return f"~{self.operand!r}"


def P(name: str) -> ProbePredicate:
    """Leaf predicate over the named probe's decision.

    Args:
        name: The probe name, matching a key of the decisions mapping at evaluation time.

    Returns:
        A `ProbePredicate` that reads the named probe's per-row decision.
    """
    return _Probe(name)


@dataclass(frozen=True)
class Rule:
    """One named routing rule: a predicate and the action to take when it matches.

    Attributes:
        name: Rule name, unique within a `RoutingRules` set; keys diagnostics and per-call
            action overrides.
        when: The predicate that must hold for a row to match this rule.
        action: Opaque payload interpreted by the consumer (e.g. lowered to a phase plan by a
            decoding driver).
    """

    name: str
    when: ProbePredicate
    action: Any

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Rule name must be a non-empty string; got {self.name!r}.")
        if not isinstance(self.when, ProbePredicate):
            raise TypeError(
                f"Rule '{self.name}': `when` must be a ProbePredicate (build one with P(name) "
                f"and &, |, ~); got {type(self.when).__name__}."
            )


def _action_label(action: Any) -> str:
    """Compact display label for an action: `str(action)` when the type defines its own
    `__str__`, else the type name (`"None"` for a missing action)."""
    if action is None:
        return "None"
    if type(action).__str__ is not object.__str__:
        return str(action)
    return type(action).__name__


class RoutingRules:
    """Ordered rules with first-match-wins semantics, evaluated independently per row.

    Each rule pairs a `ProbePredicate` with an action payload. `route()` evaluates every rule's
    predicate over all rows, then assigns each row the first rule whose predicate holds; rows
    matching no rule fall to the default (returned as None, with `default_action` available to
    the consumer).

    Args:
        rules: Ordered rules; earlier rules take precedence. Names must be unique.
        default_action: Action payload for rows matching no rule. Opaque, like `Rule.action`.

    Raises:
        ValueError: If two rules share a name.
        TypeError: If an entry is not a `Rule`.
    """

    def __init__(self, rules: Sequence[Rule], default_action: Any = None):
        rules = tuple(rules)
        for rule in rules:
            if not isinstance(rule, Rule):
                raise TypeError(f"RoutingRules entries must be Rule instances; got {type(rule).__name__}.")
        seen: set[str] = set()
        for rule in rules:
            if rule.name in seen:
                raise ValueError(f"Duplicate rule name '{rule.name}'; rule names must be unique.")
            seen.add(rule.name)
        self.rules = rules
        self.default_action = default_action

    def route(self, decisions: Mapping[str, torch.Tensor | bool]) -> list[Rule | None]:
        """Match each row to its first satisfied rule.

        Every rule's predicate is evaluated once over all rows, then each row takes the first
        rule whose predicate holds for it. There is no batch-wide short-circuit; rows are routed
        independently.

        Args:
            decisions: Mapping from probe name to per-row decisions (see
                `ProbePredicate.evaluate`).

        Returns:
            One entry per row: the matched `Rule`, or None for rows matching no rule (the
            default route).

        Raises:
            KeyError: If a rule references a probe name absent from `decisions`.
            ValueError: If the decision tensors are malformed.
        """
        normalized, num_rows = _normalize_decisions(decisions)
        masks = [rule.when._eval(normalized) for rule in self.rules]
        routes: list[Rule | None] = []
        for row in range(num_rows):
            matched: Rule | None = None
            for rule, mask in zip(self.rules, masks):
                if bool(mask[row]):
                    matched = rule
                    break
            routes.append(matched)
        return routes

    def probe_names(self) -> set[str]:
        """The union of probe names referenced by all rules."""
        names: set[str] = set()
        for rule in self.rules:
            names |= rule.when.probe_names()
        return names

    def validate_names(self, available: set[str]) -> None:
        """Check that every referenced probe name exists among `available`.

        Args:
            available: The probe names the consumer provides at routing time.

        Raises:
            ValueError: If any rule references a probe absent from `available`; the message
                names the missing probes.
        """
        missing = sorted(self.probe_names() - set(available))
        if missing:
            raise ValueError(
                f"Routing rules reference unknown probe(s) {missing}; available probes: "
                f"{sorted(available)}."
            )

    def describe(self) -> str:
        """Render the rule set as a plain-text flowchart.

        One line per rule, in precedence order, followed by the default line. The action
        column uses `str(action)` when the action type defines its own `__str__`, else the
        type name.

        Returns:
            The multi-line flowchart string.
        """
        name_width = max((len(rule.name) for rule in self.rules), default=0)
        name_width = max(name_width, len("default"))
        pred_reprs = [repr(rule.when) for rule in self.rules]
        pred_width = max((len(p) for p in pred_reprs), default=0)
        index_width = len(str(len(self.rules))) if self.rules else 1
        body_width = index_width + 2 + name_width + 3 + 3 + pred_width + 3

        lines = ["RoutingRules"]
        for i, (rule, pred) in enumerate(zip(self.rules, pred_reprs), start=1):
            body = f"{i:>{index_width}}. {rule.name:<{name_width}}   if {pred:<{pred_width}}   "
            lines.append(f"├─ {body}-> {_action_label(rule.action)}")
        lines.append(f"└─ {'default':<{body_width}}-> {_action_label(self.default_action)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        rule_names = ", ".join(rule.name for rule in self.rules)
        return f"RoutingRules([{rule_names}], default_action={_action_label(self.default_action)})"
