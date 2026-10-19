"""Finished-beam detection in `Frontier` over right-padded beams."""
import pytest
import torch

from steerability.algorithms.output_control.common.drivers.frontier import Frontier


def test_padded_beam_after_eos_is_finished():
    beams = torch.tensor([[1, 2, 5, 9, 0, 0], [1, 2, 5, 6, 7, 8]])
    frontier = Frontier(keep_k=2, eos_token_id=9, input_length=2, max_new_tokens=None, pad_token_id=0)
    step = frontier.keep(beams, [0.9, 0.1])
    assert step.finished_flags == [True, False]


def test_pad_equals_eos_flags_padded_beam_finished():
    beams = torch.tensor([[1, 2, 5, 9, 9, 9], [1, 2, 5, 6, 7, 8]])
    frontier = Frontier(keep_k=2, eos_token_id=9, input_length=2, max_new_tokens=None, pad_token_id=9)
    step = frontier.keep(beams, [0.9, 0.1])
    assert step.finished_flags == [True, False]


def test_budget_flag_fires_at_max_new_tokens():
    beams = torch.tensor([[1, 2, 5, 6, 7, 8]])
    frontier = Frontier(keep_k=1, eos_token_id=9, input_length=2, max_new_tokens=4, pad_token_id=0)
    step = frontier.keep(beams, [0.5])
    assert step.finished_flags == [True]


def test_padding_does_not_count_toward_budget():
    beams = torch.tensor([[1, 2, 5, 6, 0, 0]])
    frontier = Frontier(keep_k=1, eos_token_id=9, input_length=2, max_new_tokens=4, pad_token_id=0)
    step = frontier.keep(beams, [0.5])
    assert step.finished_flags == [False]


def test_list_eos_accepts_every_member():
    beams = torch.tensor([[1, 2, 5, 7, 0, 0], [1, 2, 5, 9, 0, 0], [1, 2, 5, 6, 7, 8]])
    frontier = Frontier(keep_k=3, eos_token_id=[7, 9], input_length=2, max_new_tokens=None, pad_token_id=0)
    step = frontier.keep(beams, [0.9, 0.8, 0.1])
    assert step.finished_flags == [True, True, False]


def test_best_ids_tracks_finished_beams_only():
    frontier = Frontier(keep_k=1, eos_token_id=9, input_length=2, max_new_tokens=None, pad_token_id=0)
    unfinished = torch.tensor([[1, 2, 5, 6]])
    finished = torch.tensor([[1, 2, 7, 9]])
    later_finished = torch.tensor([[1, 2, 8, 9]])

    frontier.keep(unfinished, [0.9])
    assert frontier.best_ids is None

    frontier.keep(finished, [0.4])
    assert frontier.best_score == pytest.approx(0.4)
    assert torch.equal(frontier.best_ids, finished[0])

    # a tie keeps the earlier finished beam
    frontier.keep(later_finished, [0.4])
    assert torch.equal(frontier.best_ids, finished[0])
