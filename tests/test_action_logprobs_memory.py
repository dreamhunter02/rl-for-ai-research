"""The GRPO scorer must not materialize vocab logits for the whole prompt."""

import math
from types import SimpleNamespace

import torch

from financebench_onpolicy_grpo import action_logprobs


class TailOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, logits_to_keep=0):
        if logits_to_keep != 3:
            raise AssertionError(f"expected only 3 tail logits, got {logits_to_keep}")
        logits = torch.zeros((1, 3, 5))
        logits[0, 0, 3] = 2
        logits[0, 1, 4] = 2
        return SimpleNamespace(logits=logits)


def test_action_logprobs_scores_actions_with_tail_only_logits():
    actual = action_logprobs(TailOnlyModel(), [1, 2], [3, 4], 16)
    expected = 2 - math.log(math.exp(2) + 4)
    assert actual.shape == (2,)
    assert torch.allclose(actual, torch.tensor([expected, expected]))
