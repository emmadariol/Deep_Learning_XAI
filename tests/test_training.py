"""Regression tests for frozen BatchNorm handling."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.model import set_frozen_batchnorm_eval


class FrozenBatchNormTests(unittest.TestCase):
    def test_frozen_batchnorm_is_forced_to_eval_mode(self) -> None:
        model = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 4, kernel_size=1),
            nn.BatchNorm2d(4),
        )
        for parameter in model[0].parameters():
            parameter.requires_grad = False
        model.train()

        self.assertTrue(model[0].training)
        self.assertTrue(model[2].training)
        set_frozen_batchnorm_eval(model)

        self.assertFalse(model[0].training)
        self.assertTrue(model[2].training)

    def test_frozen_batchnorm_running_stats_do_not_change_in_train_pass(self) -> None:
        model = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 4, kernel_size=1))
        for parameter in model[0].parameters():
            parameter.requires_grad = False
        before = model[0].running_mean.detach().clone()
        model.train()
        set_frozen_batchnorm_eval(model)
        _ = model(torch.randn(8, 3, 8, 8))
        after = model[0].running_mean.detach().clone()
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
