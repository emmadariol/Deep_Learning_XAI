"""Regression tests for Concept Bottleneck Model training safeguards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.bottleneck import CBMOutputs, load_backbone_checkpoint, run_cbm_epoch


class _TinyCBM(nn.Module):
    def __init__(self, initialization: str = "random") -> None:
        super().__init__()
        self.backbone_initialization = initialization
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        for parameter in self.backbone[0].parameters():
            parameter.requires_grad = False
        self.concept_head = nn.Linear(2, 2)
        self.class_head = nn.Linear(2, 2)

    def forward(self, images: torch.Tensor) -> CBMOutputs:
        features = self.backbone(images)
        concept_logits = self.concept_head(features)
        concept_probs = torch.sigmoid(concept_logits)
        return CBMOutputs(
            class_logits=self.class_head(concept_probs),
            concept_logits=concept_logits,
            concept_probs=concept_probs,
        )


class CBMSafeguardTests(unittest.TestCase):
    def test_frozen_backbone_batchnorm_statistics_do_not_change(self) -> None:
        model = _TinyCBM()
        dataset = TensorDataset(
            torch.randn(8, 3, 8, 8),
            torch.randint(0, 2, (8, 2)).float(),
            torch.randint(0, 2, (8,)),
        )
        loader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01,
        )
        before = model.backbone[0].running_mean.detach().clone()
        run_cbm_epoch(model, loader, torch.device("cpu"), optimizer=optimizer)
        after = model.backbone[0].running_mean.detach().clone()
        self.assertTrue(torch.equal(before, after))

    def test_missing_checkpoint_rejects_random_frozen_features(self) -> None:
        model = _TinyCBM(initialization="random")
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                load_backbone_checkpoint(model, missing, torch.device("cpu"))

    def test_missing_checkpoint_accepts_explicit_imagenet_fallback(self) -> None:
        model = _TinyCBM(initialization="imagenet")
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            source = load_backbone_checkpoint(
                model,
                missing,
                torch.device("cpu"),
                allow_imagenet_fallback=True,
            )
        self.assertEqual(source, "imagenet")


if __name__ == "__main__":
    unittest.main()
