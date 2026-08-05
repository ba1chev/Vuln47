import torch
import unittest
import numpy as np
import torch.nn.functional as F

from source.training.model_training.loss.focal_loss import FocalLoss
from source.training.model_training.early_stopping.early_stopper import EarlyStopper
from source.preprocessing.data_preprocessing.data_fetching.data_fetcher import DataFetcher
from source.training.model_training.model_evaluation.metrics_evaluator import best_f1_threshold


class FocalLossTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.tensor([1.0, 3.0])
        self.logits = torch.tensor([[3.0, -2.0], [0.0, 4.0]])
        self.target = torch.tensor([0, 1])

    def test_gamma_zero_matches_mean_of_weighted_ce(self):
        loss = FocalLoss(alpha=self.weight, gamma=0.0)(self.logits, self.target)
        ce_none = F.cross_entropy(self.logits, self.target, weight=self.weight, reduction="none")
        self.assertTrue(torch.allclose(loss, ce_none.mean()))

    def test_gamma_downweights_easy_examples(self):
        easy = FocalLoss(alpha=self.weight, gamma=2.0)(self.logits, self.target)
        base = FocalLoss(alpha=self.weight, gamma=0.0)(self.logits, self.target)
        self.assertLess(float(easy), float(base))

    def test_returns_scalar(self):
        loss = FocalLoss(alpha=self.weight, gamma=2.0)(self.logits, self.target)
        self.assertEqual(loss.dim(), 0)


class EarlyStopperTests(unittest.TestCase):
    def test_stops_after_patience_without_improvement(self):
        stopper = EarlyStopper(patience=2, mode="max")
        stopper.update(0.5)
        self.assertFalse(stopper.should_stop())
        stopper.update(0.4)
        self.assertFalse(stopper.should_stop())
        stopper.update(0.3)
        self.assertTrue(stopper.should_stop())

    def test_improvement_resets_the_counter(self):
        stopper = EarlyStopper(patience=2, mode="max")
        stopper.update(0.5)
        stopper.update(0.4)
        stopper.update(0.6)
        stopper.update(0.55)
        self.assertFalse(stopper.should_stop())
        self.assertEqual(stopper.best, 0.6)

    def test_min_mode_tracks_decreasing_metric(self):
        stopper = EarlyStopper(patience=1, mode="min")
        stopper.update(1.0)
        stopper.update(0.5)
        self.assertFalse(stopper.should_stop())
        stopper.update(0.7)
        self.assertTrue(stopper.should_stop())

    def test_state_dict_roundtrip(self):
        stopper = EarlyStopper(patience=3, mode="max")
        stopper.update(0.5)
        stopper.update(0.4)
        restored = EarlyStopper(patience=3, mode="max")
        restored.load_state_dict(stopper.state_dict())
        self.assertEqual(restored.best, stopper.best)
        self.assertEqual(
            restored.epochs_without_improvement, stopper.epochs_without_improvement
        )


class ThresholdTests(unittest.TestCase):
    def test_separable_scores_pick_a_splitting_threshold(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        t = best_f1_threshold(y, score)
        self.assertGreater(t, 0.2)
        self.assertLessEqual(t, 0.9)

    def test_degenerate_input_returns_default(self):
        y = np.array([0, 0])
        score = np.array([0.3, 0.3])
        self.assertIsInstance(best_f1_threshold(y, score), float)


class NormalizeCweTests(unittest.TestCase):
    def test_passes_through_a_list(self):
        self.assertEqual(DataFetcher._normalize_cwe(["CWE-119"]), ["CWE-119"])

    def test_parses_stringified_list(self):
        self.assertEqual(
            DataFetcher._normalize_cwe("['CWE-119', 'CWE-125']"), ["CWE-119", "CWE-125"]
        )

    def test_empty_and_nan_become_empty_list(self):
        self.assertEqual(DataFetcher._normalize_cwe(None), [])
        self.assertEqual(DataFetcher._normalize_cwe(""), [])
        self.assertEqual(DataFetcher._normalize_cwe("nan"), [])

    def test_plain_string_is_wrapped(self):
        self.assertEqual(DataFetcher._normalize_cwe("CWE-20"), ["CWE-20"])


if __name__ == "__main__":
    unittest.main()
