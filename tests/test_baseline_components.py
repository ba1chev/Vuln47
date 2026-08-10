import os
import json
import tempfile
import unittest
import numpy as np
from scipy.sparse import issparse

from source.training.baseline_training.vuln47_lr import Vuln47LR
from source.training.baseline_training.token_featurizer import TokenFeaturizer
from source.training.baseline_training.baseline_dataset import BaselineDataset
from source.training.baseline_training.baseline_evaluation.baseline_metrics_evaluator import BaselineMetricsEvaluator


class TokenFeaturizerTests(unittest.TestCase):
    def setUp(self):
        self.feat = TokenFeaturizer(n_features=1024, use_tfidf=False)

    def test_transform_is_sparse_with_fixed_width(self):
        X = self.feat.fit_transform(["int a = b + c;", "char *p = malloc(n);"])
        self.assertTrue(issparse(X))
        self.assertEqual(X.shape, (2, 1024))

    def test_different_code_maps_to_different_rows(self):
        X = self.feat.fit_transform(["int x = 0;", "free(ptr);"])
        self.assertFalse((X[0].toarray() == X[1].toarray()).all())

    def test_identical_code_maps_to_identical_rows(self):
        src = "memcpy(dst, src, n);"
        X = self.feat.fit_transform([src, src])
        self.assertTrue((X[0].toarray() == X[1].toarray()).all())

    def test_empty_string_is_handled(self):
        X = self.feat.fit_transform([""])
        self.assertEqual(X.shape, (1, 1024))
        self.assertEqual(X.nnz, 0)


class BaselineDatasetTests(unittest.TestCase):
    def test_load_returns_texts_and_labels(self):
        records = [
            {"func": "int safe() { return 0; }", "target": 0},
            {"func": "void bad() { gets(buf); }", "target": 1},
            {"func": "int mid() { return 1; }", "target": 0}
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name
        try:
            texts, y = BaselineDataset().load(path)
            self.assertEqual(len(texts), 3)
            self.assertEqual(len(y), 3)
            self.assertTrue(np.array_equal(y, np.array([0, 1, 0])))
            self.assertEqual(texts[1], "void bad() { gets(buf); }")
        finally:
            os.remove(path)


class Vuln47LRTests(unittest.TestCase):
    def setUp(self):
        feat = TokenFeaturizer(n_features=256, use_tfidf=False)
        texts = ["gets(buf); strcpy(d, s);"] * 6 + ["int x = 0; return x;"] * 6
        self.X = feat.fit_transform(texts)
        self.y = np.array([1] * 6 + [0] * 6)

    def test_train_returns_fitted_model(self):
        lr = Vuln47LR(max_iter=200)
        model = lr.train((self.X, self.y))
        self.assertTrue(hasattr(model, "coef_"))

    def test_predict_scores_shape_and_range(self):
        lr = Vuln47LR(max_iter=200)
        lr.train((self.X, self.y))
        scores = lr.predict_scores(self.X)
        self.assertEqual(scores.shape, (12,))
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())


class BaselineMetricsEvaluatorTests(unittest.TestCase):
    def setUp(self):
        feat = TokenFeaturizer(n_features=256, use_tfidf=False)
        texts = ["gets(buf); strcpy(d, s);"] * 8 + ["int x = 0; return x;"] * 8
        self.X = feat.fit_transform(texts)
        self.y = np.array([1] * 8 + [0] * 8)
        self.lr = Vuln47LR(max_iter=300)
        self.lr.train((self.X, self.y))
        self.evaluator = BaselineMetricsEvaluator(self.lr)

    def test_evaluate_returns_expected_keys(self):
        metrics = self.evaluator.evaluate((self.X, self.y))
        self.assertEqual(set(metrics), {"f1", "precision", "recall", "pr_auc"})

    def test_separable_data_scores_high_f1(self):
        metrics = self.evaluator.evaluate((self.X, self.y))
        self.assertGreater(metrics["f1"], 0.9)

    def test_best_threshold_returns_float(self):
        t = self.evaluator.best_threshold((self.X, self.y))
        self.assertIsInstance(t, float)


if __name__ == "__main__":
    unittest.main()
