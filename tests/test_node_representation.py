import unittest

from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node import CodeNode
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_representator import CodeNodeRepresentator
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_config import (
    token_bucket, NUM_TOKEN_BUCKETS, NUM_TOKEN_FEATURES, DANGEROUS_FUNCS,
)


class TokenBucketTests(unittest.TestCase):
    def test_empty_token_reserved_bucket_zero(self):
        self.assertEqual(token_bucket(""), 0)

    def test_non_empty_token_never_uses_reserved_bucket(self):
        for tok in ("memcpy", "x", "some_identifier", "42"):
            self.assertGreaterEqual(token_bucket(tok), 1)
            self.assertLess(token_bucket(tok), NUM_TOKEN_BUCKETS)

    def test_deterministic_across_calls(self):
        self.assertEqual(token_bucket("memcpy"), token_bucket("memcpy"))


class TokenFeatureTests(unittest.TestCase):
    def setUp(self):
        vocab = CodeNodeRepresentator.load_or_build_vocab()
        self.repr = CodeNodeRepresentator(vocab)

    def test_feature_vector_length(self):
        self.assertEqual(len(self.repr._token_features("x")), NUM_TOKEN_FEATURES)

    def test_empty_token_is_all_zeros(self):
        self.assertEqual(self.repr._token_features(""), [0.0] * NUM_TOKEN_FEATURES)

    def test_dangerous_function_flag(self):
        self.assertEqual(self.repr._token_features("memcpy")[1], 1.0)
        self.assertEqual(self.repr._token_features("harmless_name")[1], 0.0)
        for fn in DANGEROUS_FUNCS:
            self.assertEqual(self.repr._token_features(fn)[1], 1.0, fn)

    def test_numeric_literal_flag(self):
        self.assertEqual(self.repr._token_features("42")[3], 1.0)
        self.assertEqual(self.repr._token_features("x1")[3], 0.0)

    def test_size_like_name_flag(self):
        for name in ("len", "buf_size", "idx", "n_count", "buffer"):
            self.assertEqual(self.repr._token_features(name)[4], 1.0, name)
        self.assertEqual(self.repr._token_features("value")[4], 0.0)

    def test_normalized_length_is_capped_at_one(self):
        self.assertAlmostEqual(self.repr._token_features("a" * 10)[2], 0.5)
        self.assertEqual(self.repr._token_features("a" * 40)[2], 1.0)


class NodeRepresentTests(unittest.TestCase):
    def setUp(self):
        self.vocab = CodeNodeRepresentator.load_or_build_vocab()
        self.repr = CodeNodeRepresentator(self.vocab)

    def test_output_shape_is_type_bucket_plus_features(self):
        vec = self.repr.represent(CodeNode("identifier", "memcpy"))
        self.assertEqual(len(vec), 2 + NUM_TOKEN_FEATURES)

    def test_known_type_maps_to_its_vocab_id(self):
        vec = self.repr.represent(CodeNode("identifier", "x"))
        self.assertEqual(vec[0], float(self.vocab["identifier"]))

    def test_unknown_type_maps_to_unk(self):
        vec = self.repr.represent(CodeNode("NOT_A_REAL_AST_TYPE", "x"))
        self.assertEqual(vec[0], float(self.repr.unk))

    def test_bucket_column_matches_token_bucket(self):
        vec = self.repr.represent(CodeNode("identifier", "memcpy"))
        self.assertEqual(vec[1], float(token_bucket("memcpy")))


if __name__ == "__main__":
    unittest.main()
