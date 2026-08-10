from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from source.training.baseline_training.baseline_config import N_FEATURES, USE_TFIDF

# a crude C lexer: identifiers, numeric literals, and single punctuation tokens.
# deliberately structure-blind — it sees the bag of tokens the GNN's AST throws away.
_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\sA-Za-z0-9_]"


class TokenFeaturizer:
    """Turns raw C source into a hashed bag-of-tokens sparse matrix.

    Hashing (not CountVectorizer) means there is no fitted vocabulary to persist and
    train/valid/test always share the same fixed dimensionality.
    """

    def __init__(self, n_features: int = N_FEATURES, use_tfidf: bool = USE_TFIDF):
        self.use_tfidf = use_tfidf
        self._hasher = HashingVectorizer(
            n_features=n_features,
            token_pattern=_TOKEN_PATTERN,
            alternate_sign=False,
            norm=None
        )
        self._tfidf = TfidfTransformer() if use_tfidf else None

    def fit(self, texts: list[str]) -> "TokenFeaturizer":
        if self.use_tfidf:
            self._tfidf.fit(self._hasher.transform(texts))
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        counts = self._hasher.transform(texts)
        return self._tfidf.transform(counts) if self.use_tfidf else counts

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        return self.fit(texts).transform(texts)
