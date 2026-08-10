import json
from typing import Iterator

from source.preprocessing.loader import Loader


class DataLoader(Loader[str, Iterator[dict]]):
    """Streams a PrimeVul JSONL split one record at a time.

    Yields lazily rather than reading the whole file into memory — the train
    split is ~350 MB — so callers can process ~236k records without materializing
    them all at once.
    """

    def load(self, input: str) -> Iterator[dict]:
        with open(input) as f:
            for line in f:
                line = line.strip()
                if line:  # tolerate blank lines / trailing newline
                    yield json.loads(line)
