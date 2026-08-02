import json
from typing import Iterator

from source.preprocessing.loader import Loader


class DataLoader(Loader[str, Iterator[dict]]):
    def load(self, input: str) -> Iterator[dict]:
        with open(input) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
