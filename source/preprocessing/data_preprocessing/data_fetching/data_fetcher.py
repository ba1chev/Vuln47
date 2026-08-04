import os
import ast
import json
import urllib.request
import pyarrow.parquet as pq

from source.preprocessing.fetcher import Fetcher
from source.preprocessing.data_preprocessing.data_fetching.data_fetcher_config import (
    RAW_DATA_DIR, SPLIT_FILENAMES, SPLIT_URLS
)


class DataFetcher(Fetcher[str, list[str]]):
    def fetch(self, input: str = RAW_DATA_DIR) -> list[str]:
        os.makedirs(input, exist_ok=True)
        paths = []
        for split, filename in SPLIT_FILENAMES.items():
            dest = os.path.join(input, filename)
            if not os.path.exists(dest):
                self._download(split, dest)
            paths.append(dest)
        return paths

    def _download(self, split: str, dest: str) -> None:
        url = SPLIT_URLS.get(split)
        if not url:
            raise ValueError(
                f"No download URL configured for split '{split}' in data_fetcher_config.SPLIT_URLS. "
                f"Set it before fetching, or place {dest} manually."
            )
        parquet_tmp = dest + ".parquet.part"
        request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request) as response, open(parquet_tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)

        jsonl_tmp = dest + ".part"
        self._parquet_to_jsonl(parquet_tmp, jsonl_tmp)
        os.remove(parquet_tmp)
        os.replace(jsonl_tmp, dest)

    def _parquet_to_jsonl(self, parquet_path: str, jsonl_path: str) -> None:
        table = pq.read_table(parquet_path)
        with open(jsonl_path, "w") as f:
            for batch in table.to_batches(max_chunksize=1000):
                for record in batch.to_pylist():
                    record["cwe"] = self._normalize_cwe(record.get("cwe"))
                    record["target"] = int(record.get("target", 0) or 0)
                    f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _normalize_cwe(value) -> list[str]:
        if isinstance(value, list):
            return value
        if not value or not isinstance(value, str) or value.strip() in ("", "nan"):
            return []
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (ValueError, SyntaxError):
            return [value]
