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
    """Downloads the PrimeVul splits into 'data' if they are missing.

    The upstream dataset ships as Parquet on a HuggingFace mirror; each split is
    downloaded and converted on the fly to the 'primevul_*.jsonl' the rest of
    the pipeline reads. Idempotent: splits already present on disk are skipped.
    """

    def fetch(self, input: str = RAW_DATA_DIR) -> list[str]:
        os.makedirs(input, exist_ok=True)
        paths = []
        for split, filename in SPLIT_FILENAMES.items():
            dest = os.path.join(input, filename)
            if not os.path.exists(dest):  # skip splits already fetched
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
        # stream the parquet to a temp file, then convert; only replace the final
        # destination once conversion succeeds so a crash never leaves a partial split
        parquet_tmp = dest + ".parquet.part"
        request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request) as response, open(parquet_tmp, "wb") as out:
            while chunk := response.read(1 << 20):  # read 1 MiB at a time
                out.write(chunk)

        jsonl_tmp = dest + ".part"
        self._parquet_to_jsonl(parquet_tmp, jsonl_tmp)
        os.remove(parquet_tmp)
        os.replace(jsonl_tmp, dest)  # atomic rename into place

    def _parquet_to_jsonl(self, parquet_path: str, jsonl_path: str) -> None:
        # convert in batches so a large split never has to fit fully in memory
        table = pq.read_table(parquet_path)
        with open(jsonl_path, "w") as f:
            for batch in table.to_batches(max_chunksize=1000):
                for record in batch.to_pylist():
                    # normalize the two fields the pipeline relies on
                    record["cwe"] = self._normalize_cwe(record.get("cwe"))
                    record["target"] = int(record.get("target", 0) or 0)
                    f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _normalize_cwe(value) -> list[str]:
        """Coerce the 'cwe' field into a clean 'list[str]'.

        The mirror stores CWEs inconsistently (a real list, a stringified list,
        a bare string, or an empty/'nan' placeholder); this collapses all of
        those into a plain list so downstream EDA never has to special-case them.
        """
        if isinstance(value, list):
            return value
        if not value or not isinstance(value, str) or value.strip() in ("", "nan"):
            return []
        try:
            # e.g. "['CWE-119', 'CWE-125']" stored as a string
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (ValueError, SyntaxError):
            return [value]  # a plain "CWE-20" string
