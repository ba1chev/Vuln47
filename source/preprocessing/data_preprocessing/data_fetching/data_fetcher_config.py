import os

RAW_DATA_DIR = os.path.join("data")

SPLIT_FILENAMES = {
    "train": "primevul_train.jsonl",
    "valid": "primevul_valid.jsonl",
    "test": "primevul_test.jsonl"
}

HF_DATASET = "nimaster/primevul_dataset"

SPLIT_URLS = {
    "train": f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/data/train-00000-of-00001.parquet",
    "valid": f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/data/valid-00000-of-00001.parquet",
    "test": f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/data/test-00000-of-00001.parquet"
}
