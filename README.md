# Vuln47

Detecting vulnerabilities in C/C++ functions with a Graph Neural Network.
Each function is parsed into an AST graph and classified as vulnerable (`1`) or safe (`0`) by an edge-aware GIN (GINE) model.

## Overview

Vuln47 treats vulnerability detection as **graph classification**. A raw C/C++ function is parsed
into an Abstract Syntax Tree, the tree is turned into a typed feature graph, and an edge-aware Graph
Isomorphism Network (GINE) predicts whether the function is vulnerable. The whole path — from
downloading the dataset to a trained checkpoint and its evaluation — is split into small, reusable
components under `source/`, and walked through end to end in the notebooks under `notebooks/`.

## Data

The project uses the **PrimeVul** dataset (Ding et al., 2024): ~236k real C/C++ functions
labelled from CVEs, split into `train` / `valid` / `test` JSONL files. The data lives under
`data/` and is **not** tracked by git (see `.gitignore`).

The dataset is severely **imbalanced** — only ~2.7–3.0% of functions are vulnerable (a ~1:32
safe-to-vulnerable ratio on the training split) — which drives
every downstream choice: a class-weighted loss during training and F1 / precision / recall / PR-AUC
(never raw accuracy) during evaluation.

The original PrimeVul is distributed only via Google Drive (no stable direct-download links):
<https://github.com/DLVulDet/PrimeVul>. This project instead pulls a stable HuggingFace mirror
([`nimaster/primevul_dataset`](https://huggingface.co/datasets/nimaster/primevul_dataset)),
which serves the same records as Parquet over plain HTTP.

### Fetching

The three splits are obtained by the project's `DataFetcher`
(`source/preprocessing/data_preprocessing/data_fetching/`), which downloads any missing split
into `data/` and skips ones already present. For each split it fetches the HuggingFace Parquet
(URLs in `data_fetcher_config.py`, `SPLIT_URLS`) and converts it on the fly to the
`primevul_*.jsonl` the rest of the pipeline reads (`cwe` normalized to a list, `target` to int).
Point `SPLIT_URLS` at a different mirror if you prefer; if the files are already in `data/`,
fetching is a no-op.

Fetching happens automatically as the first step of `DataPreprocessingPipeline.run()`, and is also
demonstrated explicitly in Notebook 01.

## Preprocessing pipeline

`DataPreprocessingPipeline` chains four reusable stages, each a component in the `source/` package:

```
C code (str)
  → CodeGraphRepresentator   (tree-sitter parse → CodeGraph / AST with typed edges)
  → CodeNodeRepresentator    (node type + token → 7-D feature vector)
  → DataPreprocessingPipeline (assemble → PyG Data with typed bidirectional edges)
  → list[Data]               (saved to data/processed/*.pt)
```

Each node becomes `[type_id] ‖ [token_bucket] ‖ [5 token features]` — the `type_id` indexes a learned
embedding (vocabulary in `node_type_vocab.json`), the `token_bucket` is a crc32-hashed index into a
learned token embedding (so the model can learn token semantics without an external tokenizer,
`NUM_TOKEN_BUCKETS = 4096`), and the five hand-crafted features encode C-vulnerability intuition
(leaf flag, dangerous-call flag, token length, numeric literal, size/length/index-like name). The AST
is enriched with three edge types (`AST_CHILD`, `NEXT_SIBLING`, `USE_DEF`), each made bidirectional
for a total of `NUM_EDGE_TYPES = 6`. Very large functions are capped (`MAX_CODE_BYTES = 30_000`,
`MAX_NODES = 2_000` in `code_graph_config.py`), which discards <1% of the data.

## Model

`Vuln47GNN` (`source/vuln47_gnn_model.py`) is an edge-aware **GIN (GINE)** classifier:

- a learned embedding for the AST node type, concatenated with a learned token-bucket embedding and
  the token features, projected to a hidden dimension;
- `num_layers` (default 4) `GINEConv` layers — each consuming a learned **edge-type embedding** — with
  batch-norm, ReLU, dropout, and residual connections;
- a **JumpingKnowledge** (concat) readout over all layers, then a graph readout that concatenates
  **mean** and **max** pooling, followed by a 2-class head.

Training defaults (`training_config.py`): Adam (`lr=1e-3`, `weight_decay=1e-5`), `batch_size=64`,
`epochs=30`, **focal loss** (`gamma=2.0`) on top of a ~32× class weight for the rare vulnerable
class, `ReduceLROnPlateau` scheduler and early stopping (patience 5), best checkpoint selected on
**PR-AUC**, and a best-F1 **decision threshold** calibrated on the valid split and stored in the
checkpoint. Training is resumable (a rolling `last.pt` is written each epoch). Device auto-selects
CUDA, then Apple **MPS**, then CPU.

## Layout

- `source/preprocessing/` — fetch → load → AST graph → node features → PyG `Data`
  - abstract bases: `Fetcher`, `Loader`, `DomainRepresentator`, `Explorer`, `PreprocessingPipeline`
  - concrete PrimeVul/graph implementations under `data_preprocessing/`
- `source/training/` — batch → train GINE → evaluate (F1 / precision / recall / PR-AUC)
  - abstract bases: `Trainer`, `TrainingPipeline`; concrete under `model_training/`
- `source/vuln47_gnn_model.py` — the `Vuln47GNN` model and `build_model` factory
- `notebooks/` — `00` theory · `01` EDA · `02` preprocessing · `03` training · `04` evaluation · `05` conclusion

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All code uses full `source.`-prefixed imports and must be run from the **project root**.

## Usage

Work through the notebooks in order (`00` → `05`), or drive the pipeline directly from Python:

```python
from source.preprocessing.data_preprocessing.data_representation.data_node_representation.code_node_representator import CodeNodeRepresentator
from source.preprocessing.data_preprocessing.data_preprocessing_pipeline import DataPreprocessingPipeline

vocab = CodeNodeRepresentator.load_or_build_vocab()
graphs = DataPreprocessingPipeline(vocab).run("data/primevul_valid.jsonl")  # fetches if missing
```

Training then consumes the saved `data/processed/{train,valid,test}.pt` tensors and writes the best
checkpoint to `data/model/vuln_gnn.pt`.
