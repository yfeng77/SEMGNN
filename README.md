# SEMGNN: Self-Explainable Multi-Label Graph Neural Network

Dataset and code of Self-Explainable Multi-Label Graph Neural Network (SEMGNN).

## Environment

- Python 3.10+
- See `requirements.txt` for Python dependencies.
- Please install a `torch` / `torch-geometric` version compatible with your local CUDA or CPU environment.

## Repository files
```text
├── train_SEMGNN.py
├── SEMGNN.py
├── utils/
│   ├── utils.py
│   ├── gt_registry.py
│   ├── optimal_threshold.py
│   └── auto_tune.py
├── eval/
│   └── evaluate.py
├── data_process/
│   └── data_process.py
├── data_gen/
│   ├── gen_synthetic_anchor.py
│   └── gen_synthetic_motif.py
└── dataset/
    ├── blogcatalog/
    ├── synthetic_anchor/
    ├── synthetic_motif/
    └── youtube/
```

- `train_SEMGNN.py`: main training and evaluation script
- `SEMGNN.py`: model and loss definitions
- `data_process.py`: dataset loading utilities
- `evaluate.py`: prediction and explanation evaluation utilities
- `utils.py`, `gt_registry.py`, `optimal_threshold.py`: supporting utilities
- `gen_synthetic_anchor.py`, `gen_synthetic_motif.py`: synthetic data generation scripts
- `auto_tune.py`: Optuna-based hyperparameter tuning

## Data

The training script expects datasets under:

```text
./dataset/<dataset_name>/
```

Supported datasets in the current code include:

- `blogcatalog`
- `youtube`
- `synthetic_motif`
- `synthetic_anchor`

Example files for a dataset directory:

```text
A_coo.npz / edges.csv / edges.txt
X.npy or features.npy
Y.npy
id_mapping.csv   (optional for some datasets)
```

## Training

Example:

```bash
python train_SEMGNN.py --dataset synthetic_anchor --device cuda
```

Useful optional flags:

```bash
--data_dir PATH
--split_path PATH
--eval_nodes_path PATH
--no-use_label_aware
--no-label_use_correlation
--no-use_k_label_sampling
--no-use_label_gcn_pred
```

## Hyperparameter tuning

Example:

```bash
python auto_tune.py --datasets synthetic_anchor synthetic_motif --n_trials 40 --device cuda
```

## Notes

- This is an anonymous repository prepared for double-blind review.
- Checkpoints, logs, and generated results are also excluded.
