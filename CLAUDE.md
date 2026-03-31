# CLAUDE.md

## Project Overview

DBMS Transfer Learning is a research framework for transferring database performance models across **contexts**, where a context is a combination of hardware configuration and workload (e.g., 8c12g + TPCC). The goal is to predict DBMS throughput (TPS) on a new target context using data/models from source contexts, with minimal sampling on the target. The project supports MySQL and LineairDB.

## Tech Stack

- **Python 3.10+**
- **PyTorch** — neural network-based metric learning for context similarity
- **scikit-learn** — regression models (Gaussian Process, Random Forest, etc.)
- **Hydra** — YAML-based experiment configuration
- **pandas/numpy/scipy** — data processing
- **matplotlib** — visualization

## Project Structure

- `regression/` — core ML modules
  - `pipeline.py` — main pipeline: orchestrates context retrieval + data transfer
  - `context_retrieval/` — finds source contexts most similar to the target
    - `static/impl/` — pre-computed methods (metric similarity, parameter importance, metric learning)
    - `dynamic/impl/` — on-demand methods (concordant ranking pairs)
  - `data_transfer/impl/` — transfer learning algorithms (ChimeraTech/Libra, DataReuse, L2S, ModelShift, ModelEnsemble)
  - `system_configuration.py` — DBMS definitions (MySQLConfiguration, LineairDBConfiguration)
  - `jamshidi.py` — feature selectors (LassoCV, ElasticNet)
  - `model.py` — model wrapper
  - `utils.py` — data loading utilities
- `scripts/` — runnable experiments
  - `batch.py` — batch experiment runner
  - `transfer_learning.py` — main transfer learning script
  - `cross_validation.py`, `correlation.py`, `parameter_space.py` — other experiments
  - `conf/` — Hydra YAML configs
- `neural_network/` — neural net utilities, preprocessing, trained weights
- `dataset/` — MySQL performance CSVs (named by hardware spec, e.g., `8c12g-result.csv`)

## Running Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Run an experiment (context_retrieval | full_transfer | data_transfer)
PYTHONPATH=. python scripts/batch.py context_retrieval

# Run a single config
PYTHONPATH=. python scripts/transfer_learning.py --config-name mysql.yaml

# Run via makefile (regenerate first if configs changed)
python create_makefile.py
make all_transfer_learning
```

Outputs go to `outputs/` (CSVs + PDF plots).

## Architecture

The transfer learning pipeline has two stages:

1. **Context retrieval** — given a target context (hardware + workload), find the most similar source contexts from the dataset
2. **Data transfer** — use source context data/models to build a performance model for the target, augmented with a small number of target samples

`Pipeline` loads source contexts at each sampling iteration; `FastPipeline` loads them once at the end.

## Conventions

- snake_case for functions/variables, CamelCase for classes
- Abstract base classes for extensibility (`DataTransfer`, `SystemConfiguration`, `FeatureSelector`)
- Hydra YAML configs with `_target_` for class instantiation
- MAPE (Mean Absolute Percentage Error) as the primary evaluation metric
- Dataset filenames encode hardware specs: `{cores}c{memory}g-result.csv`
