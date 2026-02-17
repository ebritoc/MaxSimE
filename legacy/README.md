# Legacy Code

This directory contains the original MaxSimE implementation from the SIGIR 2023 paper.

## Contents

- `maxsime.py` - Original monolithic implementation with ColBERT integration
- `colbert/` - Vendored ColBERT fork for dense retrieval
- `baleen/` - Multi-hop retrieval module
- `utility/` - ColBERT utilities
- `setup.py` - Original package configuration (for ColBERT, not maxsime)
- `conda_env.yml` - Original conda environment (GPU)
- `conda_env_cpu.yml` - Original conda environment (CPU)

## Purpose

This code is preserved for:
1. Reproducing the exact results from the SIGIR 2023 paper
2. Using ColBERT-specific features not available in the new package
3. Reference implementation for the MaxSimE algorithm

## Usage

For reproduction, follow the original README instructions with:

```bash
conda env create -f legacy/conda_env.yml
conda activate maxsime
```

Then use the Jupyter notebook at `notebooks/MaxSimE.ipynb`.

## Note

For new projects, use the refactored package instead:

```python
pip install git+https://github.com/ebritoc/MaxSimE.git
from maxsime import MaxSimExplainer
```
