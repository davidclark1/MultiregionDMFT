# Structure of Activity in Multiregion Recurrent Neural Networks

## Authors
- David G. Clark
- Manuel Beiran

This repository contains code for simulations and figures from our paper "Structure of activity in multiregion recurrent neural networks."

## Repository Structure

### Notebooks for Figure Generation
- **Figs. 3, 4**: `paper-notebooks/fixed-points.ipynb`
- **Fig. 5**: `paper-notebooks/heatmaps.ipynb`
- **Fig. 6** (and appendix figure "Relationship between disorder and current-variable dynamic complexity..."): `paper-notebooks/shaping-dynamics.ipynb`
- **Appendix Fig on input flips**: `input-flip.ipynb`
- **Appendix Fig on convex structure of fixed point manifold**: `convex-figs.ipynb`

### Core Scripts
- **`numerics.py`**: Main implementation of DMFT for multi-region networks and associated utilities
- **`convex.py`**: Analysis of fixed-point manifold geometry using the cdd library for the Double Description Method
- **`run_limit_cycle_dmft.py`**: Generates DMFT results for Fig. 5 (called by `limit_cycle_dmft.slurm` on cluster)
- **`run_fp_sims.py`**: Runs high-dimensional network simulations for finding fixed points in Figs. 3 and 4 (called by `fp_sims.slurm` on cluster)

### Utilities
- **`paper-notebooks/style.py`**: Plotting utilities and style definitions

## Technical Details

The codebase is implemented in Python, using:
- Python 3.8.13
- NumPy for numerical computations
- PyTorch for large-scale simulations on GPU
- Matplotlib for plotting

## Data Generation

The figure generation notebooks load pre-computed `.npz` files. These data files are not included in the repository due to size constraints but can be regenerated using the provided code. The analysis routines that generate these outputs are either:
1. Located in the plotting notebooks themselves
2. Found in separate scripts that can be run locally or on a cluster
