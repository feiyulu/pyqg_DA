# pyqg_DA
Data assimilation system for the PyQG model

This repository provides data assimilation (DA) capability for the PyQG model, 
as documented in the [Lorenz-96 notebooks](https://m2lines.github.io/L96_demo/notebooks/DA_demo_L96.html).

## File Structure

### DA related
* `DA_core.py`: data assimilation related functions
* `DA_QG2L.ipynb`: notebook for data assimilation examples in 2-layer QG model
* `DA_analysis.ipynb`: notebook to analyze DA results
* `ENKF.py`, `ENKF_training.py`: Python script to run EnKF DA experiments in bulk

### ML related
* `ML_core.py`: machine learning related functions
* `B_UNet.ipynb`: notebook to train U-Net to predict ensemble covariances
* `UNet_analysis`.ipynb: notebook to analyze trained U-Nets
* `training.py`: Python script to train U-Net
* `UNetKF.py`: Python script to run UNetKF DA experiments

## Current Issues

### `ML_core.py`

### `B_UNet.ipynb`
