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

## Instructions

For the code clinic, I would like to optimize the code related to the training and inference, especially in the case of using GPU.

The scheme of the data pipline is as follows:

At each time step, the full `q` datasets have the size of `(level,Ny,Nx)`, so the full covariance matrix of `q` would have size of `(level,Ny,Nx,level,Ny,Nx)`. 
In this current `PyQG` implementation of EnKF, we use the full covariance matrices during the data assimilation step. However, the full covariance matrices at all time steps are prohibitively big to save for training U-Nets.
Since we normally use covariance localization in EnKF applications, only part of the full covariance matrix is used (usually based on physical distance), we can save only localized matrices. 
As a result, the saved `q` datasets have the size of `(time,level,Ny,Nx)`, while the saved covariance matrices `B` have the size of `(time,level,Nx,Nx,level,Ny_local,Nx_local)`, where `Ny_local` and `Nx_local` are significantly smaller than `Ny` and `Nx`.

During training, each data sample consists of a localized `q` matrix and a localized `B` matrix. The localized `B` with size of `(level,Ny_local,Nx_local)` would simply be a subset of the full dataset, while the localized `q` is taken as subset of the full matrix at runtime.

The same process happens during inference. When the U-Net is applied in the DA processs, a localized `q` matrix is constructed around each model gridpoint.

### `ML_core.py`

### `B_UNet.ipynb`

### `DA_core.py`