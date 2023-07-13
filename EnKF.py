import xarray as xr
import numpy as np
import DA_core as DA

DA_kwargs={}
for R_W in [75]:
    for nens,relax in zip([5,5,5],[0.7,0.75,0.8]):
        DA_paras={'nens':nens,
                'DA_method':'EnKF',
                'Nx_DA':32,
                'Nx_truth':128,
                'obs_freq':10,
                'obs_err':[1,-5,5,-7],
                'nobs':[50,50],
                'R_W':R_W,
                'DA_freq':10,
                'save_B':False,
                'inflate':[1,relax]}
        DA_exp=DA.DA_exp(**DA_paras)
        DA_exp.run_exp(DA_days=7300,DA_start=0,**DA_kwargs)