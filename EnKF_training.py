import xarray as xr
import numpy as np
import DA_core as DA
from os.path import exists

DA.read_data_dir='/scratch/cimes/feiyul/PyQG/data/training'
DA.save_data_dir='/scratch/cimes/feiyul/PyQG/data/training'

DA_setup_paras={'nens':1,
                'Nx_DA':64,          
                'Nx_truth':128,          
                'obs_freq':10,          
                'obs_err':[1,-5,5,-7],          
                'nobs':[50,50]}
DA_setup=DA.DA_exp(**DA_setup_paras)

try:
    ds_truth=DA_setup.read_truth(years=10)
except:
    ds_truth=DA_setup.generate_truth(years=10)
print(ds_truth)

## Generate or read observations
if exists(DA_setup.obs_name()):
    obs_ds=DA_setup.read_obs()
else:
    obs_ds=DA_setup.generate_obs(years=10)
print(obs_ds)

DA_kwargs={}
for R_W in [100]:
    for nens,relax in zip([1280],[0.45]):
        DA_paras={'nens':nens,
                'DA_method':'EnKF',
                'Nx_DA':64,
                'Nx_truth':128,
                'obs_freq':10,
                'obs_err':[1,-5,5,-7],
                'nobs':[50,50],
                'R_W':R_W,
                'DA_freq':10,
                'save_B':True,
                'inflate':[1,relax]}
        DA_exp=DA.DA_exp(**DA_paras)
        DA_exp.run_exp(DA_days=3650,DA_start=0,**DA_kwargs)