import xarray as xr
import numpy as np
import DA_core as DA

DA_kwargs={}
for seed in range(25,41):
        for delta_days in [60]:
                for R_W in [100]:
                        DA_paras={'nens':1,
                                'DA_method':'3DVar',
                                'Nx_DA':64,
                                'Nx_truth':128,
                                'obs_freq':10,
                                'obs_err':[1,-5,5,-7],
                                'nobs':[50,50],
                                'R_W':R_W,
                                'DA_freq':10,
                                'delta_days':delta_days,
                                'save_B':False,
                                'inflate':[1,0.0],
                                'output_str':''}
                        DA_exp=DA.DA_exp(**DA_paras)
                        DA_exp.run_exp(DA_days=365*50,DA_start=0,ic_seed=seed,**DA_kwargs)
        
