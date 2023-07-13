import xarray as xr
import numpy as np
import DA_core as DA
from ML_core import Unet,Unet_2L
import torch

DA_kwargs={}

# Setting up DA experiment what was used to train Unet
DA_training_paras={'nens':320,
                   'DA_method':'EnKF',
                   'Nx_DA':64,
                   'Nx_truth':128,
                   'obs_freq':10,
                   'obs_err':[1,-5,5,-7],
                   'DA_freq':10,
                   'save_B':False,
                   'nobs':[50,50],
                   'R_W':100,
                   'inflate':[1,0.45]}
DA_training=DA.DA_exp(**DA_training_paras)

# Read trained Unet and normalization factors (standard deviations)
in_ch=[0,1]
out_ch=[0,1,2]
epoch=30
R_training=12
R_DA=12
features=32
Ulevels=2
if Ulevels==3:
    model=Unet(in_ch=len(in_ch),out_ch=len(out_ch),features=features).double()
elif Ulevels==2:
    model=Unet_2L(in_ch=len(in_ch),out_ch=len(out_ch),features=features).double()
model_file='./ML/{}/{}L_{}f/unet_epoch{}_in{}_out{}_B{}_{}.pt'.format(
    DA_training.file_name(),Ulevels,features,epoch,''.join(map(str,in_ch)),''.join(map(str,out_ch)),
    R_training*2,DA_training.file_name())
print(model_file)
model.load_state_dict(torch.load(model_file))
model.eval()
ml_std_ds=xr.open_dataset('./ML/{0}/std_{0}.nc'.format(DA_training.file_name()))
DA_kwargs['ml_model']=model
DA_kwargs['ml_std_ds']=ml_std_ds


for R_W in [100]:
    for nens,relax in zip([1],[0.0]):
        DA_paras={'nens':nens,
                'DA_method':'UnetKF',
                'Nx_DA':64,
                'Nx_truth':128,
                'obs_freq':10,
                'obs_err':[1,-5,5,-7],
                'nobs':[50,50],
                'R_W':R_W,
                'DA_freq':10,
                'save_B':False,
                'inflate':[1,relax],
                'B_alpha':0.0,
                'R_training':R_training,
                'R_DA':R_DA,
                'training_exp':DA_training}
        DA_exp=DA.DA_exp(**DA_paras)
        DA_kwargs['output_str']='UnetKF_Nx{}_128_ens{}_{}L{}f'.format(
            DA_exp.Nx_DA,DA_training.nens,Ulevels,features)
        if DA_exp.Nx_DA!=DA_training.Nx_DA:
            DA_kwargs['output_str']=DA_kwargs['output_str']+'_interp'
            
        DA_exp.run_exp(DA_days=7300,DA_start=0,**DA_kwargs)