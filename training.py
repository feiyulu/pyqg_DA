import xarray as xr
import numpy as np
import importlib
from matplotlib import pyplot as plt
import DA_core as DA
from glob import glob
import torch.utils.data as Data
from torch import optim
import torch
from torchsummary import summary
import ML_core as ML
from numpy.random import default_rng
import os

rng = default_rng()
DA.read_data_dir='/scratch/cimes/feiyul/PyQG/data/training'
DA.save_data_dir='/scratch/cimes/feiyul/PyQG/data/training'

data_dir='/scratch/cimes/feiyul/PyQG/data'

DA_paras={'nens':1280,
          'DA_method':'EnKF',
          'Nx_DA':32,
          'Nx_truth':128,
          'obs_freq':10,
          'obs_err':[1,-5,5,-7],
          'DA_freq':10,
          'save_B':False,
          'nobs':[50,50],
          'R_W':100,
          'inflate':[1,0.45]}
DA_exp=DA.DA_exp(**DA_paras)
print(DA_exp.file_name())
# obs_ds=DA_exp.read_obs()
in_ch=[0,1]
out_ch=[0,1,2]
print(in_ch,out_ch)

mean_ds=DA_exp.read_mean().load()
print(mean_ds.q.shape)
# if DA_exp.nens>1:  
#     std_ds=DA_exp.read_std()

# B_ens_ds=xr.open_mfdataset(['{}/{}/B_ens_day{:04d}.nc'.format(data_dir,DA_exp.file_name(),day) for day in np.arange(9,3650,10)])
B_ens_ds=xr.open_dataset('{}/training/{}/B_ens.nc'.format(data_dir,DA_exp.file_name()))
print(B_ens_ds.B_ens.shape)

ml_std_ds=xr.open_dataset('./ML/{0}/std_{0}.nc'.format(DA_exp.file_name()))
print(ml_std_ds)

B_R=int((len(B_ens_ds.x_d)-1)/2)
B_size=16
B_start=0

DA_days=slice(369,3650,DA_exp.DA_freq)
DA_it=slice(int((DA_days.start-DA_exp.DA_freq+1)/DA_exp.DA_freq),
            int((DA_days.stop-DA_exp.DA_freq+1)/DA_exp.DA_freq)+1)
i_x=np.arange(0,DA_exp.Nx_DA)
i_y=np.arange(0,DA_exp.Nx_DA)

B_shape=B_ens_ds.B_ens.isel(time=DA_it,y=i_y,x=i_x).shape
print(B_shape)
print(mean_ds.q.isel(time=DA_days).shape)
B_nt=B_shape[0]
B_ny=B_shape[2]
B_nx=B_shape[3]
B_total=B_nt*B_ny*B_nx
print(B_total)
n_train=int(B_total*0.8)
# rngs=rng.permutation(B_total)
rngs=np.arange(B_total)
partition={'train':rngs[:n_train],'valid':rngs[n_train:]}
        
train_ds=ML.Dataset(mean_ds.q.isel(time=DA_days),DA_exp.Nx_DA,
                    B_ens_ds.B_ens.isel(time=DA_it,y=i_y,x=i_x),i_y,i_x,
                    partition['train'],ml_std_ds.q_std.data,ml_std_ds.B_std.data,
                    in_ch,out_ch,B_size=B_size,B_start=B_start)
valid_ds=ML.Dataset(mean_ds.q.isel(time=DA_days),DA_exp.Nx_DA,
                    B_ens_ds.B_ens.isel(time=DA_it,y=i_y,x=i_x),i_y,i_x,
                    partition['valid'],ml_std_ds.q_std.data,ml_std_ds.B_std.data,
                    in_ch,out_ch,B_size=B_size,B_start=B_start)

params = {'batch_size':8192,'num_workers':8,'shuffle':True}
training_generator = torch.utils.data.DataLoader(train_ds, **params)
validation_generator = torch.utils.data.DataLoader(valid_ds, **params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features=8
Ulevels=3
if Ulevels==3:
    model=ML.Unet(in_ch=len(in_ch),out_ch=len(out_ch),features=features)
elif Ulevels==2:
    model=ML.Unet_2L(in_ch=len(in_ch),out_ch=len(out_ch),features=features)
model=model.to(device)
print(device)
os.makedirs('./ML/{}/{}L{}f'.format(DA_exp.file_name(),Ulevels,features),exist_ok=True)

# check keras-like model summary using torchsummary
summary(model, input_size=(len(in_ch),train_ds.B_size,train_ds.B_size))

criterion = torch.nn.MSELoss() # MSE loss function
optimizer = optim.Adam(model.parameters(), lr=0.002)

model=model.double()
n_epochs = 200 #Number of epocs
validation_loss = list()
train_loss = list()
start_epoch=0
if start_epoch>0:
    model_file='./ML/{}/{}L_{}f/unet_epoch{}_in{}_out{}_B{}_{}.pt'.format(
        DA_exp.file_name(),Ulevels,features,start_epoch,''.join(map(str,in_ch)),
        ''.join(map(str,out_ch)),B_size,DA_exp.file_name())
    print(model_file)
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
# time0 = time()  
for epoch in range(start_epoch+1, n_epochs + 1):
    train_loss.append(ML.train_model(model,criterion,training_generator,optimizer,device))
    validation_loss.append(ML.test_model(model,criterion,validation_generator,optimizer,device))
    torch.save(model.state_dict(), './ML/{}/{}L{}f/unet_epoch{}_in{}_out{}_B{}_{}.pt'.\
        format(DA_exp.file_name(),Ulevels,features,epoch,''.join(map(str,in_ch)),
               ''.join(map(str,out_ch)),B_size,DA_exp.file_name()))