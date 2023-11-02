from distutils.log import error
from tokenize import endpats
import xarray as xr
import numpy as np
import pyqg
from numpy.random import default_rng
from os.path import exists
from pathlib import Path
from numba import jit
from scipy.interpolate import griddata
from tqdm import tqdm
import torch
import multiprocessing as mp
import psutil

rng = default_rng()

read_data_dir='./data'
save_data_dir='./output'
# read_data_dir='/work/Feiyu.Lu/PyQG/data'
# save_data_dir='/work/Feiyu.Lu/PyQG/data'
model_para={'rek':3.5E-8,'delta':0.05,'beta':0.5E-11}

year=int(60*60*24*365)
default_model=pyqg.QGModel()
year_step=year/int(default_model.dt)

# Data assimilation class for DA related functions
class DA_exp():
    def __init__(self,Nx_truth=64,**kwargs):
        '''
        Nx_truth [int]: Grid size for "truth" model
        Nx_DA [int]: Grid size for DA model
        obs_freq [int]: cycle length (days) of synthetic observations
        obs_err [float,float]: standard deviations for random obs errors of both levels
        DA_method [str]: 'NoDA'/'3DVar'/'EnKF'/'UnetKF'
            kwargs for EnKF: inflate [float,float], save_B [logical]
            kwargs for UnetKF: B_alpha [float], R_training [int], R_DA [int], save_B [logical]
        DA_frequency [int]: cycle length (Days) of DA
        nens [int]: ensemble size (1 for 3DVar and UnetKF)
        R_W [int]: localization radius (km)
        '''
        self.Nx_truth=Nx_truth
        if 'Nx_DA' in kwargs:
            self.Nx_DA=kwargs['Nx_DA']
        if 'nobs' in kwargs:
            self.nobs=kwargs['nobs']
        if 'obs_freq' in kwargs:
            self.obs_freq=kwargs['obs_freq']
        if 'obs_err' in kwargs:
            self.obs_err=kwargs['obs_err']
        if 'DA_method' in kwargs:
            self.DA_method=kwargs['DA_method']
            if self.DA_method=='UnetKF':
                self.B_alpha=kwargs['B_alpha']
                self.R_training=kwargs['R_training']
                self.R_DA=kwargs['R_DA']
                self.training_exp=kwargs['training_exp']
            if self.DA_method=='EnKF' or self.DA_method=='UnetKF':
                self.save_B=kwargs['save_B'] if 'save_B' in kwargs else False
                self.inflate=kwargs['inflate']
            if 'training_var' in kwargs:
                self.training_var=kwargs['training_var']
            else:
                self.training_var=''
        if 'nens' in kwargs:
            self.nens=kwargs['nens']
        if 'DA_freq' in kwargs:
            self.DA_freq=kwargs['DA_freq']
            self.DA_cycle=12*self.DA_freq
        if 'R_W' in kwargs:
            self.R_W=kwargs['R_W']

    def ens_spinup(self,years=10,save_netcdf=True,overwrite=False):
        '''Spin up model ensemble'''
        
        ens = Ensemble([pyqg.QGModel(nx=self.Nx_truth,**model_para) for i in range(self.nens)])
        for model in ens.models:
            model.q=model.q+rng.standard_normal((model.q.shape))*1e-10
        ds_spinup=ens.run_for_steps(years*int(year_step), save_every=years*int(year_step))
        q_init=ds_spinup.q[:,-1,:,:,:]
        if save_netcdf:
            file_name='{}/IC_q_Nx{}_ens{}.nc'.format(save_data_dir,self.Nx_truth,self.nens)
            if not exists(file_name) or overwrite:
                q_init.to_netcdf(file_name)
        return q_init
    
    def generate_truth(self,years,save_every=12,var_list=['q','u','v','Qy']):
        q_init_file='{}/IC_q_Nx{}_ens{}.nc'.format(read_data_dir,self.Nx_truth,self.nens)
        if exists(q_init_file):
            q_init=xr.open_dataarray(q_init_file)
        else:
            q_init=self.ens_spinup()
        ens=Ensemble([pyqg.QGModel(nx=self.Nx_truth,**model_para) for i in range(self.nens)])
        for i,model in enumerate(ens.models):
            model.q=q_init[i,...].data
        ds_truth=ens.run_for_steps(years*int(year_step), save_every=save_every, var_list=var_list)

        file_name='{}/Truth_Nx{}_{}years.nc'.format(save_data_dir,self.Nx_truth,years)
        if not exists(file_name):
            ds_truth.to_netcdf(file_name)
                
        return ds_truth
    
    def obs_name(self,folder=''):
        '''file name for the synthetic observations'''
        obs_name='{}/{}/Obs_Nx{:d}_freq{}_nobs{:s}_err{:d}E{:d}_{:d}E{:d}.nc'.format(
            read_data_dir,folder,self.Nx_truth,self.obs_freq,'_'.join(map(str,self.nobs)),
            self.obs_err[0],self.obs_err[1],self.obs_err[2],self.obs_err[3])
        return obs_name
    
    def read_obs(self,folder=''):
        print(self.obs_name(folder=folder))
        obs_ds=xr.open_dataset(self.obs_name(folder=folder))
        return obs_ds

    def read_truth(self,years,interp=False,folder=''):
        if self.Nx_DA==self.Nx_truth:
            truth_file='{}/{}/Truth_Nx{}_{}years.nc'.format(read_data_dir,folder,self.Nx_truth,years)
        else:
            if interp:
                truth_file='{}/{}/Truth_Nx{}_from_Nx{}_{}years.nc'.format(read_data_dir,folder,self.Nx_DA,self.Nx_truth,years)
            else:
                truth_file='{}/{}/Truth_Nx{}_{}years.nc'.format(read_data_dir,folder,self.Nx_truth,years)
    
        truth_ds=xr.open_dataset(truth_file)
        truth_ds.attrs['truth_file']=truth_file

        return truth_ds

    def read_control(self,years):
        if self.Nx_DA==self.Nx_truth:
            truth_file='{}/Truth_Nx{}_{}years.nc'.format(read_data_dir,self.Nx_DA,years)
        else:
            truth_file='{}/Truth_Nx{}_from_Nx{}_{}years.nc'.format(read_data_dir,self.Nx_DA,self.Nx_truth,years)
    
        truth_ds=xr.open_dataset(truth_file)
        truth_ds.attrs['truth_file']=truth_file

        return truth_ds
    
    def hires_to_lores(self,years=10,save_netcdf=True):
        truth_ds=self.read_truth(years=years,interp=False)
        print(truth_ds.q)
        x_truth,y_truth=np.meshgrid(truth_ds.x,truth_ds.y)
        nx_truth,ny_truth=len(truth_ds.x),len(truth_ds.y)
        
        model_DA=pyqg.QGModel(nx=self.Nx_DA,**model_para)
        x_DA,y_DA=np.meshgrid(model_DA.x,model_DA.y)
        nx_DA,ny_DA=len(model_DA.x),len(model_DA.y)
        q_shape=truth_ds.q.shape
        
        q_low=np.empty((nx_DA,ny_DA,q_shape[2],q_shape[1],q_shape[0]))
        q=truth_ds.q.transpose('x','y','lev','time','model')
        batch=500
        total=len(q.time)
        for i in tqdm(np.arange(int(total/batch)+1)):
            q_slice=q.isel(time=slice(i*batch,(i+1)*batch)).squeeze()
            # print(q_slice.shape)
            q_interp=griddata((x_truth.flatten(),y_truth.flatten()), q_slice.data.reshape((nx_truth*ny_truth,-1)),
                            (model_DA.x,model_DA.y), method='linear')
            # print(q_interp.shape)
            q_low[:,:,:,i*batch:(i+1)*batch,:]=q_interp.reshape((nx_DA,ny_DA,q_slice.shape[2],q_slice.shape[3]))[:,:,:,:,None]

        q_low_da=xr.DataArray(q_low,dims=['x','y','lev','time','model'],
                              coords=[model_DA.x[0,:],model_DA.y[:,0],truth_ds.lev,truth_ds.time,truth_ds.model])
        q_low_da=q_low_da.transpose(*truth_ds.q.dims)
        q_low_ds=xr.Dataset({'q':q_low_da},attrs={'source_file':truth_ds.attrs['truth_file']})
        
        if save_netcdf:
            q_low_ds.to_netcdf('{}/Truth_Nx{}_from_Nx{}_{}years.nc'.
                            format(save_data_dir,self.Nx_DA,self.Nx_truth,years))
        
        return q_low_ds
    
    def generate_obs(self,years=10,save_netcdf=True,overwrite=False):
        """ Sample observations from a "truth" control simulations
        """
    
        truth_ds=self.read_truth(years=years,interp=False)
        
        obs_time=truth_ds.time.isel(time=slice(self.obs_freq-1,None,self.obs_freq))
        obs_days=np.arange(self.obs_freq-1,len(truth_ds.time),self.obs_freq)
        n_time=len(obs_time)
        q_truth=truth_ds.q.isel(model=0).sel(time=obs_time).squeeze()
        Nxy=len(q_truth.x)*len(q_truth.y)
        obs_lev=len(self.nobs)

        obs_q=np.zeros((n_time,sum(self.nobs)))
        obs_x=np.zeros((n_time,sum(self.nobs)))
        obs_y=np.zeros((n_time,sum(self.nobs)))
        obs_xi=np.zeros((n_time,sum(self.nobs)),dtype=int)
        obs_yi=np.zeros((n_time,sum(self.nobs)),dtype=int)
        obs_li=np.zeros((n_time,sum(self.nobs)),dtype=int)
        obs_std=np.zeros((n_time,sum(self.nobs)))
        obs_err_std=[self.obs_err[0]*10**(self.obs_err[1]),
                     self.obs_err[2]*10**(self.obs_err[3])]
        obs_err_std_da=xr.DataArray(obs_err_std,coords=[q_truth.lev])
        obs_errors_da=xr.DataArray(rng.standard_normal((q_truth.shape)),
                                coords=q_truth.coords)
        q_perturbed=q_truth+obs_errors_da*obs_err_std_da
        
        for time in range(n_time):
            rngs=rng.permutation(Nxy)
            for l in range(obs_lev):
                if l==0:
                    ind=np.arange(0,self.nobs[0])
                elif l==1:
                    ind=np.arange(self.nobs[0],self.nobs[0]+self.nobs[1])
                obs_xi[time,ind]=rngs[0:self.nobs[l]]%self.Nx_truth
                obs_yi[time,ind]=rngs[0:self.nobs[l]]//self.Nx_truth
                obs_li[time,ind]=l
                obs_std[time,ind]=obs_err_std[l]
                obs_y[time,ind]=q_truth.y.data[obs_yi[time,ind]]
                obs_x[time,ind]=q_truth.x.data[obs_xi[time,ind]]
                obs_q[time,ind]=q_perturbed.data[time,l,obs_yi[time,ind],obs_xi[time,ind]]

        obs_ds = xr.Dataset({"q": (["day", "obs"], obs_q),
                            "xi": (["day", "obs"], obs_xi),
                            "yi": (["day", "obs"], obs_yi),
                            "li": (["day", "obs"], obs_li),
                            'err_std': (["day","obs"], obs_std),
                            'time':(["day"],obs_time.data)},
                            coords={'day': obs_days,
                                    'obs': np.arange(sum(self.nobs)),
                                    'x': q_truth.x,
                                    'y': q_truth.y},
                            attrs={'truth_file':truth_ds.attrs['truth_file'],
                                   'nobs':' '.join(map(str,self.nobs)),
                                   'obs error':'{:d}E{:d} {:d}E{:d}'.format(self.obs_err[0],self.obs_err[1],self.obs_err[2],self.obs_err[3]),
                                   'obs_freq':self.obs_freq})
        if save_netcdf:
            file_name='{:s}/Obs_Nx{:d}_freq{}_nobs{:s}_err{:d}E{:d}_{:d}E{:d}.nc'.format(
                save_data_dir,self.Nx_truth,self.obs_freq,'_'.join(map(str,self.nobs)),
                self.obs_err[0],self.obs_err[1],self.obs_err[2],self.obs_err[3])
            if not exists(file_name) or overwrite:
                obs_ds.to_netcdf(file_name)

        return obs_ds
        
    def file_name(self):
        '''file name specific to the DA experiment'''
        if self.DA_method=='EnKF':
            f='{}_Nx{}_from_Nx{}_ens{}_freq{}_relax{}_R{}_nobs{}_err{}E{}_{}E{}'.\
                format(self.DA_method,self.Nx_DA,self.Nx_truth,self.nens,self.DA_freq,self.inflate[1],self.R_W,'_'.join(map(str,self.nobs)),
                self.obs_err[0],self.obs_err[1],self.obs_err[2],self.obs_err[3])
        elif self.DA_method=='3DVar':
            f='{}_Nx{}_from_Nx{}_freq{}_R{}_nobs{}_err{}E{}_{}E{}'.\
                format(self.DA_method,self.Nx_DA,self.Nx_truth,self.DA_freq,self.R_W,'_'.join(map(str,self.nobs)),
                self.obs_err[0],self.obs_err[1],self.obs_err[2],self.obs_err[3])
        elif self.DA_method=='UnetKF':
            f='{}_Nx{}_from_Nx{}_ens{}_freq{}_relax{}_hybrid{}_R{}_nobs{}_err{}E{}_{}E{}'.\
                format(self.DA_method,self.Nx_DA,self.Nx_truth,self.nens,self.DA_freq,self.inflate[1],self.B_alpha,self.R_W,'_'.join(map(str,self.nobs)),
                self.obs_err[0],self.obs_err[1],self.obs_err[2],self.obs_err[3])
        else:
            f='{}_Nx{}_from_Nx{}_nobs{}_err{}E{}_{}E{}'.\
                format(self.DA_method,self.Nx_DA,self.Nx_truth,'_'.join(map(str,self.nobs)),
                self.obs_err[0],self.obs_err[1],self.obs_err[2],self.obs_err[3])
        return f

    def file_name_short(self):
        if self.DA_method=='EnKF':
            f='{}_ens{}_relax{}_R{}'.\
                format(self.DA_method,self.nens,self.inflate[1],self.R_W)
        elif self.DA_method=='3DVar':
            f='{}_R{}'.\
                format(self.DA_method,self.R_W)
        elif self.DA_method=='UnetKF':
            f='{}_ens{}_relax{}_R{}'.\
                format(self.DA_method,self.nens,self.inflate[1],self.R_W)
        else:
            f='{}'.\
                format(self.DA_method)
        return f

    def file_name_label(self):
        if self.DA_method=='EnKF':
            # f='{}\nens{}'.\
            #     format(self.DA_method,self.nens)
            f='{}\nens{}\nrel{}\nR{}'.\
                format(self.DA_method,self.nens,self.inflate[1],self.R_W)
        elif self.DA_method=='3DVar':
            # f='{}'.\
            #     format(self.DA_method)
            f='{}\nR{}'.\
                format(self.DA_method,self.R_W)
        elif self.DA_method=='UnetKF':
            # f='{}\nens{}'.\
            #     format(self.DA_method,self.nens)
            f='{}\nens{}\nrel{}\nR{}'.\
                format(self.DA_method,self.nens,self.inflate[1],self.R_W)
        else:
            f='{}'.\
                format(self.DA_method)
        return f
        
    def read_mean(self,folder=''):
        file_name=self.file_name()
        if folder=='training':
            mean_file='{}/{}/EnsMean_{}.nc'.format(read_data_dir,folder,file_name)
        elif folder!='':
            if self.DA_method=='UnetKF':
                mean_file='{}/{}_Nx{}_{}_ens{}/EnsMean_{}.nc'.format(read_data_dir,self.DA_method,self.Nx_DA,self.Nx_truth,folder,file_name)
            else:
                mean_file='{}/{}/EnsMean_{}.nc'.format(read_data_dir,self.DA_method,file_name)
        else:
            mean_file='{}/{}/EnsMean_{}.nc'.format(read_data_dir,self.DA_method,file_name)
        print(mean_file)
        mean_ds=xr.open_dataset(mean_file)
        return mean_ds
              
    def read_std(self,folder=''):
        if self.nens>1:
            file_name=self.file_name()
            if folder!='':
                if self.DA_method=='UnetKF':
                    std_file='{}/{}_Nx{}_{}_ens{}/EnsStd_{}.nc'.format(read_data_dir,self.DA_method,self.Nx_DA,self.Nx_truth,folder,file_name)
                else:
                    std_file='{}/{}/EnsStd_{}.nc'.format(read_data_dir,self.DA_method,file_name)
            else:
                std_file='{}/{}/EnsStd_{}.nc'.format(read_data_dir,self.DA_method,file_name)
            std_ds=xr.open_dataset(std_file)
        else:
            error('no ensemble spread')
            
        return std_ds
        
    def init_DA(self,DA_start,ic_nens=100,ic_seed=0):
        '''
        Get initial conditions for DA experiment
        If DA_start==0, read from existing IC files
        If DA_start>0, continue from previous DA experiment (only ensemble mean at the moment)
        '''
        self.ens = Ensemble([pyqg.QGModel(nx=self.Nx_DA,**model_para) for i in range(self.nens)])
        self.obs_ds=self.read_obs()
        
        if DA_start==0:
            print('{}/IC_q_Nx{}_ens{}.nc'.format(read_data_dir,self.Nx_DA,ic_nens))
            q_init=xr.open_dataarray('{}/IC_q_Nx{}_ens{}.nc'.format(read_data_dir,self.Nx_DA,ic_nens))
            for i,model in enumerate(self.ens.models):
                model.q=q_init.isel(model=(i+ic_seed)%len(q_init.model)).data
        else:
            print('{}/EnsMean_{}.nc'.format(read_data_dir,self.file_name()))
            q_DA_ds=xr.open_dataset('{}/EnsMean_{}.nc'.format(read_data_dir,self.file_name()))
            for i,model in enumerate(self.ens.models):
                model.q=q_DA_ds.q.isel(time=DA_start).data+rng.standard_normal((model.q.shape))*1e-10
                
        return self.ens
    
    def assimilation(self,forecast_ds,day,**kwargs):
        prior=forecast_ds.q.isel(time=-1)
        Nlev=len(forecast_ds.lev)
        prior_data=prior.data.reshape((self.nens,-1))

        [H,R_obs]=ObsOp(forecast_ds,self.obs_ds,day)
        obs_q=self.obs_ds.q.sel(day=day).data

        if self.DA_method=='3DVar':
            B_static=self.B_ds.cov.data*self.W_ds.W.data
            posterior=Lin3dvar(prior_data.squeeze(),obs_q,H,R_obs,B_static)
        elif self.DA_method=='EnKF':
            if self.inflate[0]>1.00001:
                prior_data=ens_inflate(prior_data,prior_data,1,self.inflate[0])
            B_ens=calculate_cov(prior_data)
            B_ens_loc=mat_mul(B_ens,self.W_ds.W.data)
            posterior=EnKF(prior_data,obs_q,H,R_obs,B_ens_loc)
            if self.inflate[1]>0.00001:
                posterior=ens_inflate(prior_data,posterior,2,self.inflate[1])
            
            if self.save_B:
                B_mat=Localize_B(B_ens,self.Nx_DA,Nlev,self.B_loc)
                Path('{}/{}'.format(save_data_dir,self.file_name())).mkdir(exist_ok=True)
                B_filename='{}/{}/B_ens_day{:04d}.nc'.format(save_data_dir,self.file_name(),day)
                B_ens_da=xr.DataArray(B_mat,coords=[forecast_ds.lev,forecast_ds.y,forecast_ds.x,forecast_ds.lev,
                                                    np.arange(-self.B_loc,self.B_loc+1),np.arange(-self.B_loc,self.B_loc+1)],
                                      dims=['lev','y','x','lev_d','y_d','x_d'])
                B_ens_da = B_ens_da.assign_coords(time=forecast_ds.time[-1])
                B_ens_da = B_ens_da.expand_dims('time')
                B_ens_ds=xr.Dataset({'B_ens':B_ens_da})
                B_ens_ds.to_netcdf(B_filename,unlimited_dims=['time'])
                
        elif self.DA_method=='UnetKF':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ml_model=kwargs['ml_model'].to(device)
            ml_std=kwargs['std_file']
            analysis_prev=kwargs['analysis_prev']
            DA_Unet_size=int(self.R_DA/2)*4
            DA_B_size=self.R_DA*2+1
            training_Unet_size=int(self.R_training/2)*4
            
            # Localized q of the size of the U-Net are taken from the full q fields
            q_data=np.zeros((self.Nx_DA,self.Nx_DA,Nlev,DA_Unet_size,DA_Unet_size))
            # q_data=np.zeros((self.Nx_DA,self.Nx_DA,4,DA_Unet_size,DA_Unet_size))
            for i in np.arange(self.Nx_DA):
                for j in np.arange(self.Nx_DA):
                    q_data[j,i,0,:,:]=localize_q(prior.mean('model').isel(lev=0),j,i,self.Nx_DA,self.R_DA)[...,0:DA_Unet_size,0:DA_Unet_size]
                    q_data[j,i,1,:,:]=localize_q(prior.mean('model').isel(lev=1),j,i,self.Nx_DA,self.R_DA)[...,0:DA_Unet_size,0:DA_Unet_size]
                    # q_data[j,i,2,:,:]=q_data[j,i,0,:,:]-localize_q(analysis_prev.mean('model').isel(lev=0),j,i,self.Nx_DA,self.R_DA)[...,0:DA_Unet_size,0:DA_Unet_size]
                    # q_data[j,i,3,:,:]=q_data[j,i,1,:,:]-localize_q(analysis_prev.mean('model').isel(lev=1),j,i,self.Nx_DA,self.R_DA)[...,0:DA_Unet_size,0:DA_Unet_size]

            # Normalization
            q_stacked=q_data.reshape((-1,Nlev,DA_Unet_size,DA_Unet_size))/ml_std.q_std.data[None,:,None,None]
            # print(q_stacked.shape)

            
            B_data=np.zeros((self.Nx_DA,self.Nx_DA,3,DA_B_size,DA_B_size))
            B_stacked=B_data.reshape((-1,3,DA_B_size,DA_B_size))
            B_pred_all=np.zeros((B_stacked.shape[0],B_stacked.shape[1],training_Unet_size,training_Unet_size))
            chunk=2048
            total=q_stacked.shape[0]
            B_std=np.array([ml_std.B_std.data[0,0],ml_std.B_std.data[0,1],ml_std.B_std.data[1,1]])
            
            # q is interpolated if the DA model has different grids than than model used to train the U-Net
            # Here it is exclusively downsampled to half the resolution (double the drig size)
            if self.training_exp.Nx_DA!=self.Nx_DA:
                q_interp=np.zeros((q_stacked.shape[0],q_stacked.shape[1],training_Unet_size,training_Unet_size))
                # q_interp=q_stacked[:,:,0::2,0::2]
                q_interp=1/9*q_stacked[:,:,2:-2:2,2:-2:2]+\
                    1/9*(q_stacked[:,:,1:-3:2,2:-2:2]+q_stacked[:,:,3:-2:2,2:-2:2]+\
                         q_stacked[:,:,2:-2:2,1:-3:2]+q_stacked[:,:,2:-2:2,3:-2:2])+\
                    1/9*(q_stacked[:,:,1:-3:2,1:-3:2]+q_stacked[:,:,1:-3:2,3:-2:2]+\
                         q_stacked[:,:,3:-2:2,1:-3:2]+q_stacked[:,:,3:-2:2,3:-2:2])
            else:
                q_interp=q_stacked
            
            # Covariance matrices are predicted by the U-Net using the localized (and interpolated) q as input
            for i in np.arange(int(total/chunk)+1):
                # B_pred=ml_model(torch.from_numpy(q_interp[i*chunk:(i+1)*chunk,:,:,:]).double()).detach().numpy()*B_std[None,:,None,None]
                B_pred=ml_model(torch.from_numpy(q_interp[i*chunk:(i+1)*chunk,:,:,:]).double().to(device)).to('cpu').detach().numpy()*B_std[None,:,None,None]
                B_pred_all[i*chunk:(i+1)*chunk,:,:,:]=B_pred
        
            # print(B_pred_all.shape)
            # print(B_stacked.shape)
            
            # The covariance matrices are interpolated back to the DA resoultion from the U-Net predicted resolution
            if self.training_exp.Nx_DA!=self.Nx_DA:
                # X_training=np.arange(0,DA_Unet_size,2)
                # Y_training=np.arange(0,DA_Unet_size,2)
                X_training=np.arange(2,DA_Unet_size-2,2)
                Y_training=np.arange(2,DA_Unet_size-2,2)
                X_training_mesh,Y_training_mesh=np.meshgrid(X_training,Y_training)
                X_DA=np.arange(0,DA_Unet_size)
                Y_DA=np.arange(0,DA_Unet_size)
                X_DA_mesh,Y_DA_mesh=np.meshgrid(X_DA,Y_DA)
                B_interp=griddata((X_training_mesh.flatten(),Y_training_mesh.flatten()),
                                  B_pred_all.reshape((B_stacked.shape[0]*B_stacked.shape[1],-1)).transpose(),
                                  (X_DA_mesh,Y_DA_mesh),method='linear',fill_value=0.0)
                # print(B_interp.shape)
                B_stacked[:,:,0:DA_Unet_size,0:DA_Unet_size]=B_interp.transpose().reshape((B_stacked.shape[0],B_stacked.shape[1],DA_Unet_size,DA_Unet_size))
                
            else:
                B_stacked[:,:,0:DA_Unet_size,0:DA_Unet_size]=B_pred_all
                
            # The localized covariance matrices are put back into the global form 
            B_Unet=globalize_B(B_stacked.reshape(B_data.shape),self.Nx_DA,Nlev,self.R_DA)
            B_Unet_loc=(B_Unet+self.B_alpha*self.B_ds.cov.data)*self.W_ds.W.data
            
            if self.nens==1:
                posterior=Lin3dvar(prior_data.squeeze(),obs_q,H,R_obs,B_Unet_loc)
            elif self.nens>1:
                posterior=EnKF(prior_data,obs_q,H,R_obs,B_Unet_loc)
                
            if self.inflate[1]>0.00001:
                posterior=ens_inflate(prior_data,posterior,2,self.inflate[1])
                
            if self.save_B:
                Path('{}/{}'.format(save_data_dir,self.file_name())).mkdir(exist_ok=True)
                B_filename='{}/{}/B_ens_day{:04d}.nc'.format(save_data_dir,self.file_name(),day)
                B_pred_da=xr.DataArray(B_stacked.reshape(B_data.shape),
                                       coords=[forecast_ds.y,forecast_ds.x,np.array([0,1,2]),
                                               np.arange(-self.R_DA,self.R_DA+1),np.arange(-self.R_DA,self.R_DA+1)],
                                       dims=['y','x','lev_d','y_d','x_d'])
                B_pred_da = B_pred_da.assign_coords(time=forecast_ds.time[-1])
                B_pred_da = B_pred_da.expand_dims('time')
                B_ens_ds=xr.Dataset({'DA_Unet_size':B_pred_da})
                B_ens_ds.to_netcdf(B_filename,unlimited_dims=['time'])
            
        posterior=posterior.reshape(prior.shape)
        return posterior

    def run_exp(self,DA_days=365,DA_start=0,ic_seed=0,**kwargs):
        self.init_DA(DA_start,ic_seed=ic_seed)
        
        B_filename='{}/B_Nx{}_100years_2lev.nc'.format(read_data_dir,self.Nx_DA)
        W_filename='{}/W_Nx{}_L{}.nc'.format(read_data_dir,self.Nx_DA,self.R_W)
        self.B_loc=max(int(np.ceil(2*self.R_W*1000/(self.ens.models[0].L/self.ens.models[0].nx))),8)
        self.B_ds=xr.open_dataset(B_filename)
        self.W_ds=xr.open_dataset(W_filename)
        DA_kwargs={}
        if 'output_str' in kwargs:
            output_str=kwargs['output_str']
        else:
            output_str=''    
        if self.DA_method=='UnetKF':
            DA_kwargs['ml_model']=kwargs['ml_model']
            DA_kwargs['std_file']=kwargs['ml_std_ds']
            
        for day in tqdm(np.arange(DA_start,DA_days,self.DA_freq)):
            ds_forecast=self.ens.run_for_steps(self.DA_cycle, save_every=12)
            analysis_prev=ds_forecast.q.isel(time=0)
            if self.DA_method=='UnetKF':
                DA_kwargs['analysis_prev']=analysis_prev
            ds_forecast['q_post']=ds_forecast['q'].copy(deep=True)
            DA_day=day+self.DA_freq-1
            if not self.DA_method=='NoDA':
                analysis=self.assimilation(ds_forecast,DA_day,**DA_kwargs)
                ds_forecast.q_post[:,-1,...]=analysis
                for i,m in enumerate(self.ens.models):
                    m.q=analysis[i,...]
                    
            if day==DA_start:
                ds_mean=ds_forecast.mean('model')
                if self.nens>1:
                    ds_std=ds_forecast.std('model')
            else:
                ds_mean=xr.concat([ds_mean,ds_forecast.mean('model')],dim='time')
                if self.nens>1:
                    ds_std=xr.concat([ds_std,ds_forecast.std('model')],dim='time')
                        
        file_name=self.file_name()
        mean_file='{}/{}/EnsMean_{}.nc'.format(save_data_dir,output_str,file_name)
        ds_mean.to_netcdf(mean_file,mode='w')
        if self.nens>1:  
            std_file='{}/{}/EnsStd_{}.nc'.format(save_data_dir,output_str,file_name)    
            ds_std.to_netcdf(std_file,mode='w')
    
# Simple helper for running ensembles pyqg models for a specified number of steps & saving results
class Ensemble():
    def __init__(self, models):
        self.models = models
        self.ens=len(models)
        
    def bunch_step_forward(self,index):
        self.models[index]._step_forward()

    def parallel_step_forward(self):
        self.pool.map(self.bunch_step_forward,range(4))
                
        # procs = []
        # for i in range(4):
        #     model = self.models[i]
        #     proc = mp.Process(target=self.bunch_step_forward, args=(model,))
        #     procs.append(proc)
        #     proc.start()
            
        # for proc in procs:
        #     proc.join() 
        
    def step_forward(self):
        for m in self.models:
            m._step_forward()

    def run_for_steps(self, steps, save_every=1, var_list=['q','u','v','Qy']):
        results = []

        for i in range(steps):
            self.step_forward()
            if (i+1) % save_every == 0:
                results.append(xr.concat([m.to_dataset()[var_list] for m in self.models],dim='model'))

        return xr.concat(results, dim='time')

# Calculate background covariance matrix for 3DVar
def B_calculation_3DVar(Nx=64,years=100,lev=2,save_netcdf=True):
    truth_file='{}/Truth_Nx{}_{}years.nc'.format(read_data_dir,Nx,years)
    try:
        ds_truth=xr.open_dataset(truth_file)
    except:
        print('No such truth file')

    q1=ds_truth.q.isel(lev=slice(0,lev)).squeeze('model').stack(loc=('lev','y','x'))
    nxy=len(ds_truth.x)*len(ds_truth.y)*lev
    q1_mean=q1.mean('time')
    q1_std=(np.sqrt(((q1-q1_mean)**2).sum('time')/(len(q1.time)-1))).data
    q1_std_np=q1_std.reshape((len(q1_std),1))
    q1_var_mat=(q1_std_np@q1_std_np.T).data

    q1_cov=((q1-q1_mean).data.T @ (q1-q1_mean).data)/(len(q1.time)-1)
    q1_corr=q1_cov/q1_var_mat

    ds = xr.Dataset({"cov": (["loc", "loc"], q1_cov),
                    "corr": (["loc", "loc"], q1_corr),
                    'std': (["loc"], q1_std)},
                    coords={"loc": np.arange(nxy),
                            'x':ds_truth.x,
                            'y':ds_truth.y},
                    attrs={'truth_file':truth_file})

    if save_netcdf:
        file_name='{}/B_Nx{}_{}years_{}lev.nc'.format(save_data_dir,Nx,years,lev)
        if not exists(file_name):
            ds.to_netcdf(file_name)
            
    return ds

# Take subset (subdomain) of data from the global matrix (domain)
def localize_q(q,y,x,Nx,B_R):
    # q: global data matrix
    # x,y: center point for the subset
    # Nx: global domain size
    # B_R: radius for the subset
    
    i_y=np.arange(y-B_R,y+B_R+1)
    i_x=np.arange(x-B_R,x+B_R+1)
    i_y1=np.where(i_y>=0,i_y,i_y+Nx)
    i_y2=np.where(i_y1<=Nx-1,i_y1,i_y1-Nx)
    i_x1=np.where(i_x>=0,i_x,i_x+Nx)
    i_x2=np.where(i_x1<=Nx-1,i_x1,i_x1-Nx)
    q_loc=q[...,i_y2,i_x2]

    return q_loc

@jit
def Localize_B(B,Nx:int,Nlev:int,R:int):
    Nxyl=B.shape[0]
    Nxy=int(Nxyl/Nlev)
    Ny=int(Nxy/Nx)
    
    B_mat=np.zeros((Nlev,Ny,Nx,Nlev,2*R+1,2*R+1))
    
    center=np.arange(Nxyl)
    center_l=center//Nxy
    center_y=(center%Nxy)//Nx
    center_x=(center%Nxy)%Nx

    for j in np.arange(-R,R+1):
        for i in np.arange(-R,R+1):
            range_x=center_x+i
            range_x_1=np.where(range_x<Nx,range_x,range_x-Nx)
            range_x_2=np.where(range_x_1>-1,range_x_1,range_x_1+Nx)
            range_y=center_y+j
            range_y_1=np.where(range_y<Ny,range_y,range_y-Ny)
            range_y_2=np.where(range_y_1>-1,range_y_1,range_y_1+Ny)
            range_xy_l1=center_l*Nxy+range_y_2*Nx+range_x_2
            range_xy_l2=(1-center_l)*Nxy+range_y_2*Nx+range_x_2

            B_mat[center_l,center_y,center_x,center_l,j+R,i+R]=B[center,range_xy_l1]
            B_mat[center_l,center_y,center_x,1-center_l,j+R,i+R]=B[center,range_xy_l2]
    
    return B_mat

def globalize_B(B,Nx:int,Nlev:int,R:int):
    Nxyl=Nlev*Nx*Nx
    Nxy=Nx*Nx
    Ny=int(Nxy/Nx)
    
    B_mat=np.zeros((Nlev*Nx*Nx,Nlev*Nx*Nx))
    
    center=np.arange(Nxy)
    center_y=center//Nx
    center_x=center%Nx

    for j in np.arange(-R,R+1):
        for i in np.arange(-R,R+1):
            range_x=center_x+i
            range_x_1=np.where(range_x<Nx,range_x,range_x-Nx)
            range_x_2=np.where(range_x_1>-1,range_x_1,range_x_1+Nx)
            range_y=center_y+j
            range_y_1=np.where(range_y<Ny,range_y,range_y-Ny)
            range_y_2=np.where(range_y_1>-1,range_y_1,range_y_1+Ny)
            range_xy_l1=range_y_2*Nx+range_x_2
            range_xy_l2=Nxy+range_y_2*Nx+range_x_2

            B_mat[center,range_xy_l1]=B[center_y,center_x,0,j+R,i+R]
            B_mat[center,range_xy_l2]=B[center_y,center_x,1,j+R,i+R]
            B_mat[range_xy_l2,center]=B[center_y,center_x,1,j+R,i+R]
            B_mat[center+Nxy,range_xy_l1+Nxy]=B[center_y,center_x,2,j+R,i+R]
    
    return B_mat

# Observation (forward) operator for model space-observation space conversion
def ObsOp(forecast_ds,obs_ds,day):
    nx_DA=forecast_ds.x.size
    ny_DA=forecast_ds.y.size
    nx_obs=obs_ds.x.size
    ny_obs=obs_ds.y.size
    nxy_DA=nx_DA*ny_DA
    nxyl_DA=nxy_DA*forecast_ds.lev.size
    nobs=obs_ds.obs.size
    
    xi=obs_ds.xi.sel(day=day).data
    yi=obs_ds.yi.sel(day=day).data
    li=obs_ds.li.sel(day=day).data
    err_std=obs_ds.err_std.sel(day=day).data
    R=np.zeros((nobs,nobs),np.float64)
    R[np.arange(nobs),np.arange(nobs)]=err_std*err_std
    
    if nx_DA==nx_obs and ny_DA==ny_obs:
        H=np.zeros((nobs,nxyl_DA),np.float64)
        H[np.arange(nobs),li*nxy_DA+yi*nx_DA+xi]=1
    else:
        x_obs=obs_ds.x[xi]
        y_obs=obs_ds.y[yi]
        x_DA=forecast_ds.x
        y_DA=forecast_ds.y
        x_d=x_DA[1]-x_DA[0]
        y_d=y_DA[1]-y_DA[0]
        x_DA_ex=np.concatenate((np.array([x_DA[0]-x_d]),x_DA,np.array([x_DA[-1]+x_d])))
        y_DA_ex=np.concatenate((np.array([y_DA[0]-y_d]),y_DA,np.array([y_DA[-1]+y_d])))
        xi_DA=np.interp(x_obs,x_DA_ex,np.arange(-1,nx_DA+1))
        yi_DA=np.interp(y_obs,y_DA_ex,np.arange(-1,ny_DA+1))

        xi_DA_1=np.floor(np.where(xi_DA>=0,xi_DA,xi_DA+nx_DA)).astype(np.int_)
        xi_DA_2=np.floor(np.where(xi_DA<nx_DA-1,xi_DA,xi_DA-nx_DA)).astype(np.int_)
        
        yi_DA_1=np.floor(np.where(yi_DA>=0,yi_DA,yi_DA+ny_DA)).astype(np.int_)
        yi_DA_2=np.floor(np.where(yi_DA<ny_DA-1,yi_DA,yi_DA-ny_DA)).astype(np.int_)
        
        wt_x=xi_DA-np.floor(xi_DA).astype(np.int_)
        wt_y=yi_DA-np.floor(yi_DA).astype(np.int_)
        ind1=yi_DA_1*nx_DA+(xi_DA_1)+li*nxy_DA
        ind2=yi_DA_1*nx_DA+(xi_DA_2+1)+li*nxy_DA
        ind3=(yi_DA_2+1)*nx_DA+(xi_DA_1)+li*nxy_DA
        ind4=(yi_DA_2+1)*nx_DA+(xi_DA_2+1)+li*nxy_DA
        wt1=np.zeros((nobs,nxyl_DA),np.float64)
        wt2=np.zeros((nobs,nxyl_DA),np.float64)
        wt3=np.zeros((nobs,nxyl_DA),np.float64)
        wt4=np.zeros((nobs,nxyl_DA),np.float64)
        
        wt1[np.arange(nobs),ind1]=(1-wt_x)*(1-wt_y)
        wt2[np.arange(nobs),ind2]=wt_x*(1-wt_y)
        wt3[np.arange(nobs),ind3]=(1-wt_x)*wt_y
        wt4[np.arange(nobs),ind4]=wt_x*wt_y
        
        H=wt1+wt2+wt3+wt4

    return [H,R]

@jit(nopython=True)
def calculate_cov(data):
    return np.cov(data.T)

@jit
def mat_mul(a,b):
    return a*b

@jit(nopython=True,parallel=True)
def EnKF(prior,obs,H,R,B):
    
    # The analysis step for the (stochastic) ensemble Kalman filter 
    # with virtual observations

    nens,nxy = prior.shape # n is the state dimension and N is the size of ensemble
    nobs = obs.shape[0] # m is the size of measurement vector

    # compute Kalman gain
    D = H@B@H.T + R
    K = B @ H.T @ np.linalg.inv(D)
    
    # perturb observations
    obs_ens=obs.repeat(nens).reshape(nobs,nens)+\
        np.sqrt(R)@np.random.standard_normal((nobs,nens))
    # compute analysis ensemble
    posterior = prior.T + K @ (obs_ens-H@(prior.T))

    return posterior.T

@jit(nopython=True,parallel=True)
def Lin3dvar(ub,w,H,R,B):
    
    A = R + H@B@(H.T)
    b = (w-H@ub)
    ua = ub + B@(H.T)@np.linalg.solve(A,b) #solve a linear system
        
    return ua

@jit(nopython=True,parallel=True)
def ens_inflate(prior,posterior,opt,factor):
    
    inflated=np.zeros(prior.shape)
    nens,nxy=prior.shape
    if opt == 1: 
        mean_prior=(prior.T.sum(axis=-1)/nens).repeat(nens).reshape(nxy,nens).T
        inflated=prior+factor*(prior-mean_prior)
    
    elif opt == 2:
        mean_prior=(prior.T.sum(axis=-1)/nens).repeat(nens).reshape(nxy,nens).T
        mean_post=(posterior.T.sum(axis=-1)/nens).repeat(nens).reshape(nxy,nens).T
        inflated=mean_post+(1-factor)*(posterior-mean_post)+factor*(prior-mean_prior)      
        
    return inflated

# Generate localization weight matrix
def Localize_weights(Nx=64,R=1.0E5,save_netcdf=True):
    """
    R: localization radius
    """
    model=pyqg.QGModel(nx=Nx,**model_para)
    x=model.x[0,:]
    y=model.y[:,0]
    L=model.L
    Nx=len(x)
    Ny=len(y)
    Nxy=Nx*Ny

    D=np.zeros((Nxy,Nxy))
    W=np.zeros((Nxy,Nxy))
    for i in range(Nxy):
        for j in range(Nxy):
            x1=i%Nx
            y1=i//Nx
            x2=j%Nx
            y2=j//Nx
            D[i,j]=get_dist(x[x1],y[y1],x[x2],y[y2],L)
            
    W=np.vectorize(gaspari_cohn)(D,R)
    W_ds = xr.Dataset({"W": (["loc", "loc"], np.tile(W,(2,2)))},
                    coords={"loc": np.arange(Nxy*2),
                            'x':x,
                            'y':y},
                    attrs={'L':L,
                           'R':R})
    if save_netcdf:
        W_ds.to_netcdf('{}/W_Nx{}_L{:d}.nc'.format(save_data_dir,Nx,int(R/1000)))

    return W_ds

def gaspari_cohn(distance,radius):
    if distance==0:
        weight=1.0
    else: 
        if radius==0:
            weight=0.0
        else:
            ratio=distance/radius
            weight=0.0
            if ratio<=1:
                weight=-ratio**5/4+ratio**4/2+5*ratio**3/8-5*ratio**2/3+1
            elif ratio<=2:
                weight=ratio**5/12-ratio**4/2+5*ratio**3/8+5*ratio**2/3-5*ratio+4-2/3/ratio
    return weight

def get_dist(x1,y1,x2,y2,L):
    xd_abs=abs(x1-x2)
    xd=xd_abs if xd_abs<=L/2 else L-xd_abs
    yd=abs(y1-y2)
    return np.sqrt(xd**2+yd**2)