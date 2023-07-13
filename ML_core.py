from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from DA_core import localize_q
import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self,q_da,Nx,B_da,indy,indx,partition,q_std,B_std,in_ch,out_ch,**kwargs):
        super(Dataset, self).__init__()

        self.q_da=q_da
        self.B_da=B_da
        self.B_R=int(len(B_da.x_d)/2)
        if 'B_size' in kwargs:
            self.B_size=kwargs['B_size']
        else:
            self.B_size=int(self.B_R/2)*4
        if 'B_start' in kwargs:
            self.B_start=kwargs['B_start']
        else:
            self.B_start=0
        self.B_shape=self.B_da.shape
        self.B_nt=len(self.B_da.time)
        self.B_ny=len(self.B_da.y)
        self.B_nx=len(self.B_da.x)
        self.indx=indx
        self.indy=indy
        self.Nx=Nx
        self.B_total=self.B_nt*self.B_ny*self.B_nx
        self.ind=partition
        self.B_std=B_std
        self.q_std=q_std
        self.in_ch=in_ch
        self.out_ch=out_ch
        
    def __len__(self):
        return len(self.ind)
        
    def __getitem__(self, i):
        B=np.empty((len(self.out_ch),self.B_size,self.B_size)).astype(np.double)
        q=np.empty((len(self.in_ch),self.B_size,self.B_size)).astype(np.double)
        i_t=self.ind[i]//(self.B_ny*self.B_nx)
        i_y=(self.ind[i]%(self.B_ny*self.B_nx))//self.B_nx
        i_x=(self.ind[i]%(self.B_ny*self.B_nx))%self.B_nx
        for i_ch,ch in enumerate(self.out_ch):
            if ch==0:
                B[i_ch,...]=self.B_da.isel(time=i_t,y=i_y,x=i_x,lev=0,lev_d=0).\
                    isel(x_d=slice(self.B_start,self.B_start+self.B_size),
                        y_d=slice(self.B_start,self.B_start+self.B_size))/self.B_std[0,0]
            if ch==1:
                B[i_ch,...]=self.B_da.isel(time=i_t,y=i_y,x=i_x,lev=0,lev_d=1).\
                    isel(x_d=slice(self.B_start,self.B_start+self.B_size),
                         y_d=slice(self.B_start,self.B_start+self.B_size))/self.B_std[0,1]
            if ch==2:
                B[i_ch,...]=self.B_da.isel(time=i_t,y=i_y,x=i_x,lev=1,lev_d=1).\
                    isel(x_d=slice(self.B_start,self.B_start+self.B_size),
                         y_d=slice(self.B_start,self.B_start+self.B_size))/self.B_std[1,1]
        if len(self.in_ch)==2:
            q[0,...]=localize_q(self.q_da.isel(time=i_t,lev=0),self.indy[i_y],self.indx[i_x],self.Nx,self.B_R)\
                [...,self.B_start:self.B_start+self.B_size,self.B_start:self.B_start+self.B_size]/self.q_std[0]
            q[1,...]=localize_q(self.q_da.isel(time=i_t,lev=1),self.indy[i_y],self.indx[i_x],self.Nx,self.B_R)\
                [...,self.B_start:self.B_start+self.B_size,self.B_start:self.B_start+self.B_size]/self.q_std[1]
        elif len(self.in_ch)==4:
            q[0,...]=localize_q(self.q_da.isel(time=i_t,lev=0),self.indy[i_y],self.indx[i_x],self.Nx,self.B_R)\
                [...,self.B_start:self.B_start+self.B_size,self.B_start:self.B_start+self.B_size]/self.q_std[0]
            q[1,...]=localize_q(self.q_da.isel(time=i_t,lev=1),self.indy[i_y],self.indx[i_x],self.Nx,self.B_R)\
                [...,self.B_start:self.B_start+self.B_size,self.B_start:self.B_start+self.B_size]/self.q_std[1]
            q[2,...]=localize_q(self.q_da.isel(time=i_t,lev=0)-self.q_da.isel(time=i_t-1,lev=0),self.indy[i_y],self.indx[i_x],self.Nx,self.B_R)\
                [...,self.B_start:self.B_start+self.B_size,self.B_start:self.B_start+self.B_size]/self.q_std[0]
            q[3,...]=localize_q(self.q_da.isel(time=i_t,lev=1)-self.q_da.isel(time=i_t-1,lev=1),self.indy[i_y],self.indx[i_x],self.Nx,self.B_R)\
                [...,self.B_start:self.B_start+self.B_size,self.B_start:self.B_start+self.B_size]/self.q_std[1]
        return q,B

#double 3x3 convolution 
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3,padding=1),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3,padding=1),
        nn.ReLU(inplace= True),
    )
    return conv

# crop the image(tensor) to equal size 
# as shown in architecture image , half left side image is concated with right side image
def crop_tensor(target_tensor, tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size- delta, delta:tensor_size-delta]

class Unet(nn.Module):
    def __init__(self,in_ch=1,out_ch=1,features=16):
        super(Unet, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(in_ch, features)
        self.dwn_conv2 = dual_conv(features, features*2)
        self.dwn_conv3 = dual_conv(features*2, features*4)
        self.dwn_conv4 = dual_conv(features*4, features*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Right side  (expnsion path) 
        #transpose convolution is used showna as green arrow in architecture image
        self.trans1 = nn.ConvTranspose2d(features*8,features*4, kernel_size=2, stride=2)
        self.up_conv1 = dual_conv(features*8,features*4)
        self.trans2 = nn.ConvTranspose2d(features*4,features*2, kernel_size=2, stride=2)
        self.up_conv2 = dual_conv(features*4,features*2)
        self.trans3 = nn.ConvTranspose2d(features*2,features, kernel_size=2, stride=2)
        self.up_conv3 = dual_conv(features*2,features)
        
        #output layer
        self.out = nn.Conv2d(features, out_ch, kernel_size=1)

    def forward(self, image):

        #forward pass for Left side
        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)      
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        
        #forward pass for Right side

        x = self.trans1(x7)
        y = crop_tensor(x, x5)
        x = self.up_conv1(torch.cat([x,y], 1))
        
        x = self.trans2(x)
        y = crop_tensor(x, x3)
        x = self.up_conv2(torch.cat([x,y], 1))
        
        x = self.trans3(x)
        y = crop_tensor(x, x1)
        x = self.up_conv3(torch.cat([x,y], 1))
        
        x = self.out(x)
        
        return x

class Unet_2L(nn.Module):
    def __init__(self,in_ch=1,out_ch=1,features=16):
        super(Unet_2L, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(in_ch, features)
        self.dwn_conv2 = dual_conv(features, features*2)
        self.dwn_conv3 = dual_conv(features*2, features*4)
        self.dwn_conv4 = dual_conv(features*4, features*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Right side  (expnsion path) 
        #transpose convolution is used showna as green arrow in architecture image
        self.trans1 = nn.ConvTranspose2d(features*8,features*4, kernel_size=2, stride=2)
        self.up_conv1 = dual_conv(features*8,features*4)
        self.trans2 = nn.ConvTranspose2d(features*4,features*2, kernel_size=2, stride=2)
        self.up_conv2 = dual_conv(features*4,features*2)
        self.trans3 = nn.ConvTranspose2d(features*2,features, kernel_size=2, stride=2)
        self.up_conv3 = dual_conv(features*2,features)
        
        #output layer
        self.out = nn.Conv2d(features, out_ch, kernel_size=1)

    def forward(self, image):

        #forward pass for Left side
        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)      
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        # x6 = self.maxpool(x5)
        # x7 = self.dwn_conv4(x6)
        
        #forward pass for Right side

        # x = self.trans1(x7)
        # y = crop_tensor(x, x5)
        # x = self.up_conv1(torch.cat([x,y], 1))
        
        x = self.trans2(x5)
        y = crop_tensor(x, x3)
        x = self.up_conv2(torch.cat([x,y], 1))
        
        x = self.trans3(x)
        y = crop_tensor(x, x1)
        x = self.up_conv3(torch.cat([x,y], 1))
        
        x = self.out(x)
        
        return x
        
def train_model(net,criterion,trainloader,optimizer,device):
    net.train()
    test_loss = 0
    for step, (batch_x, batch_y) in enumerate(trainloader):  # for each training step
        b_x = Variable(batch_x).to(device) # Inputs
        b_y = Variable(batch_y).to(device) # outputs
        prediction = net(b_x)
        loss = criterion(prediction, b_y)   # Calculating loss 
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients to update weights
        test_loss = test_loss + loss # Keep track of the loss for convenience 
    test_loss /= len(trainloader) # dividing by the number of batches
    print('the loss in this Epoch',test_loss.data)
    return test_loss

def test_model(net,criterion,trainloader,optimizer,device,text = 'validation'):
    net.eval() # Evaluation mode (important when having dropout layers)
    test_loss = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(trainloader):  # for each training step
            b_x = Variable(batch_x).to(device) # Inputs
            b_y = Variable(batch_y).to(device) # outputs
            prediction = net(b_x)
            loss = criterion(prediction, b_y)   # Calculating loss
            test_loss = test_loss + loss # Keep track of the loss 
        test_loss /= len(trainloader) # dividing by the number of batches
#         print(len(trainloader))
        print(text + ' loss:',test_loss.data)
    return test_loss