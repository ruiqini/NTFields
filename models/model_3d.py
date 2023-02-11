import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.autograd import Variable, grad
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from models import database as db

import matplotlib
import matplotlib.pylab as plt

import pickle5 as pickle 

from timeit import default_timer as timer

torch.backends.cudnn.benchmark = True

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class NN(torch.nn.Module):
    
    def __init__(self, device, dim):#10
        super(NN, self).__init__()
        self.dim = dim

        h_size = 256 #512,256
        fh_size = 128
        

        #3D CNN encoder
        self.conv_in = Conv3d(1, 16, 3, padding=1, padding_mode='zeros')  # out: 256 ->m.p. 128
        self.conv_0 = Conv3d(16, 32, 3, padding=1, padding_mode='zeros')  # out: 128
        
        self.actvn = torch.nn.ReLU()

        self.maxpool = torch.nn.MaxPool3d(2)

        self.conv_in_bn = torch.nn.BatchNorm3d(16)
        self.device = device

        feature_size = (1 +  16 ) * 7 #+ 3

        displacment = 0.0222#0.0222
        displacments = []
        
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)
        
        self.displacments = torch.Tensor(displacments).to(self.device)#cuda

        #decoder

        self.scale = 10

        self.act = torch.nn.ELU()#Softplus(beta=self.scale)#ELU,CELU

        self.nl1=5
        self.nl2=7

        self.encoder = torch.nn.ModuleList()
        self.encoder1 = torch.nn.ModuleList()
        #self.encoder.append(Linear(self.dim,h_size))
        
        self.encoder.append(Linear(dim,h_size))
        self.encoder1.append(Linear(dim,h_size))
        
        for i in range(self.nl1-1):
            self.encoder.append(Linear(h_size, h_size)) 
            self.encoder1.append(Linear(h_size, h_size)) 
        
        self.encoder.append(Linear(h_size, h_size)) 

        self.generator = torch.nn.ModuleList()
        self.generator1 = torch.nn.ModuleList()

        self.generator.append(Linear(2*h_size + 2*fh_size, 2*h_size)) 
        self.generator1.append(Linear(2*h_size + 2*fh_size, 2*h_size)) 

        for i in range(self.nl2-1):
            self.generator.append(Linear(2*h_size, 2*h_size)) 
            self.generator1.append(Linear(2*h_size, 2*h_size)) 
        
        self.generator.append(Linear(2*h_size,h_size))
        self.generator.append(Linear(h_size,1))

        self.fc_env0 = Linear(feature_size, fh_size)
        self.fc_env1 = Linear(fh_size, fh_size)
    
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            #stdv = np.sqrt(6 / 64.) / self.T
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)

    def env_encoder(self, x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        #net = self.maxpool(net)  # out 128
        return f_0, f_1#, f_2, f_3#, f_4, f_5, f_6
        #return f_0, f_1, f_2, f_3, f_4, f_5, f_6
    
    def env_features(self, coords, f_0, f_1):#, f_3, f_4, f_5, f_6):
        
        coords = coords.clone().detach().requires_grad_(False)

        p0=coords[:,:3]
        p1=coords[:,3:]

        size=p0.shape[0]

        p = torch.vstack((p0,p1))
        #print(p.shape)
        #p = p
        '''
        ptmp = p.clone()
        ptmp[:,0]=p[:,2]
        ptmp[:,2]=p[:,0]
        p=ptmp
        '''
        p = torch.index_select(p, 1, torch.LongTensor([2,1,0]).to(self.device))


        p=2*p
        
        p = p.unsqueeze(0)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        #print(p.shape)
        feature_0 = F.grid_sample(f_0, p, mode='bilinear', padding_mode='border')
        feature_1 = F.grid_sample(f_1, p, mode='bilinear', padding_mode='border')
        

        features = torch.cat((feature_0, feature_1), dim=1)  # (B, features, 1,7,sample_num)
        #features0 = torch.cat((feature_0,feature_1,feature_2, feature_3), dim=1)  # (B, features, 1,7,sample_num)
        
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #print(features.size())
        features = torch.squeeze(features.transpose(1, -1))

        features = self.act(self.fc_env0(features))
        features = self.act(self.fc_env1(features))

        features0=features[:size,:]
        features1=features[size:,:]
        
        return features0, features1

    def out(self, coords, features0, features1):
        
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))

        #x = x.unsqueeze(1)
        
        
        x  = self.act(self.encoder[0](x))
        for ii in range(1,self.nl1):
            #i0 = x
            x_tmp = x
            x  = self.act(self.encoder[ii](x))
            x  = self.act(self.encoder1[ii](x) + x_tmp) 
        
        x = self.encoder[-1](x)

        x0 = x[:size,...]
        x1 = x[size:,...]
        
        x_0 = torch.max(x0,x1)
        x_1 = torch.min(x0,x1)

        features_0 = torch.max(features0,features1)
        features_1 = torch.min(features0,features1)

        x = torch.cat((x_0, x_1, features_0, features_1),1)
        
        x = self.act(self.generator[0](x)) 

        for ii in range(1, self.nl2):
            x_tmp = x
            x = self.act(self.generator[ii](x)) 
            x = self.act(self.generator1[ii](x) + x_tmp) 
        
        y = self.generator[-2](x)
        x = self.act(y)

        y = self.generator[-1](x)
        x = torch.sigmoid(0.1*y) 
        
        return x, coords

    def forward(self, coords, grid):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        f_0, f_1 = self.env_encoder(grid)
        feature0, feature1= self.env_features(coords, f_0, f_1)
        output, coords = self.out(coords, feature0, feature1)
        
        return output, coords


class Model():
    def __init__(self, ModelPath, DataPath, dim, pos,device='cpu'):

        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.Params['DataPath'] = DataPath
        self.dim = dim
        self.pos = pos

        # Pass the JSON information
        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Normlisation'] = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 10000
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 20000
        self.Params['Training']['Resampling Bounds'] = [0.2, 0.95]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 10
        self.Params['Training']['Learning Rate'] = 2e-4#5e-5
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler (bool)'] = False

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x  

    def Loss(self, points, features0, features1, Yobs):
        
      
        start=time.time()
        tau, Xp = self.network.out(points, features0, features1)
        dtau = self.gradient(tau, Xp)
        end=time.time()
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)#torch.norm(D, p=2, dim =1)**2
        
        
        DT0=dtau[:,:self.dim]
        DT1=dtau[:,self.dim:]

        T01    = T0*torch.einsum('ij,ij->i', DT0, DT0)
        T02    = -2*tau[:,0]*torch.einsum('ij,ij->i', DT0, D)

        T11    = T0*torch.einsum('ij,ij->i', DT1, DT1)
        T12    = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, D)
        
    
        T3    = tau[:,0]**2
        
        S0 = (T01-T02+T3)
        S1 = (T11-T12+T3)
       
        #0.001
        sq_Ypred0 = 1/torch.sqrt(torch.sqrt(S0)/T3)
        sq_Ypred1 = 1/torch.sqrt(torch.sqrt(S1)/T3)


        sq_Yobs0=torch.sqrt(Yobs[:,0])
        sq_Yobs1=torch.sqrt(Yobs[:,1])

        
        diff = abs(1-sq_Ypred0/sq_Yobs0)+abs(1-sq_Ypred1/sq_Yobs1)+\
            abs(1-sq_Yobs0/sq_Ypred0)+abs(1-sq_Yobs1/sq_Ypred1)

        loss_n = torch.sum(diff)/Yobs.shape[0]

        loss = loss_n

        return loss, loss_n, diff

    def train(self):

        # Initialising the network
        #self._init_network()

        self.network = NN(self.Params['Device'],self.dim)
        self.network.apply(self.network.init_weights)
        #self.network.float()
        self.network.to(self.Params['Device'])
        # Defining the optimization scheme

        

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate']
            ,weight_decay=0.1)
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            
        self.dataset = db.Database(self.Params['DataPath'])

        len_dataset = len(self.dataset)
        n_batches = int(len(self.dataset) /
                        int(self.Params['Training']['Batch Size']) + 1)
        training_start_time = time.time()

        # --------- Splitting the dataset into training and validation -------
        indices = list(range(int(len_dataset)))
        #validation_idx = np.random.choice(indices, size=int(
        #    len_dataset*(self.Params['Training']['Validation Percentage']/100)), replace=False)
        #train_idx = list(set(indices) - set(validation_idx))
        train_idx = list(set(indices))
        #validation_sampler = SubsetRandomSampler(validation_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        '''
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size']),
            num_workers = 0,
            shuffle=True)
        '''
        #'''
        dataloader = FastTensorDataLoader(self.dataset.data, 
                    batch_size=int(self.Params['Training']['Batch Size']), 
                    shuffle=True)

        weights = Tensor(torch.ones(len(self.dataset))).to(
            torch.device(self.Params['Device']))
        
        dists=torch.norm(self.dataset.data[:,0:3]-self.dataset.data[:,3:6],dim=1)
        weights = dists.max()-dists

        weights = torch.clamp(
                weights/weights.max(), self.Params['Training']['Resampling Bounds'][0], self.Params['Training']['Resampling Bounds'][1])
        
        train_sampler_wei = WeightedRandomSampler(
                weights, len(weights), replacement=True)
            
        train_loader_wei = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size']),
            sampler=train_sampler_wei
        )

        speed = self.dataset.data[:,2*self.dim:]

        grid = self.dataset.grid
        grid = grid.to(self.Params['Device'])
        grid = grid.unsqueeze(0)
        print(speed.min())
        #'''
        
        weights = Tensor(torch.ones(len(self.dataset))).to(
                        torch.device(self.Params['Device']))
        PATH = self.Params['ModelPath']+'/check.pt'
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1.0
        #step = 1.0
        tt =time.time()

        current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
        #p=(torch.rand((5,6))-0.5).cuda()
        prev_state_queue = []
        prev_optimizer_queue = []

        self.l0 = 500

        self.l1 = 500

        for epoch in range(1, self.Params['Training']['Number of Epochs']+1):
            t_0=time.time()
            
            print_every = 1
            start_time = time.time()
            running_sample_count = 0
            total_train_loss = 0
            total_val_loss = 0
            total_diff=0

            
            self.lamb = min(1.0,max(0,(epoch-self.l0)/self.l1))
            

            #current_state = copy.deepcopy(self.network.state_dict())
            #if epoch%10 == 0:
            #    prev_state = current_state
            prev_state_queue.append(current_state)
            prev_optimizer_queue.append(current_optimizer)
            if(len(prev_state_queue)>5):
                prev_state_queue.pop(0)
                prev_optimizer_queue.pop(0)
            
            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
            
            
            
            self.optimizer.param_groups[0]['lr']  = max(5e-4*(1-epoch/self.l0),1e-5)
            
            prev_lr = self.optimizer.param_groups[0]['lr'] 
            t_1=time.time()
            #print(t_1-t_0)
            t_0=time.time()
            #print(prev)
            prev_diff = current_diff
            iter=0
            while True:
                total_train_loss = 0
                total_diff = 0
                #for i in range(10):
                for i, data in enumerate(train_loader_wei, 0):#train_loader_wei,dataloader
                    #print('----------------- Epoch {} - Batch {} --------------------'.format(epoch,i))
                    t0 = time.time()
                    
                    data=data[0].to(self.Params['Device'])
                    #data, indexbatch = data
                    points = data[:,:2*self.dim]#.float()#.cuda()
                    speed = data[:,2*self.dim:]#.float()#.cuda()

                    feature0=torch.zeros((points.shape[0],128)).to(self.Params['Device'])
                    feature1=torch.zeros((points.shape[0],128)).to(self.Params['Device'])
                    
                    if self.lamb > 0:
                        f_0, f_1 = self.network.env_encoder(grid)
                        feature0, feature1 = self.network.env_features(points, f_0, f_1)
                        feature0 = feature0*self.lamb
                        feature1 = feature1*self.lamb

                    loss_value, loss_n, wv = self.Loss(points, feature0, feature1, speed)
                    t1 = time.time()
                    #print(t1-t0)
                    
                    t0 = time.time()
                    loss_value.backward()

                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    
                    total_train_loss += loss_value.clone().detach()
                    total_diff += loss_n.clone().detach()
                    t1 = time.time()
                    
                    
                    del points, speed, loss_value, loss_n, wv#,indexbatch
                
               
                total_train_loss /= len(dataloader)#dataloader train_loader
                total_diff /= len(dataloader)#dataloader train_loader

                current_diff = total_diff
                diff_ratio = current_diff/prev_diff
            
                if (diff_ratio < 1.2 and diff_ratio > 0):#1.5
                    #self.optimizer.param_groups[0]['lr'] = prev_lr 
                    break
                else:
                    
                    iter+=1
                    with torch.no_grad():
                        random_number = random.randint(0, min(4,epoch-1))
                        self.network.load_state_dict(prev_state_queue[random_number], strict=True)
                        self.optimizer.load_state_dict(prev_optimizer_queue[random_number])
                    
                    print("RepeatEpoch = {} -- Loss = {:.4e}".format(
                        epoch, total_diff))
                
                
            #'''
            self.total_train_loss.append(total_train_loss)
            
            t_1=time.time()
            #print(t_1-t_0)

            #del train_loader_wei, train_sampler_wei

            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_train_loss)

            t_tmp = tt
            tt=time.time()
            
            if epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    print("Epoch = {} -- Loss = {:.4e}".format(
                        epoch, total_diff.item()))

            if (epoch % self.Params['Training']['Save Every * Epoch'] == 0) or (epoch == self.Params['Training']['Number of Epochs']) or (epoch == 1):
                self.plot(epoch,total_diff.item(),grid)
                with torch.no_grad():
                    self.save(epoch=epoch, val_loss=total_diff)

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.Params['ModelPath'], str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        #B = torch.load(self.Params['ModelPath']+'/B.pt')
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))

        self.network = NN(self.Params['Device'],self.dim)

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

        
    def load_pretrained_state_dict(self, state_dict):
        own_state=self.state_dict


    def TravelTimes(self, Xp, feature0, feature1):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        
        tau, coords = self.network.out(Xp, feature0, feature1)
       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.sqrt(T0)/tau[:, 0]

        del Xp, tau, T0
        return TT
    
    def Tau(self, Xp, feature0, feature1):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.network.out(Xp, feature0, feature1)
        
        return tau

    def Speed(self, Xp, feature0, feature1):
        Xp = Xp.to(torch.device(self.Params['Device']))

        tau, Xp = self.network.out(Xp, feature0, feature1)
        dtau = self.gradient(tau, Xp)        
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.einsum('ij,ij->i', D, D)

        DT1 = dtau[:,self.dim:]

        T1    = T0*torch.einsum('ij,ij->i', DT1, DT1)
        T2    = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        T3    = tau[:,0]**2
        
        S = (T1-T2+T3)

        Ypred = T3 / torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3
        return Ypred
    
    def Gradient(self, Xp, f_0, f_1):
        Xp = Xp.to(torch.device(self.Params['Device']))
       
        #Xp.requires_grad_()
        feature0, feature1 = self.network.env_features(Xp, f_0, f_1)
        
        tau, Xp = self.network.out(Xp, feature0, feature1)
        dtau = self.gradient(tau, Xp)
        
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D))
        T3 = tau[:, 0]**2

        V0 = D
        V1 = dtau[:,self.dim:]
        
        Y1 = 1/(T0*tau[:, 0])*V0
        Y2 = T0/T3*V1


        Ypred1 = -(Y1-Y2)
        Spred1 = torch.norm(Ypred1)
        Ypred1 = 1/Spred1**2*Ypred1

        V0=-D
        V1=dtau[:,:self.dim]
        
        Y1 = 1/(T0*tau[:, 0])*V0
        Y2 = T0/T3*V1

        Ypred0 = -(Y1-Y2)
        Spred0 = torch.norm(Ypred0)

        Ypred0 = 1/Spred0**2*Ypred0
        
        return torch.cat((Ypred0, Ypred1),dim=1)
     
    def plot(self,epoch,total_train_loss, grid):
        limit = 0.5
        xmin     = [-limit,-limit]
        xmax     = [limit,limit]
        spacing=limit/40.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))

        Xsrc = [0]*self.dim
        #print(self.dim)
        
        Xsrc[0] = self.pos[0]
        Xsrc[1] = self.pos[1]

        XP       = np.zeros((len(X.flatten()),2*self.dim))
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim+0]  = X.flatten()
        XP[:,self.dim+1]  = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.Params['Device'])

        feature0=torch.zeros((XP.shape[0],128)).to(self.Params['Device'])
        feature1=torch.zeros((XP.shape[0],128)).to(self.Params['Device'])
        
        if self.lamb > 0:
            f_0, f_1 = self.network.env_encoder(grid)
            feature0, feature1 = self.network.env_features(XP, f_0, f_1)
            feature0 = feature0*self.lamb
            feature1 = feature1*self.lamb
            
        
        tt = self.TravelTimes(XP,feature0, feature1)
        ss = self.Speed(XP,feature0, feature1)#*5
        tau = self.Tau(XP,feature0, feature1)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,3,0.05), cmap='bone', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.Params['ModelPath']+"/plots"+str(epoch)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')

        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,TAU,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,3,0.05), cmap='bone', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.Params['ModelPath']+"/tauplots"+str(epoch)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')

        plt.close(fig)