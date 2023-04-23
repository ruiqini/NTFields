import sys
sys.path.append('.')
from models import model_3d as md
import torch
import os 
import numpy as np
import math
import matplotlib
import matplotlib.pylab as plt
import torch
from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler

from timeit import default_timer as timer
import math

import igl
import bvh_distance_queries

our_path=[]
our_time=[]
our_dis=[]
collision=0

modelPath = './Experiments/Gib'

dataPath = './datasets/gibson/'
womodel    = md.Model(modelPath, dataPath, 3,[0,0], device='cuda:0')


for gib_id in range(2):
    womodel.load('./Experiments/Gib/Model_Epoch_01100_'+str(gib_id)+'.pt')

    womodel.network.eval()
    v, f = igl.read_triangle_mesh("datasets/gibson/"+str(gib_id)+"/mesh_z_up_scaled.off")        
    print(gib_id)

    vertices=v*20
    faces=f

    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda:0')
    faces = torch.tensor(faces, dtype=torch.long, device='cuda:0')
    triangles = vertices[faces].unsqueeze(dim=0)
    
    path = "datasets/gibson/"+str(gib_id)+"/voxelized_point_cloud_128res_20000points.npz"

    occupancies = np.unpackbits(np.load(path)['compressed_occupancies'])
    input = np.reshape(occupancies, (128,)*3)
    grid = np.array(input, dtype=np.float32)
    print(np.shape(grid))
    grid = torch.from_numpy(grid).to('cuda:0').float()
    grid = grid.unsqueeze(0)
    

    f_0, f_1 = womodel.network.env_encoder(grid)
    

    sg = np.load("sample_sg_"+str(gib_id)+".npy")
    sg = Variable(Tensor(sg)).to('cuda')
   
    coll_sg =[]
    for id in range(500):
        

        start_goal = sg[id,:].unsqueeze(0)

        XP=start_goal
        
        XP=XP/20.0

        dis=torch.norm(XP[:,3:6]-XP[:,0:3])

        start = timer()

        point0=[]
        point1=[]

        point0.append(XP[:,0:3].clone())
        point1.append(XP[:,3:6].clone())
        #print(id)

        iter=0
        while dis>0.06:
            gradient = womodel.Gradient(XP.clone(), f_0, f_1)
            
            XP = XP + 0.03 * gradient
            dis=torch.norm(XP[:,3:6]-XP[:,0:3])
            
            point0.append(XP[:,0:3].clone())
            point1.append(XP[:,3:6].clone())
            iter=iter+1
            if(iter>500):
                break
        
        end = timer()

        
        point1.reverse()
        point=point0+point1
        
        xyz= torch.cat(point).to('cpu').data.numpy()#np.asarray(point)
        

        xyz=20*xyz
        STEP = 0.001
        all_p=[]
        for j in range(xyz.shape[0]-1):
            ll = np.linalg.norm(xyz[j+1] - xyz[j])
            all_p.append(xyz[j])
            all_p.append(xyz[j+1])
            for k in range(math.floor(ll/STEP)):
                cur_p=xyz[j]+k/math.floor(ll/STEP)*(xyz[j+1] - xyz[j])
                all_p.append(cur_p)
        query_points = np.vstack(all_p)
        #print(query_points)

        #unsigned_distance = igl.signed_distance(query_points, vertices, faces)[0]
        query_points = torch.tensor(query_points, dtype=torch.float32, device='cuda')
        query_points = query_points.unsqueeze(dim=0)
        #print(query_points.shape)
        bvh = bvh_distance_queries.BVH()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        distances, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points)
        torch.cuda.synchronize()
        #unsigned_distance = abs()
        #print(distances.shape)
        unsigned_distance = torch.sqrt(distances).squeeze()
        unsigned_distance = unsigned_distance.detach().cpu().numpy()

        if np.min(unsigned_distance)<=0.01 or iter>500 or np.max(xyz)>13 or np.min(xyz)<-13:
            collision+=1
            coll_sg.append(start_goal.to('cpu').data.numpy())
        else:
            our_path.append(xyz)
            our_time.append(end - start)
            our_dis.append(np.min(unsigned_distance))
        
    coll_sg=np.asarray(coll_sg)

print(collision)

def length(path):
    size=path.shape[0]
    l=0
    for i in range(size-1):
        l+=np.linalg.norm(path[i+1,:]-path[i,:])
    return l
#our_path=np.asarray(our_path)
our_time=np.asarray(our_time)

our_len=[]
for i in range(len(our_path)):
    our_len.append(length(our_path[i]))
our_len=np.asarray(our_len)
our_dis=np.asarray(our_dis)
print(np.max(our_len))
print("len p",len(our_path))
print("len t",len(our_time))
print(np.sum(our_time)/len(our_path))
print(np.sum(our_len)/len(our_path))

np.save("gib_len",our_len)
np.save("gib_time",our_time)
np.save("gib_dis",our_dis)

