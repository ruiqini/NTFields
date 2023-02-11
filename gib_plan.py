import sys

sys.path.append('.')
from models import model_3d as md
import torch
import os 
import numpy as np
import math
import torch
from torch import Tensor
from torch.autograd import Variable, grad

from timeit import default_timer as timer
import math

import igl
import open3d as o3d

    #except:
    #    continue
our_path=[]
our_time=[]
our_dis=[]
collision=0

modelPath = './Experiments/Gib'#arona,bolton,cabin,A_test
#modelPath = './Experiments/Gib_res_changelr_scale'

dataPath = './datasets/gibson/'#Arona,Cabin,Bolton#filePath = './Experiments/Gibson'
womodel    = md.Model(modelPath, dataPath, 3,[0,0], device='cuda')



womodel.load('./Experiments/Gib/Model_Epoch_01100_0.pt')



path = "datasets/gibson/0/voxelized_point_cloud_128res_20000points.npz"

occupancies = np.unpackbits(np.load(path)['compressed_occupancies'])
input = np.reshape(occupancies, (128,)*3)
grid = np.array(input, dtype=np.float32)
print(np.shape(grid))
grid = torch.from_numpy(grid).to('cuda:0').float()
grid = grid.unsqueeze(0)


f_0, f_1 = womodel.network.env_encoder(grid)

for i in range(5):

    start_goal = np.array([[-6,-4,-6,6,7,-2.5]])
    XP=start_goal
    XP = Variable(Tensor(XP)).to('cuda')
    XP=XP/20.0

    #print(XP)
    dis=torch.norm(XP[:,3:6]-XP[:,0:3])
    start = timer()

    point0=[]
    point1=[]

    point0.append(XP[:,0:3])
    point1.append(XP[:,3:6])
    #print(id)

    iter=0
    while dis>0.06:
        gradient = womodel.Gradient(XP.clone(), f_0, f_1)

        XP = XP + 0.03 * gradient
        dis=torch.norm(XP[:,3:6]-XP[:,0:3])
        point0.append(XP[:,0:3])
        point1.append(XP[:,3:6])
        iter=iter+1
        if(iter>500):
            break

    end = timer()

    print("time",end-start)
point1.reverse()
point=point0+point1

xyz= torch.cat(point).to('cpu').data.numpy()

xyz=20*xyz


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

mesh = o3d.io.read_triangle_mesh("datasets/gibson/0/mesh_z_up_scaled.off")
        
mesh.scale(20, center=(0,0,0))

mesh.compute_vertex_normals()

pcd.paint_uniform_color([0, 0.706, 1])


o3d.visualization.draw_geometries([mesh,pcd])








