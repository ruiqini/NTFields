import sys
sys.path.append('.')
import torch
import pytorch_kinematics as pk
import numpy as np

import matplotlib.pylab as plt

import open3d as o3d
import math
import torch
from torch import Tensor
from torch.autograd import Variable, grad

from models import model_arm as md
from timeit import default_timer as timer

file_path = 'datasets/arm/'
path_name_ = '4DOF'

out_path_ = file_path+path_name_
end_effect_='Link_3'

modelPath = './Experiments/4DOF'         
dataPath = './datasets/arm/4DOF'
model = md.Model(modelPath,dataPath,4,[-0.0,-0.0],device='cuda')
model.load('./Experiments/4DOF/Model_Epoch_01100.pt')

path = 'datasets/arm/'+path_name_+'/voxelized_point_cloud_128res_20000points.npz'

occupancies = np.unpackbits(np.load(path)['compressed_occupancies'])
input = np.reshape(occupancies, (128,)*3)
x = np.array(input, dtype=np.float32)
print(np.shape(x))
x = torch.from_numpy(x).cuda().float()
x = x.unsqueeze(0)
f_0, f_1 = model.network.env_encoder(x)

for ii in range(5):
    start_goal = [[0.2,-1.5,-0.6,-1.5,
                    -1,1.5,0.8,-1.8]]

    start_goal = [[-0.2*math.pi,0.1,0.5,1.5,
                    0.3*math.pi,0.1,0.5,1.5]]

    #start_goal = [[1.8,-0.2,0.2,2.2,
    #                -2.6,0.2,0.8,1.8]]

    scale = 1.8 * math.pi

    XP=start_goal
    XP = Variable(Tensor(XP)).to('cuda')/scale
    dis=torch.norm(XP[:,4:8]-XP[:,0:4])

    

    start = timer()

    point0=[]
    point1=[]
    point0.append(XP[:,:4].clone())
    point1.append(XP[:,4:].clone())
    while dis>0.04:
        gradient = model.Gradient(XP.clone(), f_0, f_1)
        
        XP = XP + 0.02 * gradient
        dis=torch.norm(XP[:,4:8]-XP[:,0:4])
        point0.append(XP[:,:4].clone())
        point1.append(XP[:,4:].clone())
    end = timer()
    print("time",end-start)

point1.reverse()
point=point0+point1

xyz=torch.cat(point)*scale

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

chain = pk.build_serial_chain_from_urdf(
    open(out_path_+'/'+path_name_+".urdf").read(), end_effect_)
chain = chain.to(dtype=dtype, device=d)

N = 1
th_batch = torch.tensor([[-0.2*math.pi,0.1,0.5,1.5,0,1],
                [0.3*math.pi,0.1,0.5,1.5,1,0]]).cuda()
tg_batch = chain.forward_kinematics(xyz, end_only = False)

p_list=[]
iter = 0
pointsize = 0
for tg in tg_batch:
    print(iter,tg)
    if iter>0:
        v = np.load(out_path_+'/meshes/collision/'+tg+'.npy')
        nv = np.ones((v.shape[0],4))
        pointsize = pointsize+v.shape[0]

        nv[:,:3]=v[:,:3]
        m = tg_batch[tg].get_matrix()
        #print(m.shape)
        t=torch.from_numpy(nv).float().cuda()
        p=torch.matmul(m[:],t.T)
        #p=p.cpu().numpy()
        p = torch.permute(p, (0, 2, 1)).contiguous()
        #p=np.transpose(p,(0,2,1))
        p_list.append(p)
        del m,p,t,nv, v
    iter = iter+1

p = torch.cat(p_list, dim=1)
p = torch.reshape(p,(p.shape[0]*p.shape[1],p.shape[2])).contiguous()
query_points = p[:,0:3].contiguous()

xyz = 0.4*query_points.detach().cpu().numpy()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz[:,0:3])

mesh_name = 'model_scaled.off'

path = file_path + path_name_+'/'

obstacle = o3d.io.read_triangle_mesh(path + mesh_name)
    
obstacle.compute_vertex_normals()

o3d.visualization.draw_geometries([obstacle,pcd])

