import sys
sys.path.append('.')
import numpy as np
#import open3d as o3d
import igl

from timeit import default_timer as timer


for gib_id in range(2):

    v, f = igl.read_triangle_mesh("datasets/gibson/"+str(gib_id)+"/mesh_z_up_scaled.off")
    v=v*20

    Xmin=[-9.9,-9.9,-9.9]
    Xmax=[9.9,9.9,9.9]
    Xmin = np.append(Xmin,Xmin)
    Xmax = np.append(Xmax,Xmax)
    dim=3
    numsamples = 2500*(gib_id+1)
    X  = np.zeros((numsamples,2*dim))
    PointsOutside = np.arange(numsamples)
    while len(PointsOutside) > 0:
        P  = np.random.rand(len(PointsOutside),dim)*(Xmax[:dim]-Xmin[:dim])[None,None,:] + Xmin[:dim][None,None,:]
        dP = np.random.rand(len(PointsOutside),dim)-0.5
        rL = ((np.random.rand(len(PointsOutside),1))+0.10)*np.sqrt(np.sum((Xmax-Xmin)**2))
        nP = P + (dP/np.sqrt(np.sum(dP**2,axis=1))[:,np.newaxis])*rL

        X[PointsOutside,:dim] = P
        X[PointsOutside,dim:] = nP

        maxs          = np.any((X[:,dim:] > Xmax[:dim][None,:]),axis=1)
        mins          = np.any((X[:,dim:] < Xmin[:dim][None,:]),axis=1)
        OutOfDomain   = np.any(np.concatenate((maxs[:,None],mins[:,None]),axis=1),axis=1)
        PointsOutside = np.where(OutOfDomain)[0]
    print(X)
    vec=X[:,0:3]-X[:,3:6]

    p0=X[:,0:3]

    d0 = igl.signed_distance(p0, v, f)[0]

    p1=X[:,3:6]

    d1 = igl.signed_distance(p1, v, f)[0]

    mask = np.logical_and(d0>2.0 , d1>2.0)
    newX=X[mask,:]

    w0 = igl.winding_number(v, f, newX[:,0:3])
    w1 = igl.winding_number(v, f, newX[:,3:6])

    mask1 = np.logical_and(w0<0.0 , w1<0.0)
    finalX = newX[mask1,:]

    print(len(finalX))
    finalX=finalX[0:500,:]
    np.save("sample_sg_"+str(gib_id),finalX)
#'''
'''
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(finalX[:,0:3])

pcd2.paint_uniform_color([0, 0.706, 1])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(finalX[:,3:6])

pcd.paint_uniform_color([1, 0.706, 0])

mesh = o3d.io.read_triangle_mesh("datasets/gibson/0/mesh_z_up_scaled.off")
mesh.scale(20, center=(0,0,0))
o3d.visualization.draw_geometries([mesh,pcd, pcd2],mesh_show_wireframe=True)
'''