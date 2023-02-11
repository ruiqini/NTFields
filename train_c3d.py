from models import model_3d as md
from os import path


modelPath = './Experiments/C3D'         
dataPath = './datasets/c3d/0'

model    = md.Model(modelPath, dataPath, 3,[-0.4,-0], device='cuda:0')

model.train()


