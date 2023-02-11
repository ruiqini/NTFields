from models import model_3d as md
from os import path


modelPath = './Experiments/Gib'         
dataPath = './datasets/gibson/0'

model    = md.Model(modelPath, dataPath, 3,[-0.25,-0.2], device='cuda:0')

model.train()


