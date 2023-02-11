import sys
sys.path.append('.')
from models import model_arm as md
from os import path


modelPath = './Experiments/4DOF'         
dataPath = './datasets/arm/4DOF'

model    = md.Model(modelPath, dataPath, 4,[-0.0,-0.0], device='cuda:0')

model.train()


