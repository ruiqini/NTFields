import sys

sys.path.append('.')

from dataprocessing.convert_to_scaled_off import to_off
from dataprocessing.speed_sampling_gpu import sample_speed
import dataprocessing.voxelized_pointcloud_sampling as voxelized_pointcloud_sampling
from glob import glob
import configs.config_loader as cfg_loader
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os

cfg = cfg_loader.get_config()
print(cfg.data_dir)
print(cfg.input_data_glob)

print('Finding raw files for preprocessing.')
paths = glob( "./"+cfg.data_dir + cfg.input_data_glob)
print(paths)
paths = sorted(paths)



num_cpus = mp.cpu_count()


def multiprocess(func):
	p = Pool(num_cpus)
	p.map(func, paths)
	p.close()
	p.join()

print('Start scaling.')
multiprocess(to_off)

print('Start speed sampling.')
for path in paths:
	sample_speed(path, cfg.num_samples, cfg.num_dim)

print('Start voxelized pointcloud sampling.')
voxelized_pointcloud_sampling.init(cfg)
multiprocess(voxelized_pointcloud_sampling.voxelized_pointcloud_sampling)


