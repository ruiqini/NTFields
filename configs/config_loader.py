import configargparse
import numpy as np
import os


def config_parser():
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', is_config_file=True, default='configs/shapenet_cars.txt',
                        help='config file path')
    parser.add_argument("--exp_name", type=str, default=None,
                        help='Experiment name, used as folder name for the experiment. If left blank, a \
                         name will be auto generated based on the configuration settings.')
    parser.add_argument("--data_dir", type=str,
                        help='input data directory')
    parser.add_argument("--input_data_glob", type=str,
                        help='glob expression to find raw input files')
    # Training Data Parameters
    parser.add_argument("--num_points", type=int, default=20000,
                        help='Number of points sampled from each ground truth shape.')
    parser.add_argument("--num_samples", type=int, default=1e6,
                        help='Number of start-goal pairs sampled in space.')
    parser.add_argument("--num_dim", type=int, default=3,
                        help='Number of dimension for configuration space.')

    parser.add_argument("--bb_min", default=-0.5, type=float,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the min value in x,y and z for the bounding box.')
    parser.add_argument("--bb_max", default=0.5, type=float,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the max value in x,y and z for the bounding box.')
    parser.add_argument("--input_res", type=int, default=128,
                        help='Training and testing shapes are normalized to be in a common bounding box.\
                             This value defines the max value in x,y and z for the bounding box.')
    

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    
    return cfg
