import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import random
import sys
import traceback
import logging
import igl
import numpy as np
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def to_off(path):

    file_path = os.path.dirname(path)
    data_type = file_path.split('/')[2]
    #print(data_type)
    file_name = os.path.splitext(os.path.basename(path))[0]
    output_file = os.path.join(file_path,file_name + '_scaled.off')

    if os.path.exists(output_file):
        print('Exists: {}'.format(output_file))
        return

    try:
        
        v, f = igl.read_triangle_mesh(path)

        bb_max = v.max(axis=0, keepdims=True)
        bb_min = v.min(axis=0, keepdims=True)
        #print(centers)
        #print(bb_max-bb_min)
        if data_type == 'c3d':
            v/=40
        elif data_type != 'arm':
            centers = (bb_max+bb_min)/2.0
            v = v-centers
            v = v/(bb_max-bb_min)
            
        igl.write_triangle_mesh(output_file, v, f) 

        print('Finished: {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

