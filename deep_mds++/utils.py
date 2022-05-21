"""Utilities for training and testing
"""

import sys
import os
import numpy as np
from scipy import misc
import imp
import time
import math
import random
from datetime import datetime
import shutil
from nntools.common.dataset import Dataset
from nntools.common.imageprocessing import *

def import_file(full_path_to_module, name='module.name'):
    
    module_obj = imp.load_source(name, full_path_to_module)
    
    return module_obj

def create_log_dir(config, config_file):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.log_base_dir), config.name, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    shutil.copyfile(config_file, os.path.join(log_dir,'config.py'))

    return log_dir

def get_updated_learning_rate(global_step, config):
    if config.learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in config.learning_rate_schedule.items():
            if global_step >= step and step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % global_step)
    elif config.learning_rate_strategy == 'cosine':
        initial = config.learning_rate_schedule['initial']
        interval = config.learning_rate_schedule['interval']
        end_step = config.learning_rate_schedule['end_step']
        step = math.floor(float(global_step) / interval) * interval
        assert step <= end_step
        learning_rate = initial * 0.5 * (math.cos(math.pi * step / end_step) + 1)
    return learning_rate

def display_info(epoch, step, duration, watch_list):
    sys.stdout.write('[%d][%d] time: %2.2f' % (epoch+1, step+1, duration))
    for item in watch_list.items():
        if type(item[1]) in [float, np.float32, np.float64]:
            sys.stdout.write('   %s: %2.3f' % (item[0], item[1]))
        elif type(item[1]) in [int, bool, np.int32, np.int64, np.bool]:
            sys.stdout.write('   %s: %d' % (item[0], item[1]))
    sys.stdout.write('\n')

def get_pairwise_score_label(score_mat, label):
    n = label.size
    assert score_mat.shape[0]==score_mat.shape[1]==n
    triu_indices = np.triu_indices(n, 1)
    if len(label.shape)==1:
        label = label[:, None]
    label_mat = label==label.T
    score_vec = score_mat[triu_indices]
    label_vec = label_mat[triu_indices]
    return score_vec, label_vec

def fuse_features(mu1, sigma_sq1, mu2, sigma_sq2):
    sigma_new = (sigma_sq1 * sigma_sq2) / (sigma_sq1 + sigma_sq2)
    mu_new = (sigma_sq2 * mu1 + sigma_sq1 * mu2) / (sigma_sq1 + sigma_sq2)
    return mu_new, sigma_new

def match_features(mu1, sigma_sq1, mu2, sigma_sq2):
    t1 = list(zip(mu1, sigma_sq1))
    t2 = list(zip(mu2, sigma_sq2))
    def metric(t1, t2):
        mu1, sigma_sq1 = tuple(t1)
        mu2, sigma_sq2 = tuple(t2)
        sigma_sq_sum = sigma_sq1 + sigma_sq2
        score = - np.sum(np.square(mu1 - mu2) / sigma_sq_sum) - np.sum(np.log(sigma_sq_sum))
        return score
    return facepy.protocol.compare_sets(t1, t2, metric)
