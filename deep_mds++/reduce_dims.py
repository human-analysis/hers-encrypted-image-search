import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
import pickle
from numpy import linalg as LA
import argparse
from nntools.tensorflow.networks import DeepMDS
import utils
from scipy.special import expit
from scipy.spatial import distance

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    # Get the configuration file
    config = utils.import_file(os.path.join(args.model_dir, 'config.py'), 'config')
    
    # Load model files and config file
    network = DeepMDS()
    network.load_model(args.model_dir)

    # load the testing features
    feats = np.load(args.feats)
    print("Loaded features with shape = {}".format(feats.shape))

    mags = 1.0 / np.linalg.norm(feats, axis=1)
    feats = np.transpose(np.multiply(np.transpose(feats), mags))

    start = time.time()

    block_size = 250
    normed_feats = network.extract_features(feats, embedding_dim = config.stages[config.current_stage][1], block_size = 250, verbose=True)
    end = time.time()
    print(end - start)

    # Output the extracted features
    print("Extracted feats with dim: {}".format(normed_feats.shape))
    np.save('out_feats.npy', normed_feats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str, required=True)
    parser.add_argument("--feats", help="The path to the feats",
                        type=str, required=True)
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=256)
    args = parser.parse_args()
    main(args)
