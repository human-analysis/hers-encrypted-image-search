"""Main training file for face
"""
import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from functools import partial
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import expit
from scipy.spatial import distance
import utils
from nntools.tensorflow.networks import DeepMDS
import matplotlib.pyplot as plt
import pickle
import random

GPU = '7'

def roc_threshold(imp_scores, gen_scores):
    # plot the histogram
    bins = np.linspace(-1, 1, 100)
    sweights = np.ones_like(gen_scores)/float(len(gen_scores))
    lweights = np.ones_like(imp_scores)/float(len(imp_scores))
    plt.hist(gen_scores, bins, alpha=.7, color='green', label='genuine', weights=sweights)
    plt.hist(imp_scores, bins, alpha=.7, color='red', label='imposter', weights=lweights)
    plt.xlabel('Similarity Score')
    plt.ylabel('Normalized Score Frequency')
    plt.legend(loc="upper right")
    plt.savefig('casia_val.png')
    print("Finished Saving Histogram")
    plt.clf()

    imp_labels = [0 for s in imp_scores]
    gen_labels =[1 for s in gen_scores]
    y = gen_labels + imp_labels
    y = np.array(y)
    scores = gen_scores + imp_scores
    scores = np.array(scores)
    fpr, tpr, thresholds = roc_curve(y, scores, 1)

    found = False
    for j, val in enumerate(fpr):
        if val > 0.01 and not found:
            #print("TDR = {}% @ FDR: {} Threshold = {}".format(tpr[j-1] * 100, fpr[j-1] * 100, thresholds[j-1]))
            return tpr[j-1] * 100


# load the testing features
train_feats = np.load('arc_face_casia/casia_train_feats.npy')
train_labels = np.load('arc_face_casia/casia_train_labels.npy')
val_feats = np.load('arc_face_casia/casia_val_feats.npy')
val_labels = np.load('arc_face_casia/casia_val_labels.npy')

unique_val, unique_indices = np.unique(val_labels, return_index=True)

last_c = 6573530
last_i = 49024

print(np.dot(val_feats[last_i], val_feats[last_i + 1]))


gen_val1 = val_feats[unique_indices]
gen_val2 = val_feats[unique_indices + 1]

ng = gen_val1.shape[0]
scores = np.dot(gen_val1, np.transpose(gen_val2))
gen_scores = np.diag(scores).tolist()

imp_scores = []
for r in range(ng):
    for c in range(ng):
        if c > r:
            imp_scores.append(scores[r, c])

# compute the original ROC
start_tar = roc_threshold(imp_scores, gen_scores)
print("TAR @ FAR 0.01 = {}".format(start_tar))
# finished computing original acc

num_features = train_labels.shape[0]
all_classes = np.unique(train_labels).tolist()
num_classes = len(all_classes)

subject_dict = {}
for class_num in all_classes:
    class_indices = np.where(train_labels == class_num)[0].tolist()
    subject_dict[class_num] = class_indices

def get_gen_pairs(num_pairs):
    pairs = []
    while len(pairs) < num_pairs:
        subj_index1 = np.random.randint(0, num_classes)
        subj_index1 = all_classes[subj_index1]
        subject_indices = subject_dict[subj_index1]
        num_features = len(subject_indices)
        if num_features < 2: continue # need at least 2 genuine features
        feat1_index = np.random.randint(0, num_features)
        feat2_index = np.random.randint(0, num_features)
        if feat1_index == feat2_index: continue

        index1 = subject_dict[subj_index1][feat1_index]
        index2 = subject_dict[subj_index1][feat2_index]
        gen_pair = (train_feats[index1, :], train_feats[index2, :])
        pairs.append(gen_pair)
    return pairs

def get_imp_pairs(num_pairs):
    pairs = []
    while len(pairs) < num_pairs:
        index1 = np.random.randint(0, num_features)
        index2 = np.random.randint(0, num_features)
        if train_labels[index1] != train_labels[index2]:
            imp_pair = (train_feats[index1, :], train_feats[index2, :])
            pairs.append(imp_pair)
    return pairs

def get_distances(gen_pairs, imp_pairs):
    g1 = []
    g2 = []
    i1 = []
    i2 = []
    for pair in gen_pairs:
        g1.append(pair[0])
        g2.append(pair[1])
    for pair in imp_pairs:
        i1.append(pair[0])
        i2.append(pair[1])
    g1 = np.array(g1)
    g2 = np.array(g2)
    i1 = np.array(i1)
    i2 = np.array(i2)

    #gen_distances = np.diag(euclidean_distances(g1, g2))
    #imp_distances = np.diag(euclidean_distances(i1, i2))
    gen_distances = np.diag(np.matmul(g1, np.transpose(g2)))
    imp_distances = np.diag(np.matmul(i1, np.transpose(i2)))
    
    return g1, g2, i1, i2, gen_distances, imp_distances

def augment(x, config):
    #random_mask = np.random.choice([0, 1], size=(x.shape[1]), p=[1./2, 1./2])
    #random_mask = np.tile(random_mask, (x.shape[0], 1))
    random_mask = np.random.choice([0, 1], size=(x.shape[0], x.shape[1]), p=[config.obfuscate_percentage, 1.0 - config.obfuscate_percentage])
    augmented = np.multiply(random_mask, x)
    return augmented


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU
    config_file = args.config_file
    # I/O
    config = utils.import_file(config_file, 'config')

    network = DeepMDS()
    network.initialize(config)
    if config.restore_model:
        network.restore_model(config.restore_model, config.restore_scopes)

    # Initalization for running
    log_dir = utils.create_log_dir(config, config_file)
    summary_writer = tf.summary.FileWriter(log_dir, network.graph)

    #
    # Main Loop
    #
    print('\nStart Training\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n'\
        % (config.num_epochs, config.epoch_size, config.batch_size))
    global_step = 0
    start_time = time.time()

    best_tar = 0.0

    # start the training
    for epoch in range(config.num_epochs):
        for step in range(config.epoch_size):
            # Prepare input
            learning_rate = utils.get_updated_learning_rate(global_step, config)

            # get the genuine pairs
            if config.hard_mining:
                gen_pairs = get_gen_pairs(config.candidate_pairs)
            else:
                gen_pairs = get_gen_pairs(config.num_pairs)

            # get the imposter pairs
            if config.hard_mining:
                imp_pairs = get_imp_pairs(config.candidate_pairs)
            else:
                imp_pairs = get_imp_pairs(config.num_pairs)

            # get distances
            g1, g2, i1, i2, gen_distances, imp_distances = get_distances(gen_pairs, imp_pairs)

            if config.hard_mining:
                num_2_mine = 50
                if epoch > 50:
                    num_2_mine = 100
                if epoch > 100:
                    num_2_mine = 150
                if epoch > 150:
                    num_2_mine = 250
                num_random = 250 - num_2_mine
                 
                g1_proj = network.extract_features(g1, embedding_dim=config.stages[config.current_stage][1], block_size=1000, verbose=False)
                g2_proj = network.extract_features(g2, embedding_dim=config.stages[config.current_stage][1], block_size=1000, verbose=False)
                i1_proj = network.extract_features(i1, embedding_dim=config.stages[config.current_stage][1], block_size=1000, verbose=False)
                i2_proj = network.extract_features(i2, embedding_dim=config.stages[config.current_stage][1], block_size=1000, verbose=False)
                gen_distances_proj = np.diag(np.matmul(g1_proj, np.transpose(g2_proj)))
                imp_distances_proj = np.diag(np.matmul(i1_proj, np.transpose(i2_proj)))
     
                gen_diffs = np.abs(gen_distances - gen_distances_proj)
                imp_diffs = np.abs(imp_distances - imp_distances_proj)
 
                # HARD IMPOSTERS, HARD GENS
                #highest_gens = gen_diffs.argsort()[-config.num_pairs:]
                highest_gens = gen_diffs.argsort()[-num_2_mine:]
                highest_imps = imp_diffs.argsort()[-num_2_mine:]
 
                # reselect g1, g2, i1, i2 based on the highest pairwise distances following projection
                g1_hard = g1[highest_gens]
                g2_hard = g2[highest_gens]
                i1_hard = i1[highest_imps]
                i2_hard = i2[highest_imps]
                gen_distances_hard = gen_distances[highest_gens]
                imp_distances_hard = imp_distances[highest_imps]
 
                # get possible random indices
                random_choices_gens = np.array(list(set(list(range(1000))) - set(highest_gens.tolist())))
                random_choices_imps = np.array(list(set(list(range(1000))) - set(highest_imps.tolist())))
                g1_random = g1[random_choices_gens[0:num_random]]
                g2_random = g2[random_choices_gens[0:num_random]]
                i1_random = i1[random_choices_imps[0:num_random]]
                i2_random = i2[random_choices_imps[0:num_random]]
                gen_distances_random = gen_distances[random_choices_gens[0:num_random]]
                imp_distances_random = imp_distances[random_choices_imps[0:num_random]]
 
                # cncat the two together
                g1 = np.concatenate([g1_hard, g1_random], axis=0)
                g2 = np.concatenate([g2_hard, g2_random], axis=0)
                i1 = np.concatenate([i1_hard, i1_random], axis=0)
                i2 = np.concatenate([i2_hard, i2_random], axis=0)
                gen_distances = np.concatenate([gen_distances_hard, gen_distances_random], axis=0)
                imp_distances = np.concatenate([imp_distances_hard, imp_distances_random], axis=0)

            if config.augment:
                batch = np.concatenate([g1, g2, i1, i2], axis=0)
                batch = augment(batch, config)
                g1 = batch[0*config.num_pairs:1*config.num_pairs, :]
                g2 = batch[1*config.num_pairs:2*config.num_pairs, :]
                i1 = batch[2*config.num_pairs:3*config.num_pairs, :]
                i2 = batch[3*config.num_pairs:4*config.num_pairs, :]

            all_distances = np.concatenate([gen_distances, imp_distances])
            all_feats = np.concatenate([g1, g2, i1, i2], axis=0)

            wl, sm, global_step = network.train(all_feats, all_distances, learning_rate, config.keep_prob)

            wl['lr'] = learning_rate

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration, wl)
                summary_writer.add_summary(sm, global_step=global_step)
                #print("Orig TAR: {} Curr TAR: {}".format(start_tar, best_tar))
                #network.get_weights()

        # compute the accuracy for the epoch
        """
        gen_val1_proj = network.extract_features(gen_val1, embedding_dim = 16, block_size=1000, verbose=False)
        gen_val2_proj = network.extract_features(gen_val2, embedding_dim = 16, block_size=1000, verbose=False)
        ng = gen_val1.shape[0]
        scores = np.dot(gen_val1_proj, np.transpose(gen_val2_proj))
        gen_scores = np.diag(scores).tolist()
        imp_scores = []
        for r in range(ng):
            for c in range(ng):
                if c > r:
                    imp_scores.append(scores[r, c])
        curr_tar = roc_threshold(imp_scores, gen_scores)
        """
        #if curr_tar > best_tar:
        # save out best model
        network.save_model(log_dir, global_step) 

        # Output test result
        summary = tf.Summary()
        #summary.value.add(tag='sd4/accuracy', simple_value=tpr)
        summary_writer.add_summary(summary, global_step)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    args = parser.parse_args()
    main(args)
