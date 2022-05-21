import sys
import os
import time
import imp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import math

from .. import utils as tfutils 
from .. import watcher as tfwatcher
from scipy.special import expit

class DeepMDS:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config):
        '''
            Initialize the graph from scratch according config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                self.config = config
                batch_size = self.config.num_pairs * 4
                batch_placeholder = tf.placeholder(tf.float32, shape=[batch_size, config.input_size], name='batch')
                distances_placeholder = tf.placeholder(tf.float32, shape=[2*self.config.num_pairs], name='distances')
                learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
                keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
                phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
                global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                splits = tf.split(batch_placeholder, config.num_gpus)
                distances_splits = tf.split(distances_placeholder, config.num_gpus)
                grads_splits = []
                split_dict = {}
                def insert_dict(k,v):
                    if k in split_dict: split_dict[k].append(v)
                    else: split_dict[k] = [v]
                        
                for i in range(config.num_gpus):
                    scope_name = '' if i==0 else 'gpu_%d' % i
                    with tf.name_scope(scope_name):
                        with tf.variable_scope('', reuse=i>0):
                            with tf.device('/gpu:%d' % i):
                                s1 = tf.identity(splits[i], name='inputs_split')
                                distances = tf.identity(distances_splits[i], name='distances_split')
                                # Save the first channel for testing
                                if i == 0:
                                    self.inputs = tf.identity(s1, name='inputs')

                                network = imp.load_source('network', config.network)

                                endpoints = network.inference(s1, config.current_stage, config.stages, keep_prob_placeholder, phase_train_placeholder, \
                                                                  res_layers=config.res_layers, long_skip=config.long_skip)
                                proj = endpoints[config.current_stage]

                                if i == 0:
                                    self.outputs = tf.identity(proj, name='outputs')

                                
                                # separate out the feats and distances from the batch
                                g1_proj = proj[0*self.config.num_pairs:1*self.config.num_pairs]
                                g2_proj = proj[1*self.config.num_pairs:2*self.config.num_pairs]
                                i1_proj = proj[2*self.config.num_pairs:3*self.config.num_pairs]
                                i2_proj = proj[3*self.config.num_pairs:4*self.config.num_pairs]
                                gen_distances = distances[:self.config.num_pairs]
                                imp_distances = distances[self.config.num_pairs:]

                                # distances
                                gen_proj_distances = tf.diag_part(tf.matmul(g1_proj, tf.transpose(g2_proj)))
                                imp_proj_distances = tf.diag_part(tf.matmul(i1_proj, tf.transpose(i2_proj)))

                                # Build all losses
                                loss_list = []
                                if 'distances_loss' in config.losses.keys():
                                    # distance is a misnomer, these are really cosine similarities
                                    gloss = tf.reduce_mean(tf.abs(gen_distances - gen_proj_distances))
                                    iloss = tf.reduce_mean(tf.abs(imp_distances - imp_proj_distances))
                                    dist_loss = config.losses['distances_loss']['gweight'] * gloss + config.losses['distances_loss']['iweight'] * iloss
                                    loss_list.append(dist_loss)
                                    insert_dict('dist_loss', dist_loss)

                                if 'cov_loss' in config.losses.keys():
                                    x = tf.concat([g1_proj, g2_proj, i1_proj, i2_proj], axis=0)
                                    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
                                    mx = tf.matmul(tf.transpose(mean_x), mean_x)
                                    vx = tf.matmul(tf.transpose(x), x) / 1000.0
                                    if config.batch_size != 1000: print("ERROR deepmds.py"); sys.exit(0)
                                    cov_xx = vx - mx
                                    diagonal = tf.diag_part(cov_xx)
                                    diag_matrix = tf.diag(diagonal)
                                    off_diagonals = cov_xx - diag_matrix
                                    cov_loss = config.losses['cov_loss']['weight'] * tf.reduce_mean(tf.abs(off_diagonals))
                                    loss_list.append(cov_loss)
                                    insert_dict('cov_loss', cov_loss)

                                if 'supervise_loss' in config.losses.keys():
                                    # distance is a misnomer, these are really cosine similarities
                                    gloss = (tf.reduce_mean(gen_proj_distances) + 1.0) * 0.5 # range needs to be 0->1
                                    iloss = (tf.reduce_mean(imp_proj_distances) + 1.0) * 0.5 # range needs to be 0->1
                                    lda_term = iloss / gloss
                                    loss_list.append(lda_term)
                                    insert_dict('sup_loss', lda_term)


                                # Collect all losses
                                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                                loss_list.append(reg_loss)
                                insert_dict('reg_loss', reg_loss)

                                total_loss = tf.add_n(loss_list, name='total_loss')
                                grads_split = tf.gradients(total_loss, tf.trainable_variables())
                                grads_splits.append(grads_split)


                # Merge the splits
                grads = tfutils.average_grads(grads_splits)
                for k,v in split_dict.items():
                    v = tfutils.average_tensors(v)
                    tfwatcher.insert(k, v)
                    if 'loss' in k:
                        tf.summary.scalar('losses/' + k, v)
                    else:
                        tf.summary.scalar(k, v)


                # Training Operaters
                apply_gradient_op = tfutils.apply_gradient(tf.trainable_variables(), grads, config.optimizer,
                                        learning_rate_placeholder, config.learning_rate_multipliers)

                # reset my optimizer
                adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]

                update_global_step_op = tf.assign_add(global_step, 1)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                train_ops = [apply_gradient_op, update_global_step_op] + update_ops
                train_op = tf.group(*train_ops)

                tf.summary.scalar('learning_rate', learning_rate_placeholder)
                summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(adam_initializers) # reset the adam optimizer
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)

                # Keep useful tensors
                self.batch_placeholder = batch_placeholder
                self.distances_placeholder = distances_placeholder 
                self.learning_rate_placeholder = learning_rate_placeholder 
                self.keep_prob_placeholder = keep_prob_placeholder 
                self.phase_train_placeholder = phase_train_placeholder 
                self.global_step = global_step
                self.train_op = train_op
                self.summary_op = summary_op
                
    def train(self, feats, distances, learning_rate, keep_prob):
        
        feed_dict = {   
                        self.batch_placeholder: feats,
                        self.distances_placeholder: distances,
                        self.learning_rate_placeholder: learning_rate,
                        self.keep_prob_placeholder: keep_prob,
                        self.phase_train_placeholder: True,
                    }
        
        # alwsays train this network
        _, wl, sm = self.sess.run([self.train_op, tfwatcher.get_watchlist(), self.summary_op], feed_dict = feed_dict)

        # update the global step
        step = self.sess.run(self.global_step)

        return wl, sm, step
    
    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)
        
    def load_model(self, *args, **kwargs):
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('keep_prob:0')
        self.batch_placeholder = self.graph.get_tensor_by_name('batch:0')
        self.outputs = self.graph.get_tensor_by_name('outputs:0')

    def extract_features(self, feats, embedding_dim = 16, block_size=1000, verbose=False):
        final_feats = np.zeros((feats.shape[0], embedding_dim), dtype=np.float32)
        num_blocks = feats.shape[0] // block_size
        if feats.shape[0] % block_size != 0:
            num_blocks += 1
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i+1)*block_size, feats.shape[0])
            chunk = feats[start_idx:end_idx, :]
            chunk_size = chunk.shape[0]
            if chunk_size != 1000:
                num_randoms = 1000 - chunk_size
                random_junk = np.zeros((num_randoms, feats.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, random_junk], axis=0)
            feed_dict = {
                            self.batch_placeholder: chunk,
                            self.phase_train_placeholder: False,
                            self.keep_prob_placeholder: 1.0
                        }
            result = self.sess.run(self.outputs, feed_dict=feed_dict)
            if chunk_size != 1000:
                result = result[0:chunk_size, :]
            final_feats[start_idx:end_idx, :] = result
        return final_feats

    def get_weights(self, scope='Stage_1/fc1/weights:0'):
        weight_tensor = self.graph.get_tensor_by_name(scope)
        weights = self.sess.run(weight_tensor)
        print(weights)
 
