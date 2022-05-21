from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def basic_block(feats, input_dim, output_dim, keep_probability, phase_train=True, 
            weight_decay=0.0, scope='', reuse=None, model_version=None):
  with tf.variable_scope(scope, scope, [feats], reuse=tf.AUTO_REUSE) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=phase_train):
      with slim.arg_scope([slim.fully_connected],
                        activation_fn=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        net = slim.fully_connected(feats, input_dim, activation_fn=tf.nn.relu, scope='fc1')
        net = slim.fully_connected(net, output_dim, activation_fn=None, scope='fc2')
  return net

def inference(feats, current_stage, stages, keep_probability, phase_train=True, 
            weight_decay=0.00004, reuse=None, model_version=None, res_layers=None, long_skip=None):

  end_points = {}
  stages_keys = sorted(list(stages.keys()))
  for stage in stages_keys:
    input_dim, output_dim = stages[stage]
    stage_scope = 'Stage_{}'.format(stage)
    if stage == 0:
      net = basic_block(feats, input_dim, output_dim, keep_probability, phase_train, 
            weight_decay, scope=stage_scope, reuse=reuse, model_version=model_version)
      net = tf.nn.l2_normalize(net, axis=1)
      end_points[stage] = net
    else:
      net = basic_block(net, input_dim, output_dim, keep_probability, phase_train, 
            weight_decay, scope=stage_scope, reuse=reuse, model_version=model_version)
      net = tf.nn.l2_normalize(net, axis=1)
      end_points[stage] = net      
  return end_points


