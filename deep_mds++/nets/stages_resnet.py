from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def basic_block(feats, input_dim, output_dim, keep_probability, phase_train=True, 
            weight_decay=0.0, scope='', reuse=None, model_version=None, res_layers=1, long_skip=False):
  with tf.variable_scope(scope, scope, [feats], reuse=tf.AUTO_REUSE) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=phase_train):
      with slim.arg_scope([slim.fully_connected],
                        activation_fn=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        compression_amount = input_dim - output_dim
        compression_chunk = compression_amount // res_layers
        remainder = 0
        if compression_chunk * res_layers < compression_amount:
            remainder = compression_amount - (compression_chunk * res_layers)
        layer_input_dim = input_dim
        residual = feats
        net = feats
        long_skip_residual = slim.fully_connected(net, output_dim, activation_fn=None, scope='fc0')
        print(input_dim, output_dim, net.shape)
        for i in range(res_layers):
            net = slim.fully_connected(net, layer_input_dim, activation_fn=tf.nn.relu, scope='fc1_{}'.format(i))
            print("{} fc1_{}; shape = {}".format(scope.name, i, net.shape))
            net = net + residual
            print("{} short-skip_{}; shape = {}".format(scope.name, i, net.shape))
            # compute current output dim
            layer_output_dim = input_dim - ((i + 1) * compression_chunk)
            if i == res_layers - 1 and remainder != 0:
                # we are on the last layer and still have to compress
                layer_output_dim -= remainder
            net = slim.fully_connected(net, layer_output_dim, activation_fn=None, scope='fc2_{}'.format(i))
            print("{} fc2_{}; shape = {}".format(scope.name, i, net.shape))
            # update input layer dim and residual
            layer_input_dim = layer_output_dim
            residual = net
        if long_skip:
            net = net + long_skip_residual
            print("{} long-skip; shape = {}".format(scope.name, net.shape))
  return net

def inference(feats, current_stage, stages, keep_probability, phase_train=True, 
            weight_decay=0.00004, reuse=None, model_version=None, res_layers=1, long_skip=False):

  end_points = {}
  stages_keys = sorted(list(stages.keys()))
  print("Network Arch.")
  for stage in stages_keys:
    input_dim, output_dim = stages[stage]
    stage_scope = 'Stage_{}'.format(stage)
    print("*{}".format(stage_scope))
    if stage == 0:
      net = basic_block(feats, input_dim, output_dim, keep_probability, phase_train, 
            weight_decay, scope=stage_scope, reuse=reuse, model_version=model_version, res_layers=res_layers, long_skip=long_skip)
      net = tf.nn.l2_normalize(net, axis=1)
      end_points[stage] = net
    else:
      net = basic_block(net, input_dim, output_dim, keep_probability, phase_train, 
            weight_decay, scope=stage_scope, reuse=reuse, model_version=model_version, res_layers=res_layers, long_skip=long_skip)
      net = tf.nn.l2_normalize(net, axis=1)
      end_points[stage] = net      
  return end_points


