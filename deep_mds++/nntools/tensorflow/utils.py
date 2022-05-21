import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def average_tensors(tensors, name=None):
    if len(tensors) == 1:
        return tf.identity(tensors[0], name=name)
    else:
        # Each tensor in the list should be of the same size
        expanded_tensors = []

        for t in tensors:
            expanded_t = tf.expand_dims(t, 0)
            expanded_tensors.append(expanded_t)

        average_tensor = tf.concat(axis=0, values=expanded_tensors)
        average_tensor = tf.reduce_mean(average_tensor, 0, name=name)

        return average_tensor


def average_grads(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of gradients. The outer list is over different 
        towers. The inner list is over the gradient calculation in each tower.
    Returns:
        List of gradients where the gradient has been averaged across all towers.
    """
    if len(tower_grads) == 1:
        return tower_grads[0]
    else:
        average_grads = []
        for grad_ in zip(*tower_grads):
            # Note that each grad looks like the following:
            #   (grad0_gpu0, ... , grad0_gpuN)
            average_grad = None if grad_[0]==None else average_tensors(grad_)
            average_grads.append(average_grad)

        return average_grads

"""
def apply_gradient(update_gradient_vars, grads, optimizer, learning_rate, learning_rate_multipliers=None):
    assert(len(grads)==len(update_gradient_vars))
    if learning_rate_multipliers is None: learning_rate_multipliers = {}
    # Build a dictionary to save multiplier config
    # format -> {scope_name: ((grads, vars), lr_multi)}
    learning_rate_dict = {}
    learning_rate_dict['__default__'] = ([], 1.0)
    for scope, multiplier in learning_rate_multipliers.items():
        assert scope != '__default__'
        learning_rate_dict[scope] = ([], multiplier)

    # Scan all the variables, insert into dict
    scopes = learning_rate_dict.keys()
    for var, grad in zip(update_gradient_vars, grads):
        count = 0
        scope_temp = ''
        for scope in scopes:
            if scope in var.name:
                scope_temp = scope
                count += 1
        assert count <= 1, "More than one multiplier scopes appear in variable: %s" % var.name
        if count == 0: scope_temp = '__default__'
        if grad is not None:
            learning_rate_dict[scope_temp][0].append((grad, var))
     
    # Build a optimizer for each multiplier scope
    apply_gradient_ops = []
    print('\nLearning rate multipliers:')
    for scope, scope_content in learning_rate_dict.items():
        scope_grads_vars, multiplier = scope_content
        print('%s:\n  # variables: %d\n  lr_multi: %f' % (scope, len(scope_grads_vars), multiplier))
        if len(scope_grads_vars) == 0:
            continue
        scope_learning_rate = multiplier * learning_rate
        optimizer_name, optimizer_params = optimizer
        if optimizer_name=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='ADAM':
            opt = tf.train.AdamOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='MOM':
            opt = tf.train.MomentumOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='SGD':
            opt = tf.train.GradientDescentOptimizer(scope_learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')
        apply_gradient_ops.append(opt.apply_gradients(scope_grads_vars))
    print('')
    apply_gradient_op = tf.group(*apply_gradient_ops)

    return apply_gradient_op
"""


def save_model(sess, saver, model_dir, global_step):
    with sess.graph.as_default():
        checkpoint_path = os.path.join(model_dir, 'ckpt')
        metagraph_path = os.path.join(model_dir, 'graph.meta')

        print('Saving variables...')
        saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
        if not os.path.exists(metagraph_path):
            print('Saving metagraph...')
            saver.export_meta_graph(metagraph_path)

def apply_gradient(update_gradient_vars, grads, optimizer, learning_rate, learning_rate_multipliers=None):
    assert(len(grads)==len(update_gradient_vars))
    if learning_rate_multipliers is None: learning_rate_multipliers = {}
    # Build a dictionary to save multiplier config
    # format -> {scope_name: ((grads, vars), lr_multi)}
    learning_rate_dict = {}
    learning_rate_dict['__default__'] = ([], 1.0)
    for scope, multiplier in learning_rate_multipliers.items():
        assert scope != '__default__'
        learning_rate_dict[scope] = ([], multiplier)

    # Scan all the variables, insert into dict
    scopes = learning_rate_dict.keys()
    for var, grad in zip(update_gradient_vars, grads):
        count = 0
        scope_temp = ''
        for scope in scopes:
            if scope in var.name:
                scope_temp = scope
                count += 1
        assert count <= 1, "More than one multiplier scopes appear in variable: %s" % var.name
        #if count == 2:
        #    if 'MinutiaeFeatures' in var.name:
        #        scope_temp = 'MinutiaeFeatures'
        #    elif 'TextureFeatures' in var.name:
        #        scope_temp = 'TextureFeatures'
        #    else:
        #        print("YOU SUCK")
        #        sys.exit(0)
        if count == 0: scope_temp = '__default__'
        if grad is not None:
            learning_rate_dict[scope_temp][0].append((grad, var))
     
    # Build a optimizer for each multiplier scope
    apply_gradient_ops = []
    print('\nLearning rate multipliers:')
    for scope, scope_content in learning_rate_dict.items():
        scope_grads_vars, multiplier = scope_content
        if type(multiplier) is tuple:
            optimizer_name, optimizer_params, scope_multiplier = multiplier
        else:
            optimizer_name, optimizer_params = optimizer
            scope_multiplier = multiplier
        scope_learning_rate = scope_multiplier * learning_rate
        print('{}:\n  #variables: {}\n  optimizer:{}\n lr_multi: {}'.format(
            scope, len(scope_grads_vars), (optimizer_name, optimizer_params), scope_multiplier))
        if len(scope_grads_vars) == 0 or scope_multiplier == 0:
            print('  This scope will be skipped!')
            continue
        if optimizer_name=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='ADAM':
            opt = tf.train.AdamOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='MOM':
            opt = tf.train.MomentumOptimizer(scope_learning_rate, **optimizer_params)
        elif optimizer_name=='SGD':
            opt = tf.train.GradientDescentOptimizer(scope_learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')
        apply_gradient_ops.append(opt.apply_gradients(scope_grads_vars))
    print('')
    apply_gradient_op = tf.group(*apply_gradient_ops)

    return apply_gradient_op

def restore_model(sess, var_list, model_dir, restore_scopes=None, replace_scopes=None, replace=None):
    ''' Load the variable values from a checkpoint file into pre-defined graph.
    Filter the variables so that they contain at least one of the given keywords.'''
    with sess.graph.as_default():
        if restore_scopes is not None:
            var_list = [var for var in var_list if any([scope in var.name for scope in restore_scopes])]
        if replace is not None:
            var_dict = {}
            for var in var_list:
                name_new = var.name
                for k,v in replace.items(): name_new=name_new.replace(k,v)
                name_new = name_new[:-2] # When using dict, numbers should be removed
                var_dict[name_new] = var
            var_list = var_dict
        if replace_scopes is not None:
            var_list = [var for var in var_list if not any([scope in var.name for scope in replace_scopes])]
        model_dir = os.path.expanduser(model_dir)
        ckpt_file = tf.train.latest_checkpoint(model_dir)

        print('Restoring variables from %s ...' % ckpt_file)
        saver = tf.train.Saver(var_list)
        saver.restore(sess, ckpt_file)

def load_model(sess, model_path, scope=None):
    ''' Load the the graph and variables values from a model path.
    Model path is either a a frozen graph or a directory with both
    a .meta file and checkpoint files.'''
    with sess.graph.as_default():
        model_path = os.path.expanduser(model_path)
        if (os.path.isfile(model_path)):
            # Frozen grpah
            print('Model filename: %s' % model_path)
            with gfile.FastGFile(model_path,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(sess, ckpt_file)
