''' Config Proto '''

import sys
import os

# change restore model
# change learning_rate_multipliers
# change other parameters



####### INPUT OUTPUT #######
dataset = 'deepprint'

# The name of the current model for output
name = 'mds_wrapper_resnet3x_{}'.format(dataset)

# The folder to save log and model
log_base_dir = './log/'

# The interval between writing summary
summary_interval = 100

# Number of GPUs
num_gpus = 1

####### MINING STRAT #######
hard_mining = True
candidate_pairs = 1000

####### NETWORK #######

# The network architecture
network = "nets/stages_resnet.py"
res_layers=3
long_skip=True

# Model version, only for some networks
model_version = None

# number of dimensions in input
input_size = 192

####### TRAINING STRATEGY #######

# Optimizer
#optimizer = ("SGD", {})
optimizer = ("ADAM", {"beta1": 0.9, "beta2": 0.999})

# Number of samples per batch
num_pairs = 250
batch_size = 4 * num_pairs

# Number of batches per epoch
epoch_size = 1000

# Number of epochs
num_epochs = 200

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 3e-4
#lr = 3e-5
learning_rate_schedule = {
    0: 1 * lr,
    25000: .9 * lr,
    50000: .5 * lr,
    75000: .1 * lr,
    100000: .05 * lr,
    125000: .02 * lr,
    150000: .01 * lr
}

# Multiply the learning rate for variables that contain certain keywords
learning_rate_multipliers = {
    #'Stage_0': 1.0,
    #'Stage_1': 1.0,
    #'Stage_2': 1.0,
    #'Stage_3': 1.0,
    #'Stage_4': 1.0,
    #'Stage_5': 0.0,
}

# Restore model
# checkpoint folder
restore_model = '/user/engelsm7/DeepMDS++/log/mds_wrapper_resnet3x_deepprint/8'
#restore_model=None

# Keywords to filter restore variables, set None for all
restore_scopes = None

# Weight decay for model variables
weight_decay = 4e-5

# Keep probability for dropouts
keep_prob = 0.8

augment = False
obfuscate_percentage = 0.33 # percentage of zeros

#32, 16, 64
current_stage = 4
stages = {
    0: (192, 128),
    1: (128, 64),
    2: (64, 32),
    3: (32, 16),
    4: (16, 8),
    5: (8, 4)
}

####### LOSS FUNCTION #######

# Scale for the logits
losses = {
    'distances_loss' : {'gweight' : 1.0, 'iweight' : 1.0},
    'cov_loss' : {'weight' : 10.0}
}

