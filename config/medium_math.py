import os

init_from = "/teamspace/studios/this_studio/nanoGPT-personal/out/owt_

# dataset and run name configs
dataset = 'openwebtext' # 'openwebtext' or 'shakespeare_char' or 'math
wandb_project = 'nanogpt'
wandb_run_name = 'math_medium_init_owt_v1'
out_dir = os.path.join('out', wandb_run_name)

# eval / logging
eval_interval = 2000
eval_iters = 100

# data 
# (tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size)
gradient_accumulation_steps = 4 # used to simulate larger batch sizes
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256 # sequence length

# model
# NOTE MODERN BERT USES DEEP AND NARROW LAYERS! 22 layers, 768 hidden size, 150 M params
n_layer = 25
n_embd = 512 # Should be divisible by 64
n_head = n_embd // 64 # 64 dim heads
norm_type = 'rms' # either 'layer' or 'rms' or 'none' for LayerNorm or RmsNorm
norm_gain = False
norm_bias = False
factorize_embed = False

# training length 
max_iters = 12500 # total number of training iterations

# learning rate schedule
learning_rate = 1e-4 # max learning rate
min_lr = learning_rate * 0.1
warmup_iters = max_iters * 0.05 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
decay_lr = True # whether to decay the learning rate