import os
log_interval = 1
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'nanogpt'
wandb_run_name = 'minigpt-shakespeare-default'
out_dir = os.path.join('out', wandb_run_name)
# data
dataset = 'shakespeare_char' # 'openwebtext' or 'shakespeare_char'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 128  # sequence length 

# model (1.6M param)
n_layer = 8
n_head = 4
n_embd = 128
norm_gain = False
norm_bias = False
norm_type = 'rms' # either 'layer' or 'rms' or 'none' for LayerNorm or RmsNorm

# learning rate and scheduler
learning_rate = 1e-4 # max learning rate
eval_interval = 2000
eval_iters = 100
max_iters = 20000 # total number of training iterations
# learning rate decay
warmup_iters = max_iters * 0.02 # how many steps to warm up for
decay_lr = True # whether to decay the learning rate
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate * 0.33
# min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per