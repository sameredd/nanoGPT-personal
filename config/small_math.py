import os
eval_interval = 2000
log_interval = 2
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'nanogpt'
wandb_run_name = 'math_small-v3-gain_rmsnorm-L16-H384'
out_dir = os.path.join('out', wandb_run_name)
# data
dataset = 'math' # 'openwebtext' or 'shakespeare_char' or 'math
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size


# model # NOTE MODERN BERT USES DEEP AND NARROW LAYERS! 22 layers, 768 hidden size, 150 M params


# Total parameters: 47.65M
# Embedding parameters: 19.32M
# Non-embedding parameters: 28.33M

block_size = 256 # sequence length
n_layer = 16
n_embd = 384 # Should be divisible by 64
n_head = n_embd // 64 # 64 dim heads
norm_gain = True
norm_bias = False
norm_type = 'rms' # either 'layer' or 'rms' or 'none' for LayerNorm or RmsNorm
factorize_embed = False

# learning rate and scheduler
learning_rate = 1e-5 # max learning rate
max_iters = 1e6 # total number of training iterations
# learning rate decay
warmup_iters = max_iters * 0.03 # how many steps to warm up for
decay_lr = True # whether to decay the learning rate
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate * 0.33
# min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per


# TOKENS PER BATCH = batch_size * grad_accum * block_size = 2048 = 2e3
# 1e6 steps = 2e9 tokens = 2B tokens = 1 epoch