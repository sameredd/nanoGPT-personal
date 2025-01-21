import os

# eval / logging
eval_interval = 4000
log_interval = 2
eval_iters = int(eval_interval * 0.07)
# wandb logging
wandb_project = 'nanogpt'
wandb_run_name = 'owt_small_v2'
out_dir = os.path.join('out', wandb_run_name)

# data
dataset = 'openwebtext' # 'openwebtext' or 'shakespeare_char' or 'math
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size


# model # NOTE MODERN BERT USES DEEP AND NARROW LAYERS! 22 layers, 768 hidden size, 150 M params
# Total parameters: 47.65M
# Embedding parameters: 19.32M
# Non-embedding parameters: 28.33M
block_size = 256 # sequence length
n_layer = 21
n_embd = 448 # Should be divisible by 64
n_head = n_embd // 64 # 64 dim heads
factorize_embed = False
norm_type = 'rms' # either 'layer' or 'rms' or 'none' for LayerNorm or RmsNorm
norm_gain = False
norm_bias = False

# training length -tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
max_iters = 2.5e4 # total number of training iterations

# learning rate schedule
learning_rate = 1e-5 # max learning rate
warmup_iters = max_iters * 0.03 # how many steps to warm up for
decay_lr = True # whether to decay the learning rate
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate * 0.33
# min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per


