# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, concatenate_datasets # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # See configs here - https://huggingface.co/datasets/math-ai/AutoMathText
    datasets = [
        load_dataset("math-ai/AutoMathText", 'web-0.50-to-1.00', num_proc=num_proc_load_dataset, trust_remote_code=True)['train'],
        # load_dataset("math-ai/AutoMathText", 'arxiv-full', num_proc=num_proc_load_dataset, trust_remote_code=True)['train'], 
    ] # Only train split available
    keep_keys = ['text']
    datasets = [d.select_columns(keep_keys) for d in datasets]
    dataset = concatenate_datasets(datasets)
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    
    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 512

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Saved {split} split with {arr_len} tokens to {filename}")

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r') # length = # tokens

    # Example
    # $ python -c "fp='/Users/sameredd/my_stuff/nanoGPT-personal/data/math/train.bin'; import numpy as np; m = np.memmap(fp, dtype=np.uint16, mode='r'); print(f'{len(m)/1e9:.2f}B tokens')"