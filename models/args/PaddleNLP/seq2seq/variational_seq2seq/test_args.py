#!/usr/bin/env python
"""
this is variational_seq2seq args
"""

train = {
    "vocab_size": [10003],
    "batch_size": [32, 16],
    "init_scale": [0.1, 0.2],
    "max_grad_norm": [5.0, 4.9],
    "dataset_prefix": ["data/ptb/ptb", "data/swda/swda"],
    "model_path": ["ptb_model", "swda_model"],
    "use_gpu": [True],
    "max_epoch": [1, 2],
}

infer = {
    "vocab_size": [10003],
    "batch_size": [32, 16],
    "init_scale": [0.1, 0.2],
    "max_grad_norm": [5.0, 4.9],
    "dataset_prefix": ["data/ptb/ptb", "data/swda/swda"],
    "use_gpu": [True],
    "reload_model": ["ptb_model/epoch_0", "swda_model/epoch_0"],
}
