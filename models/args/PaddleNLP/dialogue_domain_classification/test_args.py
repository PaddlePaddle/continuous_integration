#!/usr/bin/env python
"""
this is dialogue_domain_classification args
"""

train = {
    "use_cuda": [True, False],
    "do_train": [True],
    "do_eval": [False, True],
    "do_test": [False, True],
    "build_dict": [False],
    "data_dir": ['./data/input/'],
    "save_dir": ['./data/output/'],
    "config_path": ['./data/input/model.conf'],
    "batch_size": [64, 32],
    "max_seq_len": [50, 60],
    "checkpoints": ['checkpoints'],
    "init_checkpoint": [None],
    "skip_steps": [10, 100],
    "cpu_num": [3],
    "validation_steps": [100, 200],
    "learning_rate": [0.1, 0.05],
}
