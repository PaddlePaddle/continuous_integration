#!/usr/bin/env python

"""
this is xlnet args
"""

finetune_squad = {
    "model_config_path": ["xlnet_cased_L-24_H-1024_A-16/xlnet_config.json"],
    "spiece_model_file": ["xlnet_cased_L-24_H-1024_A-16/spiece.model"],
    "init_checkpoint": ["xlnet_cased_L-24_H-1024_A-16/params"],
    "checkpoints": ["checkpoints_squad_2.0"],
    "epoch": [1],
    "train_steps": [10],
    "warmup_steps": [5],
    "save_steps": [5],
    "skip_steps": [5],
    "train_file": ["args_test_data/squad2.0/train-v2.0.json"],
    "predict_file": ["args_test_data/squad2.0/dev-v2.0.json"],
    "random_seed": [0, 100],
    "use_cuda": [True],
    "uncased": [False],
    "verbose": [True]
}

finetune_cls = {
    "do_train": [True],
    "do_eval": [False, True],
    "do_predict": [True, False],
    "task_name": ["sts-b"],
    "data_dir": ["args_test_data/STS-B"],
    "checkpoints": ["checkpoints_sts-b"],
    "uncased": [False],
    "spiece_model_file": ["xlnet_cased_L-24_H-1024_A-16/spiece.model"],
    "model_config_path": ["xlnet_cased_L-24_H-1024_A-16/xlnet_config.json"],
    "init_pretraining_params": ["xlnet_cased_L-24_H-1024_A-16/params"],
    "max_seq_length": [128],
    "train_batch_size": [8],
    "learning_rate": [5e-5],
    "predict_dir": ["exp/sts-b"],
    "skip_steps": [10],
    "train_steps": [1200],
    "warmup_steps": [120],
    "save_steps": [600],
    "is_regression": [True],
    "epoch": [1],
    "verbose": [True],
    "random_seed": [0, 100],
    "use_cuda": [True],
    "shuffle": [True, False],
    "eval_batch_size": [2],
}
