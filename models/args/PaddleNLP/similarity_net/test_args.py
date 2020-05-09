#!/usr/bin/env python

"""
this is similarity_net args
"""

train = {
        "config_path": ["./config/bow_pairwise.json"],
        "output_dir": ["args_test_output_1", "args_test_output_2"],
        "task_mode": ["pairwise"],
        "epoch": [1, 2],
        "save_steps": [1, 2],
        "validation_steps": [1, 2],
        "skip_steps": [10, 20],
        "verbose_result": [False, True],
        "train_data_dir": ["./args_test_data/train_pairwise_data"],
        "valid_data_dir": ["./args_test_data/test_pairwise_data"],
        "test_data_dir": ["./args_test_data/test_pairwise_data"],
        "infer_data_dir": ["./args_test_data/infer_data"],
        "vocab_path": ["./args_test_data/term2id.dict"],
        "batch_size": [256, 128],
        "use_cuda": [True, False],
        "task_name": ["simnet"],
        "do_train": [True],
        "do_valid": [True],
        "do_test": [False, True],
        "do_infer": [False, True],
        "compute_accuracy": [False, True],
        "init_checkpoint": ['""'],
        }

evaluate = {
        "task_name": ["simnet"],
        "task_mode": ["pairwise"],
        "use_cuda": [True, False],
        "do_train": [False],
        "do_test": [True],
        "verbose_result": [True, False],
        "batch_size": [256, 128],
        "init_checkpoint": ["./args_test_model_files/simnet_bow_pairwise_pretrained_model/"],
        "test_data_dir": ["./args_test_data/test_pairwise_data"],
        "vocab_path": ["./args_test_data/term2id.dict"],
        "config_path": ["./config/bow_pairwise.json"],
        "test_result_path": ["./test_result"],
        "compute_accuracy": [False, True],
        }

infer = {
        "task_name": ["simnet"],
        "task_mode": ["pairwise"],
        "use_cuda": [True, False],
        "do_train": [False],
        "do_infer": [True],
        "verbose_result": [True, False],
        "batch_size": [256, 128],
        "init_checkpoint": ["./args_test_model_files/simnet_bow_pairwise_pretrained_model/"],
        "infer_data_dir": ["./args_test_data/infer_data"],
        "vocab_path": ["./args_test_data/term2id.dict"],
        "config_path": ["./config/bow_pairwise.json"],
        "infer_result_path": ["./infer_result"],
        "compute_accuracy": [False, True],
        }
