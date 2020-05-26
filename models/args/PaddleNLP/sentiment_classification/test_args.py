#!/usr/bin/env python
"""
this is sentiment_classification args
"""

run_classifier = {
    "task_name": ["senta"],
    "use_cuda": [True, False],
    "epoch": [1, 2],
    "batch_size": [256, 128],
    "random_seed": [0, 100],
    "checkpoints": ["model_a", "model_b"],
    "data_dir": ["./args_test_senta_data/", "./args_test_senta_data_1"],
    "vocab_path": [
        "./args_test_senta_data/word_dict.txt",
        "./args_test_senta_data_1/word_dict.txt"
    ],
    "senta_config_path": ["./senta_config.json"],
    "verbose": [False, True],
    "skip_steps": [10, 20],
    "save_steps": [10000, 5000],
    "validation_steps": [1000, 2000],
    "init_checkpoint": [None, "args_test_senta_model/bilstm_model/"],
    "do_val": [False, True],
    "do_infer": [False, True],
}

run_ernie_classifier = {
    "model_type": ["ernie_base"],
    "use_cuda": [True],
    "epoch": [1, 2],
    "batch_size": [8, 4],
    "random_seed": [0, 100],
    "checkpoints": ["model_a", "model_b"],
    "vocab_path": ["./args_test_senta_model/ernie_pretrain_model/vocab.txt"],
    "senta_config_path": ["./senta_config.json"],
    "verbose": [False, True],
    "skip_steps": [10, 20],
    "save_steps": [200, 1000],
    "validation_steps": [100, 200],
    "init_checkpoint": ["./args_test_senta_model/ernie_pretrain_model/params"],
    "ernie_config_path":
    ["./args_test_senta_model/ernie_pretrain_model/ernie_config.json"],
    "train_set":
    ["./args_test_senta_data/train.tsv", "./args_test_senta_data_1/train.tsv"],
    "test_set":
    ["./args_test_senta_data/test.tsv", "./args_test_senta_data_1/test.tsv"],
    "dev_set":
    ["./args_test_senta_data/dev.tsv", "./args_test_senta_data_1/dev.tsv"],
    "max_seq_len": [256],
    "use_paddle_hub": [False, True],
    "do_val": [False, True],
    "do_infer": [False, True],
}
