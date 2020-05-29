#!/usr/bin/env python
"""
this is emotion_detection args
"""

train_textcnn = {
    "init_checkpoint": ["./args_test_models/textcnn", None],
    "save_checkpoint_dir": ["args_test_output_1", "args_test_output_2"],
    "epoch": [1, 2],
    "save_steps": [10000, 1000],
    "validation_steps": [1000, 100],
    "skip_steps": [10, 20],
    "verbose": [False, True],
    "data_dir": ["./args_test_data/"],
    "vocab_path": ["./args_test_data/vocab.txt"],
    "batch_size": [512, 256],
    "random_seed": [0, 100],
    "use_cuda": [True, False],
    "task_name": ["emotion_detection"],
    "do_train": [True],
    "do_val": [False, True],
    "do_infer": [False, True],
}

evaluate_textcnn = {
    "task_name": ["emotion_detection"],
    "use_cuda": [True, False],
    "do_train": [False],
    "do_val": [True],
    "verbose": [True, False],
    "batch_size": [256, 128],
    "init_checkpoint": ["./args_test_models/textcnn"],
    "data_dir": ["./args_test_data/"],
    "vocab_path": ["./args_test_data/vocab.txt"],
}

infer_textcnn = {
    "task_name": ["emotion_detection"],
    "use_cuda": [True, False],
    "do_train": [False],
    "do_infer": [True],
    "verbose": [True, False],
    "batch_size": [256, 32],
    "init_checkpoint": ["./args_test_models/textcnn"],
    "data_dir": ["./args_test_data/"],
    "vocab_path": ["./args_test_data/vocab.txt"],
}

train_ernie = {
    "ernie_config_path":
    ["./args_test_models/ernie_finetune/ernie_config.json"],
    "init_checkpoint": ["./args_test_models/ernie_finetune/params"],
    "save_checkpoint_dir": ["args_test_output_1", "args_test_output_2"],
    "use_paddle_hub": [False, True],
    "epoch": [1, 2],
    "save_steps": [500, 100],
    "validation_steps": [50, 100],
    "skip_steps": [50, 10],
    "verbose": [False, True],
    "vocab_path": ["./args_test_models/ernie_finetune/vocab.txt"],
    "batch_size": [32, 24],
    "random_seed": [0, 100],
    "num_labels": [3],
    "max_seq_len": [64, 32],
    "train_set": ["./args_test_data/train.tsv"],
    "test_set": ["./args_test_data/test.tsv"],
    "dev_set": ["./args_test_data/dev.tsv"],
    "infer_set": ["./args_test_data/infer.tsv"],
    "label_map_config": [None],
    "do_lower_case": [True, False],
    "use_cuda": [True],
    "do_train": [True],
    "do_val": [False, True],
    "do_infer": [False, True],
}

evaluate_ernie = {
    "ernie_config_path":
    ["./args_test_models/ernie_finetune/ernie_config.json"],
    "init_checkpoint": ["./args_test_models/ernie_finetune/params"],
    "use_paddle_hub": [False, True],
    "verbose": [False, True],
    "vocab_path": ["./args_test_models/ernie_finetune/vocab.txt"],
    "batch_size": [32, 24],
    "num_labels": [3],
    "max_seq_len": [128, 64],
    "test_set": ["./args_test_data/test.tsv"],
    "label_map_config": [None],
    "do_lower_case": [True, False],
    "use_cuda": [True],
    "do_train": [False],
    "do_val": [True],
}

infer_ernie = {
    "ernie_config_path":
    ["./args_test_models/ernie_finetune/ernie_config.json"],
    "init_checkpoint": ["./args_test_models/ernie_finetune/params"],
    "use_paddle_hub": [False, True],
    "verbose": [False, True],
    "vocab_path": ["./args_test_models/ernie_finetune/vocab.txt"],
    "batch_size": [32, 24],
    "num_labels": [3],
    "infer_set": ["./args_test_data/infer.tsv"],
    "label_map_config": [None],
    "do_lower_case": [True, False],
    "use_cuda": [True],
    "do_train": [False],
    "do_infer": [True],
}
