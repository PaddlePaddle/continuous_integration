#!/usr/bin/env python
"""
this is dgu args
"""

train = {
    "use_cuda": [True],
    "epoch": [1, 2],
    "do_train": [True],
    "task_name": ["atis_intent"],
    "batch_size": [32],
    "do_lower_case": [True, False],
    "data_dir": ["./args_test_data/input/data/atis/atis_intent"],
    "bert_config_path": [
        "./args_test_data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json"
    ],
    "vocab_path":
    ["./args_test_data/pretrain_model/uncased_L-12_H-768_A-12/vocab.txt"],
    "init_from_pretrain_model":
    ["./args_test_data/pretrain_model/uncased_L-12_H-768_A-12/params"],
    "save_model_path": ["args_test_save_model"],
    "save_param": ["params"],
    "save_steps": [100, 10],
    "learning_rate": [0.001],
    "weight_decay": [0.01],
    "max_seq_len": [128],
    "print_steps": [10],
    "use_fp16": [False, True],
    "verbose": [False, True],
    "random_seed": [0, 100],
}

predict = {
    "use_cuda": [True],
    "do_predict": [True],
    "task_name": ["atis_intent"],
    "batch_size": [32],
    "do_lower_case": [True, False],
    "data_dir": ["./args_test_data/input/data/atis/atis_intent"],
    "bert_config_path": [
        "./args_test_data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json"
    ],
    "vocab_path":
    ["./args_test_data/pretrain_model/uncased_L-12_H-768_A-12/vocab.txt"],
    "init_from_pretrain_model":
    ["./args_test_data/pretrain_model/uncased_L-12_H-768_A-12/params"],
    "output_prediction_file": ["args_test_output_predict"],
    "max_seq_len": [128],
    "verbose": [False, True],
}

infer = {
    "task_name": ["atis_intent"],
    "do_save_inference_model": [True],
    "use_cuda": [True],
    "bert_config_path": [
        "./args_test_data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json"
    ],
    "init_from_pretrain_model":
    ["./args_test_data/pretrain_model/uncased_L-12_H-768_A-12/params"],
    "inference_model_dir": ["args_test_inference_model"],
}
