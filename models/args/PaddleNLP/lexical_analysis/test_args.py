#!/usr/bin/env python

"""
this is LAC args
"""

train_base = {
    "random_seed": [0, 100],
    "batch_size": [300, 500],
    "epoch": [1, 2],
    "use_cuda": [True, False],
    "traindata_shuffle_buffer": [2000, 1000],
    "enable_ce": [False],
    "word_dict_path": ["./conf/word.dic"],
    "label_dict_path": ["./conf/tag.dic"],
    "train_data": ["args_test_data/train.tsv"],
    "test_data": ["args_test_data/test.tsv"],
    "infer_data": ["args_test_data/test.tsv"],
    "model_save_dir": ["output"],
    }


eval_base = {
    "init_checkpoint": ["./args_test_model_baseline"],
    "batch_size": [300, 500],
    "use_cuda": [True, False],
    "word_dict_path": ["./conf/word.dic"],
    "label_dict_path": ["./conf/tag.dic"],
    "test_data": ["args_test_data/test.tsv"],
    }


infer_base = {
    "init_checkpoint": ["./args_test_model_baseline"],
    "batch_size": [300, 500],
    "use_cuda": [True, False],
    "word_dict_path": ["./conf/word.dic"],
    "label_dict_path": ["./conf/tag.dic"],
    "infer_data": ["args_test_data/test.tsv"],
    }


train_ernie = {
    "ernie_config_path": ["args_test_pretrained/ernie_config.json"],
    "vocab_path": ["args_test_pretrained/vocab.txt"],
    "mode": ["train"],
    "init_pretraining_params": ["args_test_pretrained/params/"],
    "random_seed": [0, 100],
    "batch_size": [64, 3],
    "epoch": [1, 2],
    "use_cuda": [True, False],
    "max_seq_len": [128, 256],
    "do_lower_case": [True, False],
    "train_data": ["args_test_data/train.tsv"],
    "test_data": ["args_test_data/test.tsv"],
    "infer_data": ["args_test_data/test.tsv"],
    "model_save_dir": ["output"],
    }


eval_ernie = {
    "ernie_config_path": ["args_test_pretrained/ernie_config.json"],
    "vocab_path": ["args_test_pretrained/vocab.txt"],
    "init_checkpoint": ["args_test_model_finetuned"],
    "mode": ["eval"],
    "batch_size": [64],
    "use_cuda": [True, False],
    "max_seq_len": [128],
    "do_lower_case": [True, False],
    "test_data": ["args_test_data/test.tsv"],
    }


infer_ernie = {
    "ernie_config_path": ["args_test_pretrained/ernie_config.json"],
    "vocab_path": ["args_test_pretrained/vocab.txt"],
    "init_checkpoint": ["args_test_model_finetuned"],
    "mode": ["infer"],
    "batch_size": [64],
    "random_seed": [0, 100],
    "use_cuda": [True, False],
    "max_seq_len": [128],
    "do_lower_case": [True, False],
    "test_data": ["args_test_data/test.tsv"],
    }
