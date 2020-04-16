#!/usr/bin/env python
"""
this is ade args
"""

pretrain = {
    "epoch": [1, 2],
    "do_train": [True],
    "loss_type": ["L2", "CLS"],
    "training_file": ["args_test_data/input/data/unlabel_data/train.ids"],
    "print_steps": [10],
    "save_steps": [100, 10],
    "use_cuda": [True, False],
    "batch_size": [512, 256],
    "hidden_size": [256],
    "emb_size": [256],
    "vocab_size": [484016],
    "learning_rate": [0.001],
    "sample_pro": [0.1, 1.0],
    "max_seq_len": [32, 128],
    "random_seed": [110, 0],
    }


finetune = {
    "epoch": [1, 2],
    "do_train": [True],
    "loss_type": ["CLS", "L2"],
    "init_from_pretrain_model": ["args_test_data/saved_models/trained_models/matching_pretrained/params"],
    "save_model_path": ["args_test_finetuned"],
    "save_param": ["params"],
    "training_file": ["args_test_data/input/data/label_data/seq2seq_att/train.ids"],
    "print_steps": [10],
    "save_steps": [100, 10],
    "use_cuda": [True, False],
    "batch_size": [512, 256],
    "hidden_size": [256],
    "emb_size": [256],
    "vocab_size": [484016],
    "learning_rate": [0.001],
    "sample_pro": [0.1, 1.0],
    "max_seq_len": [32, 128],
    "random_seed": [110, 0],
    }


predict = {
    "do_predict": [True],
    "loss_type": ["CLS", "L2"],
    "init_from_params": ["args_test_data/saved_models/trained_models/seq2seq_att_finetuned/params"],
    "predict_file": ["args_test_data/input/data/label_data/seq2seq_att/test.ids"],
    "use_cuda": [True, False],
    "batch_size": [512, 256],
    "hidden_size": [256],
    "emb_size": [256],
    "vocab_size": [484016],
    "output_prediction_file": ["args_test_output_predict"],
    }


infer = {
    "do_save_inference_model": [True],
    "init_from_params": ["args_test_data/saved_models/trained_models/seq2seq_att_finetuned/params"],
    "inference_model_dir": ["args_test_inference_model"],
    "use_cuda": [True, False],
    }
