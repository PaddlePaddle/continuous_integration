#!/usr/bin/env python
"""
this is transformer args
"""

train = {
    "src_vocab_fpath":
    ["args_test_data/wmt16_ende_data_bpe/vocab_all.bpe.32000"],
    "trg_vocab_fpath":
    ["args_test_data/wmt16_ende_data_bpe/vocab_all.bpe.32000"],
    "use_token_batch": [True],
    "batch_size": [512, 256],
    "pool_size": [200000, 100000],
    "sort_type": ["pool", "global", "none"],
    "shuffle": [True, False],
    "shuffle_batch": [True, False],
    "special_token": ["'<s>' '<e>' '<unk>'"],
    "token_delimiter": ["' '"],
    "use_cuda": [True],
    "epoch": [1, 2],
    "init_from_pretrain_model":
    ["args_test_model/base_model_params/step_final", None],
    "init_from_params": ["args_test_model/base_model_params/step_final"],
    "save_model_path": ["args_test_finetuned"],
    "save_checkpoint": ["args_test_finetuned"],
    "save_param": ["args_test_finetuned"],
    "inference_model_dir": ["args_test_inference_model"],
    "random_seed": [None, 100, 0],
    "training_file":
    ["args_test_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de"],
    "predict_file":
    ["args_test_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de"],
    "output_file": ["predict.txt"],
    "warmup_steps": [100],
    "beam_size": [1, 20],
    "max_out_len": [256, 512],
    "n_best": [1, 2],
    "do_train": [True],
    "do_eval": [False, True],
}

predict = {
    "src_vocab_fpath":
    ["args_test_data/wmt16_ende_data_bpe/vocab_all.bpe.32000"],
    "trg_vocab_fpath":
    ["args_test_data/wmt16_ende_data_bpe/vocab_all.bpe.32000"],
    "use_token_batch": [True],
    "batch_size": [512, 256],
    "special_token": ["'<s>' '<e>' '<unk>'"],
    "token_delimiter": ["' '"],
    "use_cuda": [True, False],
    "init_from_params": ["args_test_model/base_model_params/step_final"],
    "training_file":
    ["args_test_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de"],
    "predict_file":
    ["args_test_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de"],
    "output_file": ["predict.txt"],
    "beam_size": [1, 20],
    "max_out_len": [256, 512],
    "n_best": [1, 2],
    "do_train": [False],
    "do_predict": [True],
}
