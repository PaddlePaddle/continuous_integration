#!/usr/bin/env python
"""
this is seq2seq args
"""

train = {
    "tar_lang": ["vi"],
    "src_lang": ["en"],
    "attention": [True, False],
    "num_layers": [2],
    "hidden_size": [512, 256],
    "src_vocab_size": [17191],
    "tar_vocab_size": [7709],
    "batch_size": [128, 64],
    "dropout": [0.2, 0.1],
    "init_scale": [0.1, 0.2],
    "max_grad_norm": [5.0, 4.9],
    "train_data_prefix": ["data/en-vi/train"],
    "eval_data_prefix": ["data/en-vi/tst2012"],
    "test_data_prefix": ["data/en-vi/tst2013"],
    "vocab_prefix": ["data/en-vi/vocab"],
    "use_gpu": [True],
    "model_path": ["./attention_models", "./models"],
    "max_epoch": [1, 2],
    "optimizer": ["adam"],
    "learning_rate": ["0.001"],
}

infer = {
    "tar_lang": ["vi"],
    "src_lang": ["en"],
    "attention": [True],
    "num_layers": [2],
    "hidden_size": [512],
    "src_vocab_size": [17191],
    "tar_vocab_size": [7709],
    "batch_size": [128, 64],
    "dropout": [0.2, 0.1],
    "init_scale": [0.1, 0.2],
    "max_grad_norm": [5.0, 4.9],
    "vocab_prefix": ["data/en-vi/vocab"],
    "infer_file": ["data/en-vi/tst2013.en", "data/en-vi/tst2012.en"],
    "reload_model": ["attention_models/epoch_0"],
    "use_gpu": [True],
    "infer_output_file": [
        "attention_infer_output/infer_output.txt",
        "infer_output/infer_output.txt"
    ],
    "beam_size": [5, 10],
}
