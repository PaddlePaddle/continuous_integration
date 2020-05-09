#!/usr/bin/env python

"""
this is language_model args
"""

train = {
        "use_gpu": [True, False],
        "parallel": [True, False],
        "max_epoch": [1, 2],
        "batch_size": [32, 16],
        "model_type": ["test", "small"],
        "rnn_model": ["static", "padding"],
        "data_path": ["args_test_data_1/simple-examples/data/", "args_test_data_2/simple-examples/data/"],
        "save_model_dir": ["models_1", "models_2"],
        "log_path": ["tmp_log_file", None],
        }
