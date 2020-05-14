
#!/bin/sh

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/emotion_detection/data args_test_data
fi

#prepare pre_model
if [ -e  args_test_models ]
then
    echo "models has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/emotion_detection/pretrain_models args_test_models
fi

#
if [ -e args_test_output_1 ]
then
    echo "args_test_output_1 has already existed"
else
    mkdir args_test_output_1
fi

if [ -e args_test_output_2 ]
then
    echo "args_test_output_2 has already existed"
else
    mkdir args_test_output_2
fi
