
#!/bin/sh

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/lexical_analysis/data args_test_data
fi

#prepare pre_model
if [ -e args_test_pretrained ]
then
    echo "args_test_pretrained has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/lexical_analysis/pretrained args_test_pretrained
fi

if [ -e args_test_model_baseline ]
then
    echo "args_test_model_baseline has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/lexical_analysis/model_baseline.pdckpt args_test_model_baseline.pdckpt
fi

if [ -e args_test_model_finetuned ]
then
    echo "args_test_model_finetuned has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/lexical_analysis/model_finetuned args_test_model_finetuned
fi
