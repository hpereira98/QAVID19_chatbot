#!/bin/sh

if [ $# -ne 1 ]; then
    echo "You need to provide dataset name"
    exit 1
fi

export DATASET_NAME=$1

export DATA_DIR=./data/${DATASET_NAME} 
export MODEL_DIR=./models


python3 run_squad.py  \
    --model_type bert   \
    --model_name_or_path bert-base-multilingual-cased  \
    --output_dir models/bert/ \
    --data_dir data/${DATASET_NAME}    \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_train  \
    --train_file train-v2.0.json   \
    --version_2_with_negative \
    --do_eval   \
    --predict_file dev-v2.0.json   \
    --learning_rate 3e-5   \
    --num_train_epochs 3   \
    --max_seq_length 384   \
    --doc_stride 128   \
    --threads 10   \
    --save_steps 5000
