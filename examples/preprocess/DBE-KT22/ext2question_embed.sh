#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/dbekt22_embeddings.py \
    --text_type "question" \
    --data_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/Questions.csv" \
    --data_args_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/data_args.json" \
    --save_tensor_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/question_embeddings.pt" \
    --max_length 512 \
    --chunk_size 64 \
    --device "cuda"