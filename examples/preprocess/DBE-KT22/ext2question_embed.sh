#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/embeddings.py \
    --text_type "question" \
    --data_path "/home/jovyan/work/PICKT/data/DBE-KT22/Questions.csv" \
    --data_args_path "/home/jovyan/work/PICKT/data/DBE-KT22/data_args.json" \
    --save_tensor_path "/home/jovyan/work/PICKT/data/DBE-KT22/question_embeddings.pt" \
    --max_length 512 \
    --chunk_size 64 \
    --device "cuda"