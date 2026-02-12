#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/dbekt22_embeddings.py \
    --text_type "concept" \
    --data_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/KC_Relationships.csv" \
    --preprocess_data_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/KCs.csv"\
    --data_args_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/data_args.json" \
    --save_tensor_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/concept_embeddings.pt" \
    --max_length 64 \
    --chunk_size 64 \
    --device "cuda"