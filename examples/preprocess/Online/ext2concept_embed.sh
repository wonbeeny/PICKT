#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_milkt/embeddings.py \
    --text_type "concept" \
    --model_path "snumin44/simcse-ko-roberta-supervised" \
    --question_text_data_path "/home/jovyan/work/PICKT/data/Online/MilkT-Concept_List.csv" \
    --data_args_path "/home/jovyan/work/PICKT/data/Online/data_args.json" \
    --save_tensor_path "/home/jovyan/work/PICKT/data/Online/concept_embeddings.pt" \
    --max_length 64 \
    --batch_size 64 \
    --device "cuda"