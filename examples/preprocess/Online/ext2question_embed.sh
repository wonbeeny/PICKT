#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_milkt/embeddings.py \
    --text_type "question" \
    --model_path "snumin44/simcse-ko-roberta-supervised" \
    --question_text_data_path "/home/jovyan/work/PICKT/data/Online/MilkT-Question_Text.csv" \
    --data_args_path "/home/jovyan/work/PICKT/data/Online/data_args.json" \
    --save_tensor_path "/home/jovyan/work/PICKT/data/Online/question_embeddings.pt" \
    --max_length 512 \
    --batch_size 64 \
    --device "cuda"