#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/dbekt22_dim_reduction.py \
    --question_embed_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/question_embeddings.pt" \
    --concept_embed_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/concept_embeddings.pt" \
    --n_components 64 \
    --dr_type "pca" \
    --save_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/reduced_embeddings_pca.json"