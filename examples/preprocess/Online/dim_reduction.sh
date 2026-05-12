#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_milkt/dim_reduction.py \
    --question_embed_path "/home/jovyan/work/PICKT/data/Online/question_embeddings.pt" \
    --concept_embed_path "/home/jovyan/work/PICKT/data/Online/concept_embeddings.pt" \
    --n_components 64 \
    --dr_type "pca" \
    --save_path "/home/jovyan/work/PICKT/data/Online/reduced_embeddings_pca.json"