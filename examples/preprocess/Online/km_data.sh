#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_milkt/km_data.py \
    --knowledge_map_file_path "/home/jovyan/work/PICKT/data/Online/MilkT-Knowledge_Map.csv" \
    --question_meta_path "/home/jovyan/work/PICKT/data/Online/Total-Question_Meta.csv" \
    --data_args_path "/home/jovyan/work/PICKT/data/Online/data_args.json" \
    --reduced_embeds_path "/home/jovyan/work/PICKT/data/Online/reduced_embeddings_pca.json" \
    --km_data_save_path "/home/jovyan/work/PICKT/data/Online/km_data.json"
