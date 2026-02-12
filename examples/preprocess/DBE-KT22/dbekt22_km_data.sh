#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/dbekt22_km_data.py \
    --KC_Relations_file_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/KC_Relationships.csv" \
    --KCs_file_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/KCs.csv" \
    --Question_KC_Relationships_file_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/Question_KC_Relationships.csv"\
    --data_args_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/data_args.json" \
    --reduced_embeds_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/reduced_embeddings_pca.json" \
    --km_data_save_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/km_data.json"
