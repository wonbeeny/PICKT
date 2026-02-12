#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/dbekt22_split_datasets.py \
    --data_args_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/data_args.json" \
    --preprocessed_data_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/preprocessed_dbekt22_data.parquet" \
    --save_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/"
