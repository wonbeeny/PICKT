#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_milkt/split_datasets.py \
    --data_args_path "/home/jovyan/work/PICKT/data/Online/data_args.json" \
    --preprocessed_data_path "/home/jovyan/work/PICKT/data/Offline/preprocessed_data.parquet" \
    --datasets_save_path "/home/jovyan/work/PICKT/data/Offline/"
