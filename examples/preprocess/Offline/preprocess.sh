#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_milkt/preprocess.py \
    --milkt_solved_history_path "/home/jovyan/work/PICKT/data/Offline/HME-Cold_Start.csv" \
    --data_args_path "/home/jovyan/work/PICKT/data/Online/data_args.json" \
    --datasets_save_path "/home/jovyan/work/PICKT/data/Offline/preprocessed_data.parquet"
