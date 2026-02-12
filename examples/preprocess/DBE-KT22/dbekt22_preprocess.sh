#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/dbekt22_preprocess.py \
    --log_file_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/Transaction.csv" \
    --question_choice_file_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/Question_Choices.csv"\
    --birt_file_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/dbe-kt22_birt.csv"\
    --data_args_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/data_args.json" \
    --km_data_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/km_data.json" \
    --datasets_save_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/preprocessed_dbekt22_data.parquet"
