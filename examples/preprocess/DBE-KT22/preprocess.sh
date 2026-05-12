#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/preprocess.py \
    --log_file_path "/home/jovyan/work/PICKT/data/DBE-KT22/Transaction.csv" \
    --question_choice_file_path "/home/jovyan/work/PICKT/data/DBE-KT22/Question_Choices.csv"\
    --questions_path "/home/jovyan/work/PICKT/data/DBE-KT22/Questions.csv" \
    --data_args_path "/home/jovyan/work/PICKT/data/DBE-KT22/data_args.json" \
    --km_data_path "/home/jovyan/work/PICKT/data/DBE-KT22/km_data.json" \
    --datasets_save_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed_data.parquet"
