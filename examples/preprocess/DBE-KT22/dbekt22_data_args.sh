#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/dbekt22_data_args.py \
    --Question_KC_Relationships_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/Question_KC_Relationships.csv" \
    --KCs_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/KCs.csv" \
    --Questions_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/Questions.csv" \
    --Transaction_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/Transaction.csv" \
    --Question_Choices_path "/home/jovyan/work/PICKT/data/DBE-KT22/original/Question_Choices.csv" \
    --data_args_save_path "/home/jovyan/work/PICKT/data/DBE-KT22/preprocessed/data_args.json"