#!/bin/bash

python /home/jovyan/work/PICKT/src/pickt/preprocessor/preprocess_dbekt22/data_args.py \
    --Question_KC_Relationships_path "/home/jovyan/work/PICKT/data/DBE-KT22/Question_KC_Relationships.csv" \
    --KCs_path "/home/jovyan/work/PICKT/data/DBE-KT22/KCs.csv" \
    --Questions_path "/home/jovyan/work/PICKT/data/DBE-KT22/Questions.csv" \
    --Transaction_path "/home/jovyan/work/PICKT/data/DBE-KT22/Transaction.csv" \
    --Question_Choices_path "/home/jovyan/work/PICKT/data/DBE-KT22/Question_Choices.csv" \
    --data_args_save_path "/home/jovyan/work/PICKT/data/DBE-KT22/data_args.json"