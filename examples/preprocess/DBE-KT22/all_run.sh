#!/bin/bash

sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/dbekt22_data_args.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/ext2question_embed.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/ext2concept_embed.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/dim_reduction.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/dbekt22_km_data.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/dbekt22_preprocess.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/dbekt22_split_datasets.sh 