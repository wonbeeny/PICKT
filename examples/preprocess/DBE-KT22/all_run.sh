#!/bin/bash

sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/data_args.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/ext2question_embed.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/ext2concept_embed.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/dim_reduction.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/km_data.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/preprocess.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/DBE-KT22/split_datasets.sh 