#!/bin/bash

sh /home/jovyan/work/PICKT/examples/preprocess/Online/data_args.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/Online/ext2question_embed.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/Online/ext2concept_embed.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/Online/dim_reduction.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/Online/km_data.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/Online/preprocess.sh 
sh /home/jovyan/work/PICKT/examples/preprocess/Online/split_datasets.sh 