#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python /home/jovyan/work/PICKT/src/pickt/main.py \
    --config_path "/home/jovyan/work/PICKT/examples/config/Online/pickt_train_config.yaml"