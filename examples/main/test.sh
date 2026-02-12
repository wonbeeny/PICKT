#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python ./PICKT/src/pickt/main.py \
    --config_path "./PICKT/examples/config/pickt_test_config.yaml"