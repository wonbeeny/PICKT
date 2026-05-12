# coding : utf-8
# edit : 
# - author : wblee
# - comment : worker example script. Please refer to this script.
# - date : 2025-06-24


import os
import argparse

from pickt import (
    Worker,
    load_data,
)
from pickt.utils import pickt_logger


logger = pickt_logger(__name__)

parser = argparse.ArgumentParser(description="Train/Eval/Test/Pred using pickt.Worker")

parser.add_argument(
    "--config_path",
    type = str,
    help = "Insert your configure file(yaml) path."
)

args = parser.parse_args()
logger.info(args)


def main(args):   
    config = load_data(args.config_path)
    if config.model_name in ["pickt", "gkt"]:
        km_data = load_data(config.km_data_path)

    if config.pipeline == "train":
        data_args = load_data(config.data_args_path)
        
        train_datasets = load_data(config.train_dataset_path)
        valid_datasets = load_data(config.valid_dataset_path)
        if config.model_name in ["pickt","gkt"]:
            train_datasets["km_data"] = km_data
            valid_datasets["km_data"] = km_data
        
        worker = Worker(config, data_args=data_args, train_datasets=train_datasets, valid_datasets=valid_datasets)
        worker.train()
        
    elif config.pipeline == "valid":
        valid_datasets = load_data(config.valid_dataset_path)
        if config.model_name in ["pickt", "gkt"]:
            valid_datasets["km_data"] = km_data

        worker = Worker(config, valid_datasets=valid_datasets)
        worker.valid()
        
    elif config.pipeline == "test":
        test_datasets = load_data(config.test_dataset_path)
        if config.model_name in ["pickt", "gkt"]:
            test_datasets["km_data"] = km_data

        worker = Worker(config, test_datasets=test_datasets)
        worker.test()

    # pred 시에는 싱글 GPU or CPU 사용 권장.
    # pred 시 멀티 GPU 사용하면 Output 결과가 입력 데이터의 순서와 다를 수 있음.
    elif config.pipeline == "pred":
        pred_datasets = load_data(config.pred_dataset_path)
        if config.model_name in ["pickt", "gkt"]:
            pred_datasets["km_data"] = km_data

        worker = Worker(config, pred_datasets=pred_datasets)
        predictions = worker.pred()

if __name__ == "__main__":
    main(args)