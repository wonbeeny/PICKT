# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-08-14

import time

from typing import Dict
from overrides import overrides
from torch.utils.data.dataloader import DataLoader

from .base_module import BaseDataModule

from ..utils import pickt_logger
from ..preprocessor.milkt_dataset import MilkTDataset
from ..preprocessor.dbekt22_dataset import Dbekt22Dataset


logger = pickt_logger(__name__)

# Key 는 소문자여야 됨
DATA2CLS = {
    "milkt": MilkTDataset,
    "dbekt22": Dbekt22Dataset,
}

class DataModule(BaseDataModule):
    @overrides
    def __init__(
        self, 
        config,
        train_datasets: Dict[str, dict] = None,
        valid_datasets: Dict[str, dict] = None,
        test_datasets: Dict[str, dict] = None,
        pred_datasets: Dict[str, dict] = None,
    ):
        super().__init__()
        self.config = config
        
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.test_datasets = test_datasets
        self.pred_datasets = pred_datasets

        try:
            self.data_cls = DATA2CLS[self.config.data_name]
        except ValueError as e:
            logger.error(f"Unknown data name={self.config.data_name}")
            

    @overrides
    def _get_train_loader(self):
        start_time = time.time()

        dataset = self._get_dataset(self.train_datasets)
        data_loader = self._convert_dataloader(dataset, "train")

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time for loading training data: {elapsed_time:.2f} seconds")

        return data_loader

    @overrides
    def _get_valid_test_loaders(self, stage):
        start_time = time.time()

        if stage == "validate":
            dataset = self._get_dataset(self.valid_datasets)
        elif stage == "test":
            dataset = self._get_dataset(self.test_datasets)
        data_loader = self._convert_dataloader(dataset, stage)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time for loading {stage} data: {elapsed_time:.2f} seconds")
        
        return data_loader

    @overrides
    def _get_pred_loaders(self):
        start_time = time.time()

        dataset = self._get_dataset(self.pred_datasets)
        data_loader = self._convert_dataloader(dataset, "pred")

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time for loading predict data: {elapsed_time:.2f} seconds")

        return data_loader


    def _get_dataset(self, datasets):
        if self.config.data_name == "milkt":
            if self.config.model_name in ["pickt", "gkt"]:
                dataset = self.data_cls(
                    self.config,
                    encoder_inputs = datasets["encoder_inputs"],
                    decoder_inputs = datasets["decoder_inputs"],
                    km_data = datasets["km_data"],
                )
            elif self.config.model_name in ["saint", "dkt", "sakt", "akt", "dkvmn", "dtransformer"]:
                dataset = self.data_cls(
                    self.config,
                    encoder_inputs = datasets["encoder_inputs"],
                    decoder_inputs = datasets["decoder_inputs"],
                )
        elif self.config.data_name == "dbekt22":
            # import pdb; pdb.set_trace()
            if self.config.model_name in ["pickt", "gkt"]:
                dataset = self.data_cls(
                    self.config,
                    encoder_inputs = datasets["encoder_inputs"],
                    decoder_inputs = datasets["decoder_inputs"],
                    km_data = datasets["km_data"],
                )
            elif self.config.model_name in ["saint", "dkt", "sakt", "akt", "dkvmn", "dtransformer"]:
                dataset = self.data_cls(
                    self.config,
                    encoder_inputs = datasets["encoder_inputs"],
                    decoder_inputs = datasets["decoder_inputs"],
                )
        
        return dataset
    
    def _convert_dataloader(self, dataset, pipeline):
        shuffle = True if pipeline == "train" else False    # 학습 시에만 데이터셋을 섞어줌
        if pipeline == "train":
            batch_size = self.config.train_batch_size
            num_workers = self.config.train_num_workers
        elif pipeline in ("validate", "test"):
            batch_size = self.config.valid_batch_size
            num_workers = self.config.valid_num_workers
        elif pipeline == "pred":
            batch_size = self.config.pred_batch_size
            num_workers = self.config.pred_num_workers
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        return data_loader