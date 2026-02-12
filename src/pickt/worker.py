# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-08-14

import os
import time
import torch

from typing import Union, Dict

from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer as lightning_Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.strategies import (
    SingleDeviceStrategy, 
    # DataParallelStrategy,   v2.x 에서는 제거되었다고 함
    DDPStrategy,
    FSDPStrategy,
    DeepSpeedStrategy
)

from .utils import (
    pickt_logger, 
    load_data, 
    save_json, 
    create_folder_if_not_exists,
    measure_execution_time,
    PredictionCollector,
)
from .hparams import (
    PicktMilktArguments,
    SaintMilktArguments,
    DktMilktArguments,
    GktMilktArguments,
    SaktMilktArguments,
    AktMilktArguments,
    DkvmnMilktArguments,
    DTransformerMilktArguments,
)
from .models import (
    PicktMilktModel,
    SaintMilktModel,
    DktMilktModel,
    GktMilktModel,
    SaktMilktModel,
    AktMilktModel,
    DkvmnMilktModel,
    DTransformerMilktModel,
    PicktDbekt22Model,
)
from .lightning_modules import (
    ModelModule, 
    DataModule
)


logger = pickt_logger(__name__)

MODULE_MAP = {
    "pickt-milkt": [PicktMilktArguments, PicktMilktModel, ModelModule, DataModule],
    "saint-milkt": [SaintMilktArguments, SaintMilktModel, ModelModule, DataModule],
    "dkt-milkt": [DktMilktArguments, DktMilktModel, ModelModule, DataModule],
    "gkt-milkt": [GktMilktArguments, GktMilktModel, ModelModule, DataModule],
    "sakt-milkt": [SaktMilktArguments, SaktMilktModel, ModelModule, DataModule],
    "akt-milkt": [AktMilktArguments, AktMilktModel, ModelModule, DataModule],
    "dkvmn-milkt": [DkvmnMilktArguments, DkvmnMilktModel, ModelModule, DataModule],
    "dtransformer-milkt": [DTransformerMilktArguments, DTransformerMilktModel, ModelModule, DataModule],
    "pickt-dbekt22": [PicktMilktArguments, PicktDbekt22Model, ModelModule, DataModule],
    "saint-dbekt22": [SaintMilktArguments, SaintMilktModel, ModelModule, DataModule],
    "dkt-dbekt22": [DktMilktArguments, DktMilktModel, ModelModule, DataModule],
    "gkt-dbekt22": [GktMilktArguments, GktMilktModel, ModelModule, DataModule],
    "sakt-dbekt22": [SaktMilktArguments, SaktMilktModel, ModelModule, DataModule],
    "akt-dbekt22": [AktMilktArguments, AktMilktModel, ModelModule, DataModule],
    "dkvmn-dbekt22": [DkvmnMilktArguments, DkvmnMilktModel, ModelModule, DataModule],
    "dtransformer-dbekt22": [DTransformerMilktArguments, DTransformerMilktModel, ModelModule, DataModule],
}

class Worker:
    def __init__(
        self, 
        config,
        data_args: Dict[str, Union[int, dict]] = None,
        train_datasets: Dict[str, dict] = None,
        valid_datasets: Dict[str, dict] = None,
        test_datasets: Dict[str, dict] = None,
        pred_datasets: Dict[str, dict] = None,
    ):
        
        seed_everything(config.seed, workers=True)
        
        self.data_args = data_args
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.test_datasets = test_datasets
        self.pred_datasets = pred_datasets
        
        args_cls, model_cls, model_module_cls, data_module_cls = self._get_module(config)
        if config.pipeline == "train":
            self.config = args_cls.from_kwargs(**config)
            self._get_train_dataset_len()
            self._get_config()
        else:
            output_dir = os.path.dirname(os.path.dirname(config.model_path))
            _config = load_data(os.path.join(output_dir , "config.json"))
            _config.update(config)
            self.config = args_cls(**_config)
        logger.info('Loaded configure.')
            
        self.strategy = self._get_strategy()
        self.callbacks = self._get_callbacks()
        self.loggers = self._get_loggers()
        
        self.set(model_cls, model_module_cls, data_module_cls)


    def _get_module(self, config):
        try:
            # import pdb; pdb.set_trace()
            module = MODULE_MAP[f"{config.model_name}-{config.data_name}"]
            args_cls, model_cls, model_module_cls, data_module_cls = module
            logger.info(f"Prepare module about {config.model_name} model and {config.data_name} data.")
            return args_cls, model_cls, model_module_cls, data_module_cls
        except ValueError as e:
            logger.error(f"Unknown MODULE_MAP. Please check name of model:{config.model_name} and data:{config.data_name}")

    def _get_train_dataset_len(self):
        max_seq_length = self.config.max_seq_length
        question_ids = self.train_datasets["encoder_inputs"]["question_ids"]

        train_dataset_len = 0
        for solved_list in question_ids:
            quotient, remainder = divmod(len(solved_list), max_seq_length)
            if remainder == 0:
                train_dataset_len += quotient
            else:
                train_dataset_len += quotient+1

        self.train_dataset_len = train_dataset_len
    
    def _get_config(self):
        if not isinstance(self.data_args, dict):
            raise TypeError(f"Expected data_args to be dict, but got {type(self.data_args).__name__}")
        self.config.update_from_dict(**self.data_args)
        
        ## Reset batch size for DDP
        num_devices = torch.cuda.device_count()
        
        if self.config.train_batch_size // num_devices==0:
            new_batch_size=1
        else:
            new_batch_size = self.config.train_batch_size // num_devices
        self.config.train_batch_size = new_batch_size

        if num_devices > 1:
            self.config.num_samples_per_epoch = self.train_dataset_len*num_devices
        else:
            self.config.num_samples_per_epoch = self.train_dataset_len
    
    def set(self, model_cls, model_module_cls, data_module_cls):
        model = model_cls(self.config)
        # logger.info(model)
        self.model_module = model_module_cls(self.config, model)
        logger.info('Loaded model module.')
        
        if self.config.pipeline == "train":
            self.data_module = data_module_cls(self.config, train_datasets=self.train_datasets, valid_datasets=self.valid_datasets)
        elif self.config.pipeline == "valid":
            self.data_module = data_module_cls(self.config, valid_datasets=self.valid_datasets)
        elif self.config.pipeline == "test":
            self.data_module = data_module_cls(self.config, test_datasets=self.test_datasets)
        elif self.config.pipeline == "pred":
            self.data_module = data_module_cls(self.config, pred_datasets=self.pred_datasets)
        logger.info('Loaded data module.')
        
        self._set_trainer(pipeline=self.config.pipeline)
        logger.info('Loaded lightning Trainer.')

    @measure_execution_time
    def train(self):
        create_folder_if_not_exists(self.config.output_dir)
        self.config.save_config(os.path.join(self.config.output_dir, "config.json"))
        
        logger.info('Training ...')
        self.trainer.fit(self.model_module, datamodule=self.data_module)
        logger.info('Finish ...')

    @measure_execution_time
    def valid(self):
        logger.info('Validate ...')
        self.trainer.validate(model=self.model_module, datamodule=self.data_module, ckpt_path=self.config.model_path)
        logger.info('Finish ...')

    @measure_execution_time
    def test(self):
        logger.info('Test ...')
        self.trainer.test(model=self.model_module, datamodule=self.data_module, ckpt_path=self.config.model_path)
        logger.info('Finish ...')

    # pred 시에는 싱글 GPU or CPU 사용 권장.
    # pred 시 멀티 GPU 사용하면 Output 결과가 입력 데이터의 순서와 다를 수 있음.
    # 이 부분에 대한 고도화 필요.
    @measure_execution_time
    def pred(self):
        logger.info('Predict ...')
        self.trainer.predict(model=self.model_module, datamodule=self.data_module, ckpt_path=self.config.model_path)
        final_output = self.callbacks[0].final_output
        
        create_folder_if_not_exists(self.config.output_dir)
        save_json(self.config.output_dir, "predict_outputs.json", final_output)
        logger.info('Finish ...')

        return final_output

    
    def _set_trainer(self, pipeline):
        if pipeline == "train":
            self.trainer = lightning_Trainer(
                max_epochs=self.config.max_epochs,
                accelerator=self.config.accelerator,
                precision=self.config.precision,
                check_val_every_n_epoch=self.config.check_val_every_n_epoch,
                limit_train_batches=self.config.limit_train_batches,
                limit_val_batches=self.config.limit_val_batches,
                gradient_clip_val=self.config.gradient_clip_val,
                gradient_clip_algorithm=self.config.gradient_clip_algorithm,
                sync_batchnorm=self.config.sync_batchnorm,
                enable_progress_bar=self.config.enable_progress_bar,
                benchmark=self.config.benchmark,
                deterministic=self.config.deterministic,
                devices=torch.cuda.device_count(),
                strategy=self.strategy,
                callbacks=self.callbacks,
                logger=self.loggers,
            )
        elif pipeline in ("valid", "test"):
            self.trainer = lightning_Trainer(
                accelerator=self.config.accelerator,
                precision=self.config.precision,
                limit_val_batches=self.config.limit_val_batches,
                limit_test_batches=self.config.limit_val_batches,
                sync_batchnorm=self.config.sync_batchnorm,
                enable_progress_bar=self.config.enable_progress_bar,
                benchmark=self.config.benchmark,
                deterministic=self.config.deterministic,
                devices=torch.cuda.device_count(),
                strategy=self.strategy,
                callbacks=self.callbacks,
                logger=self.loggers,
            )
        elif pipeline == "pred":
            self.trainer = lightning_Trainer(
                accelerator=self.config.accelerator,
                precision=self.config.precision,
                limit_predict_batches=self.config.limit_predict_batches,
                sync_batchnorm=self.config.sync_batchnorm,
                enable_progress_bar=self.config.enable_progress_bar,
                benchmark=self.config.benchmark,
                deterministic=self.config.deterministic,
                devices=torch.cuda.device_count(),
                strategy=self.strategy,
                callbacks=self.callbacks,
                logger=self.loggers,
            )

    def _get_strategy(self):
        if self.config.strategy == "single_device":
            if self.config.accelerator in ("gpu", "auto"):
                strategy = SingleDeviceStrategy(device="cuda:0")
            elif self.config.accelerator == "cpu":
                strategy = SingleDeviceStrategy()
        elif self.config.strategy == "ddp":
            strategy = DDPStrategy(find_unused_parameters=True)
        elif self.config.strategy == "fsdp":
            strategy = FSDPStrategy()
        elif self.config.strategy == "deepspeed":
            strategy = DeepSpeedStrategy()
        else:
            strategy = "auto"
        
        return strategy

    def _get_callbacks(self):
        callbacks = list()

        if self.config.pipeline == "train":
            callback = ModelCheckpoint(
                dirpath=os.path.join(self.config.output_dir, "checkpoints"),
                filename=f"{self.config.model_name}-{self.config.data_name}-{{epoch:02d}}-{{{self.config.metric_for_best_model}:.4f}}",
                save_top_k=3,
                save_last=False,
                monitor=self.config.metric_for_best_model,    # 체크포인트를 저장할 때 기준이 되는 metric 이름
                mode="max",    # monitor로 지정한 metric 값이 클수록 더 좋은 모델로 간주
            )
            callback.FILE_EXTENSION = ".pt"
            callbacks.append(callback)
        elif self.config.pipeline == "pred":
            collector = PredictionCollector()
            callbacks.append(collector)            
        
        return callbacks

    def _get_loggers(self):
        loggers = list()

        if self.config.pipeline in ("train"):
            tensorboard_logger = TensorBoardLogger(
                os.path.join(self.config.output_dir, "tensorboard_logs"),
                name="", 
                version="", 
                default_hp_metric=False
            )
            loggers.append(tensorboard_logger)
        if self.config.pipeline in ("train", "valid", "test"):
            csv_logger = CSVLogger(
                os.path.join(self.config.output_dir, "csv_logs"),
                name="",
                version=""
            )
            loggers.append(csv_logger)
        
        return loggers