# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-05-21

import time
import torch

from overrides import overrides
from abc import ABC, abstractmethod

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from lightning.pytorch import LightningModule, LightningDataModule
from .schedulers import (
    linear_scheduler,    
    cosine_scheduler,
    multistep_scheduler,
)


class BaseModelModule(LightningModule, ABC):
    """
    모델 학습/검증/추론을 위한 Base Model class 정의.
    모델 별 특징에 맞춰 BaseModelModule 상속하여 학습 및 검증 진행.
    반드시 BaseModelModule 상속할 것.
    """
    @overrides
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

        self.optimizer_types = {
            "sgd": SGD,
            "adam": Adam,
            "adamw": AdamW,
        }

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

    @overrides
    def setup(self, stage):
        """
        학습/검증/추론 단계별 초기화 작업을 수행.
        동작 시점: Trainer의 fit(), validate(), test(), predict() 호출 시 각 단계 시작 전 1회 실행.
        """
        pass

    # @overrides
    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        각 학습 배치(batch)에서 순전파(forward), 손실(loss) 계산, 로깅 등을 처리.
        모든 학습 배치마다 반복 실행.
        """
        raise NotImplementedError("`training_step` method must be customized by models when stage='fit'.")

    @overrides
    @abstractmethod
    def on_train_epoch_end(self):
        """
        한 에폭(epoch)의 모든 배치 처리 후, 전체 결과를 집계하거나 추가 작업을 수행.
        각 학습 에폭이 끝날 때 한 번 실행.
        """
        raise NotImplementedError("`on_train_epoch_end` method must be customized by models when stage='fit'].")

    # @overrides
    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        각 검증 배치(batch)에서 모델 추론, 손실 계산 및 메트릭 계산을 수행.
        모든 검증 배치마다 반복 실행.
        """
        raise NotImplementedError("`validation_step` method must be customized by models when stage in ['fit', 'validation'].")

    @overrides
    @abstractmethod
    def on_validation_epoch_end(self):
        """
        한 에폭의 모든 검증 배치 처리 후 전체 결과를 집계.
        각 검증 에폭이 끝날 때 한 번 실행.
        """
        raise NotImplementedError("`on_validation_epoch_end` method must be customized by models when stage in ['fit', 'validation'].")

    # @overrides
    def test_step(self, batch, batch_idx):
        """
        각 테스트 배치(batch)에서 모델 추론, 손실 계산 및 메트릭 계산을 수행.
        모든 테스트 배치마다 반복 실행.
        """
        pass

    @overrides
    def on_test_epoch_end(self):
        """
        한 에폭의 모든 테스트 배치 처리 후 전체 결과를 집계.
        각 테스트 에폭이 끝날 때 한 번 실행.
        """
        pass

    # @overrides
    def predict_step(self, batch, batch_idx):
        """
        각 추론 배치(batch)에서 모델 추론 계산을 수행.
        모든 추론 배치마다 반복 실행.
        """
        pass

    @overrides
    def on_fit_end(self):
        """
        학습(fit)이 완전히 종료된 직후 호출되는 훅(hook)으로 모든 학습 프로세스(에폭, 검증, 체크포인트 저장 등)가 끝나고 최종 정리 작업을 수행하는 데 사용.
        Trainer.fit()이 완전히 종료된 후 실행.(정상 종료 또는 조기 중단 모두 포함)
        """
        pass
    
    @overrides
    def configure_optimizers(self):
        """
        옵티마이저(optimizer)와 학습률 스케줄러(LR scheduler) 정의.
        복수 옵티마이저 및 커스텀 스케줄러 설정도 가능.
        동작 시점: Trainer가 학습/검증/테스트를 시작할 때 1회 호출되어 옵티마이저와 스케줄러를 생성.
        
        BaseModelModule 상속 시 configure_optimizers 정의하지 않아도 됨. 
        다만, 존재하지 않는 방법론을 추가할 필요가 있을 시 수정 보완 필요.
        """
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,    # 실제 스케줄러 객체
                "interval": "step",    # 배치마다 업데이트 (epoch 으로 변경 가능 but 변경하지 말 것.)
                "name": "learning_rate",    # 로깅용 이름 (예: TensorBoard에서 학습률 추적 시 사용)
            }
        }

    def _get_optimizer(self):
        """
        옵티마이저(optimizer) 정의.
        권장: AdamW.
        """
        optimizer_method = self.config.optimizer.lower()

        if optimizer_method not in self.optimizer_types:
            raise ValueError(f"Unknown optimizer method={optimizer_method}")

        kwargs = {"lr": self.config.learning_rate}
        kwargs["params"] = self.model.parameters()
        optimizer = self.optimizer_types[self.config.optimizer](**kwargs)

        return optimizer

    def _get_lr_scheduler(self, optimizer):
        """
        학습률 스케줄러(LR scheduler) 정의.
        권장: linear
        """
        lr_schedule_method = self.config.lr_schedule
        lr_schedule_params = {"warmup_steps": self.config.warmup_steps}

        if lr_schedule_method is None:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1)
        elif lr_schedule_method == "step":
            scheduler = multistep_scheduler(optimizer, **lr_schedule_params)
        elif lr_schedule_method == "cosine":
            total_samples = self.config.max_epochs * self.config.num_samples_per_epoch
            total_batch_size = self.config.train_batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = cosine_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        elif lr_schedule_method == "linear":
            total_samples = self.config.max_epochs * self.config.num_samples_per_epoch
            total_batch_size = self.config.train_batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = linear_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        else:
            raise ValueError(f"Unknown lr_schedule_method={lr_schedule_method}")

        return scheduler


class BaseDataModule(LightningDataModule, ABC):
    """
    모델 학습/검증/추론을 위한 Base Data class 정의.
    사용하는 데이터셋 특징에 맞춰 BaseDataModule 상속하여 데이터셋 정의 진행.
    반드시 BaseDataModule 상속할 것.
    """
    @overrides
    @abstractmethod
    def __init__(self):
        super().__init__()

    @overrides
    def setup(self, stage):
        """
        학습/검증/테스트/추론 단계별 데이터 정의.
        
        BaseDataModule 상속하여 개발 시 해당 setup format 을 반드시 따를 것.
        """
        if stage in (None, "fit"):    # running train & validate
            self.train_loader = self._get_train_loader()
            self.valid_loader = self._get_valid_test_loaders(stage="validate")
        elif stage in ("validate", "test"):    # running only validate or test
            self.valid_loader = self._get_valid_test_loaders(stage=stage)
        elif stage == "predict":    # running only predict
            self.pred_loader = self._get_pred_loaders()
        else:
            raise ValueError(f"Unknown stage={stage}")
        

    @abstractmethod
    def _get_train_loader(self):
        raise NotImplementedError("`_get_train_loader` method must be customized by datasets when stage='fit'.")

    @abstractmethod
    def _get_valid_test_loaders(self, stage):
        raise NotImplementedError("`_get_valid_test_loaders` method must be customized by datasets when stage in ('fit', 'validate', 'test').")
    
    def _get_pred_loaders(self):
        raise NotImplementedError("`_get_pred_loaders` method must be customized by datasets when stage='pred'.")

    @overrides
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        데이터 배치를 타겟 디바이스(예: GPU)로 이동시키는 로직을 직접 제어하는 method.
        training_step, validation_step 등이 실행되기 직전에 호출됨.
        
        데이터셋의 형태에 따라 필요 시 수정 보완하여 사용.
        """
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        return batch
    
    @overrides
    def train_dataloader(self):
        """
        수정하지 말 것.
        BaseDataModule 상속하여 개발 시 setup method 만 수정하면 됨.
        """
        return self.train_loader

    @overrides
    def val_dataloader(self):
        """
        수정하지 말 것.
        BaseDataModule 상속하여 개발 시 setup method 만 수정하면 됨.
        """
        return self.valid_loader

    @overrides
    def test_dataloader(self):
        """
        수정하지 말 것.
        BaseDataModule 상속하여 개발 시 setup method 만 수정하면 됨.
        """
        return self.valid_loader

    @overrides
    def predict_dataloader(self):
        """
        수정하지 말 것.
        BaseDataModule 상속하여 개발 시 setup method 만 수정하면 됨.
        """
        return self.pred_loader