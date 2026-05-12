# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-05-21

from dataclasses import dataclass, field
from typing import Optional, Union

from ..utils import pickt_logger


logger = pickt_logger(__name__)

@dataclass
class TrainingArguments:
    """Lightning Trainer Arguments."""

    initializer_range: Optional[float] = field(
        default=0.02,
        metadata={
            "help": "The standard deviation of the truncated_normal_initializer for initializing all weight matrices."
        }
    )
    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "best model 선정 기준이 되는 성능 지표(metric)"
        }
    )
    max_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "학습 epoch 수"
        }
    )
    train_batch_size: Optional[int] = field(
        default=16,
        metadata={
            "help": "학습 mini batch size"
        }
    )
    train_num_workers: Optional[int] = field(
        default=0,
        metadata={
            "help": "학습 시 지정한 수만큼의 별도 프로세스를 생성해 데이터를 병렬로 읽어 메인 프로세스에 전달."
        }
    )
    limit_train_batches : Optional[Union[int, float]] = field(
        default=1.0,
        metadata={
            "help": "학습 시 사용할 배치 수 또는 비율을 제한. 1.0으로 설정하면 전체 학습 데이터 사용."
        }
    )
    optimizer: Optional[str] = field(
        default="adamw",
        metadata={
            "help": "옵티마이저 설정. option: ['sgd', 'adam', 'adamw']"
        }
    )
    lr_schedule: Optional[str] = field(
        default="linear",
        metadata={
            "help": "Learning Rate 스케쥴러 설정. option: ['step', 'cosine', 'linear']"
        }
    )
    learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={
            "help": "Learning Rate 값 설정."
        }
    )
    warmup_steps: Optional[float] = field(
        default=0,
        metadata={
            "help": "Warmup Steps 값 설정."
        }
    )
    accelerator: Optional[str] = field(
        default="gpu",
        metadata={
            "help": "학습에 사용할 하드웨어 유형을 지정. option: ['cpu', 'gpu', 'auto']"
        }
    )
    sync_batchnorm: Optional[bool] = field(
        default=True,
        metadata={
            "help": "다중 GPU에서 배치 정규화 레이어의 통계치를 동기화. 작은 배치 크기를 사용할 때 유용 (예: 배치 크기 8/GPU × 8 GPU = 전체 64). 분산환경(DDP, FSDP 등)에서만 의미가 있음. 단일 장치(cpu, single gpu)에서 효과 없거나 무시됨."
        }
    )
    check_val_every_n_epoch: Optional[int] = field(
        default=1,
        metadata={
            "help": "몇 번의 학습 에포크마다 검증(validation)을 수행할지 지정."
        }
    )
    enable_progress_bar: Optional[bool] = field(
        default=True,
        metadata={
            "help": "학습 진행 표시줄 활성화 여부."
        }
    )
    precision: Optional[str] = field(
        default="32-true",
        metadata={
            "help": "연산 정밀도를 제어하여 메모리 사용량과 성능을 최적화. option: ['16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']"
        }
    )
    strategy: Optional[str] = field(
        default="ddp",
        metadata={
            "help": "분산 학습 전략을 정의. option: ['single_device', 'ddp', 'fsdp', 'deepspeed']"
        }
    )
    logger: Optional[bool] = field(
        default=True,
        metadata={
            "help": "실험 추적을 위한 로거(또는 반복 가능한 로거 모음). True 값은 설치된 경우 기본 텐서보드로거를 사용하고, 그렇지 않은 경우 CSV로거를 사용합니다. False는 로깅을 비활성화합니다. 여러 로거가 제공되면 로컬 파일(체크포인트, 프로파일러 추적 등)이 첫 번째 로거의 log_dir에 저장됩니다."
        }
    )
    gradient_clip_algorithm: Optional[str] = field(
        default="norm",
        metadata={
            "help": "그래디언트 폭주를 방지하기 위해 클리핑 방식을 제어. option: ['value', 'norm']"
        }
    )
    gradient_clip_val: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "그래디언트 폭주를 방지하기 위해 클리핑 임계값 설정."
        }
    )
    deterministic: Optional[bool] = field(
        default=True,
        metadata={
            "help": "CUDA 연산의 결정론적(deterministic) 실행을 보장. True로 설정하면 재현성은 보장되지만 성능이 저하될 수 있음. gpu 연산을 해도 재현성이 유지되기 위해 default=true 로 설정."
        }
    )
    benchmark: Optional[bool] = field(
        default=True,
        metadata={
            "help": "CUDA 연산 최적화를 위해 입력 크기 기반 자동 튜닝을 활성화. True로 설정하면 입력 크기가 고정된 경우 연산 속도가 개선되지만, 메모리 사용량이 증가할 수 있음. deterministic=True 면 자동으로 False 로 설정됨. cpu 사용 시 해당 옵션은 의미가 없음."
        }
    )
    
    def __post_init__(self):
        if (self.strategy == "fsdp") and (self.accelerator != "gpu"):
            raise ValueError("If you use fsdp, must set the `accelerator` to gpu.")
        if (self.strategy == "fsdp") and (self.gradient_clip_algorithm != "value"):
            raise ValueError("If you use fsdp, must set the `gradient_clip_algorithm` to value.")
        if (self.strategy == "fsdp") and (self.gradient_clip_algorithm == "value"):
            logger.info(f"If you want to use fsdp then set `gradient_clip_val`=0.5 for model performance")
        if (self.strategy == "single_device") and (self.accelerator == "auto"):
            logger.info(f"The `accelerator`=auto is the same as the 'gpu' setting.")