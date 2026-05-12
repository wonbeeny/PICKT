# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-05-22

import os
import json

from typing import Any, Optional, Union
from dataclasses import asdict, dataclass, field, fields


@dataclass
class BaseArguments:
    """Base Arguments."""

    pipeline: Optional[str] = field(
        default=None,
        metadata={
            "help": "모델 학습, 검증, 추론 중 선택. option: ['train', 'valid', 'test', 'pred']"
        }
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "모델 이름."
        }
    )
    data_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "데이터셋 이름."
        }
    )
    model_path: Union[str, os.PathLike] = field(
        default=None,
        metadata={
            "help": "The model weight path for valid/test/pred."
        }
    )
    data_args_path: Union[str, os.PathLike] = field(
        default=None,
        metadata={
            "help": "데이터셋 arguments 정보인 dara_args.json 파일 path."
        },
    )
    km_data_path: Union[str, os.PathLike] = field(
        default=None,
        metadata={
            "help": "전처리된 지식맵 데이터 정보인 km_data.json 파일 path."
        },
    )
    train_dataset_path: Union[str, os.PathLike] = field(
        default=None,
        metadata={
            "help": "train_dataset path."
        },
    )
    valid_dataset_path: Union[str, os.PathLike] = field(
        default=None,
        metadata={
            "help": "valid_dataset path."
        },
    )
    test_dataset_path: Union[str, os.PathLike] = field(
        default=None,
        metadata={
            "help": "test_dataset path."
        },
    )
    pred_dataset_path: Union[str, os.PathLike] = field(
        default=None,
        metadata={
            "help": "pred_dataset path."
        },
    )
    output_dir: Union[str, os.PathLike] = field(
        default="./output_dir",
        metadata={
            "help": "The output directory."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "한 학생이 푼 문항을 어느 정도의 길이로 구성할지 설정."
        }
    )
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "데이터 섞기, 가중치 초기화, 데이터 증강 등에서 발생하는 모든 무작위성 통제."
        },
    )
    threshold: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "모델의 Output 에서 학생의 정답 오답을 분류하기 위한 임계값."
        }
    )
    
    def __post_init__(self):
        self.pipeline = self.pipeline.lower()
        self.data_name = self.data_name.lower()
        self.model_name = self.model_name.lower()
        
        if self.pipeline is None:
            raise ValueError("Please provide `pipeline`.")
        if self.model_name is None:
            raise ValueError("Please provide `model_name`.")
        if self.data_name is None:
            raise ValueError("Please provide `data_name`.")
        if self.pipeline == "train" and self.data_args_path is None:
            raise ValueError("Please provide `data_args_path`.")
        if self.pipeline == "train" and self.train_dataset_path is None:
            raise ValueError("Please provide `train_dataset_path`.")
        if self.pipeline == "train" and self.max_seq_length is None:
            raise ValueError("Please provide `max_seq_length`.")
        if self.pipeline == "valid" and self.valid_dataset_path is None:
            raise ValueError("Please provide `valid_dataset_path`.")
        if self.pipeline == "pred" and self.pred_dataset_path is None:
            raise ValueError("Please provide `pred_dataset_path`.")
        if self.model_name == "pickt" and self.km_data_path is None:
            raise ValueError("Please provide `km_data_path`.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update_from_dict(self, **kwargs):
        for key, value in kwargs.items():
            if key in {f.name for f in fields(self)}:
                setattr(self, key, value)
    
    def save_config(self, save_path: Union[str, os.PathLike]):
        with open(save_path, 'w', encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def from_kwargs(cls, **kwargs):
        allowed = {f.name for f in fields(cls)}
        return cls(**{k:v for k,v in kwargs.items() if k in allowed})

    def __getitem__(self, key):
        return getattr(self, key)