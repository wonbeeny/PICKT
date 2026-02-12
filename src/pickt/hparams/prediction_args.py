# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-05-21

from typing import Any, Optional, Union
from dataclasses import asdict, dataclass, field


@dataclass
class PredictionArguments:
    """Lightning Prediction Arguments."""

    pred_batch_size: Optional[int] = field(
        default=16,
        metadata={
            "help": "추론 mini batch size"
        }
    )
    pred_num_workers: Optional[int] = field(
        default=0,
        metadata={
            "help": "추론 시 지정한 수만큼의 별도 프로세스를 생성해 데이터를 병렬로 읽어 메인 프로세스에 전달."
        }
    )
    limit_predict_batches : Optional[Union[int, float]] = field(
        default=1.0,
        metadata={
            "help": "추론 시 사용할 배치 수 또는 비율을 제한. 1.0으로 설정하면 전체 추론 데이터 사용."
        }
    )
    
    def __post_init__(self):
        pass