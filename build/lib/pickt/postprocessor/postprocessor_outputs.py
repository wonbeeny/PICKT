# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-04-29

import torch

from dataclasses import dataclass
from typing import Optional


@dataclass
class Universal_Output:
    """
    Knowledge Tracing 모델의 최종 성능 지표 출력을 위한 데이터 클래스.
    Knowledge Tracing 모델의 보편적인 성능 지표만 반환.
    
    Args:
        acc_wrong (float, optional):
            학생이 틀린 문제에 대한 정확도 for imbalance datasets
            - confusion_matrix 에서 [0, 0] 위치의 값을 첫 번째 행의 합으로 나눈 값.
            - [0, 0] 위치의 값: 실제로도, 예측도 학생이 틀린 문제인 개수.
            - 첫 번째 행의 합: 실제로 학생이 틀린 문제의 전체 개수.
        acc_correct (float, optional):
            학생이 맞춘 문제에 대한 정확도 for imbalance datasets
            - confusion_matrix 에서 [1, 1] 위치의 값을 두 번째 행의 합으로 나눈 값.
            - [1, 1] 위치의 값: 실제로도, 예측도 학생이 맞춘 문제인 개수.
            - 두 번째 행의 합: 실제로 학생이 맞춘 문제의 전체 개수.
        acc_macro (float, optional):
            매크로 정확도(Macro Accuracy).
            - 모든 클래스별로 분류 정확도를 독립적으로 계산한 후, 그 평균을 구한 값.
            - 각 클래스의 샘플 수와 상관없이 모든 클래스를 동일하게 취급.
            - 각 클래스별로 정확도를 계산한 뒤, 그 결과를 평균내어 산출.
        acc_micro (float, optional):
            마이크로 정확도(Micro Accuracy).
            - 전체 샘플을 기준으로 한 분류 정확도.
            - 각 샘플이 속한 클래스와 관계없이 모든 샘플에 동일한 중요도를 부여.
            - 전체 정답 예측 수를 전체 샘플 수로 나누어 계산.
        auc_macro (float, optional):
            매크로 AUC(Area Under the Curve).
            - 모든 클래스별로 ROC AUC 점수를 독립적으로 계산한 후, 그 평균을 구한 값.
            - 각 클래스에 대해 one-vs-rest 방식으로 AUC를 계산한 뒤, 그 결과를 평균.
            - 각 클래스에 동일한 가중치를 부여.
            - 이진분류에서 auc_micro 와 동일한 값을 가지기 때문에 drop
        auc_micro (float, optional):
            마이크로 AUC(Area Under the Curve).
            - 모든 샘플과 클래스를 함께 고려하여 계산한 전체 ROC AUC 점수.
            - 모든 클래스와 샘플의 기여도를 합산한 후 AUC를 계산.
            - 클래스와 관계없이 모든 예측에 동일한 가중치를 부여.
    """
    acc_wrong: Optional[float] = None
    acc_correct: Optional[float] = None
    acc_macro: Optional[float] = None
    acc_micro: Optional[float] = None
    # auc_macro: Optional[float] = None
    auc_micro: Optional[float] = None

@dataclass
class Specific_Output:
    """
    Knowledge Tracing 모델의 최종 성능 지표 출력을 위한 데이터 클래스.
    Knowledge Tracing 모델의 성능을 면밀히 측정하기 위해 특정한 성능 지표만 반환.
    """