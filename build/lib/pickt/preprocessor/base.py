# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-04-29

from overrides import overrides
from abc import ABC, abstractmethod

from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset, ABC):
    """
    각 데이터셋에 맞게 정의하는 `Dataset` 클래스의 base class.
    아래 method 들을 반드시 포함시킬 것.
    """
    @overrides
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("`__init__` method must be customized by datasets.")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("`__len__` method must be customized by datasets.")

    @overrides
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("`__getitem__` method must be customized by datasets.")