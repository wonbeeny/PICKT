# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-06-23

import os
import json

from typing import Any, Optional, Union
from dataclasses import dataclass

from .base_args import BaseArguments
from .training_args import TrainingArguments
from .evaluation_args import EvaluationArguments
from .prediction_args import PredictionArguments

from .model_args import (
    PicktMilktModelArguments,
    SaintMilktModelArguments,
    DktMilktModelArguments,
    GktMilktModelArguments,
    SaktMilktModelArguments,
    AktMilktModelArguments,
    DkvmnMilktModelArguments,
    DTransformerMilktModelArguments,
)
from .data_args import (
    MilktDataArguments
)


@dataclass
class PicktMilktArguments(
    PicktMilktModelArguments, MilktDataArguments, BaseArguments, TrainingArguments, EvaluationArguments, PredictionArguments
):    
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)


@dataclass
class SaintMilktArguments(
    SaintMilktModelArguments, MilktDataArguments, BaseArguments, TrainingArguments, EvaluationArguments, PredictionArguments
):    
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)


@dataclass
class DktMilktArguments(
    DktMilktModelArguments, MilktDataArguments, BaseArguments, TrainingArguments, EvaluationArguments, PredictionArguments
):    
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)


@dataclass
class GktMilktArguments(
    GktMilktModelArguments, MilktDataArguments, BaseArguments, TrainingArguments, EvaluationArguments, PredictionArguments
):    
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)


@dataclass
class SaktMilktArguments(
    SaktMilktModelArguments, MilktDataArguments, BaseArguments, TrainingArguments, EvaluationArguments, PredictionArguments
):    
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)


@dataclass
class AktMilktArguments(
    AktMilktModelArguments, MilktDataArguments, BaseArguments, TrainingArguments, EvaluationArguments, PredictionArguments
):    
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)


@dataclass
class DkvmnMilktArguments(
    DkvmnMilktModelArguments, MilktDataArguments, BaseArguments, TrainingArguments, EvaluationArguments, PredictionArguments
):    
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)


@dataclass
class DTransformerMilktArguments(
    DTransformerMilktModelArguments, MilktDataArguments, BaseArguments, TrainingArguments, EvaluationArguments, PredictionArguments
):    
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        TrainingArguments.__post_init__(self)