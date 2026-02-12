# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-04-15
# Model Output Form :
# - Knowledge Tracing Model Output 은 항상 logits & predictions 가 존재해야 됨.
# - 그 외의 값은 Model 에 따라 Customizing 하는 것으로 Model Output 형태 통일.

import torch

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.modeling_outputs import ModelOutput


@dataclass
class KTModelOutput(ModelOutput):
    """
    Knowledge Tracing Model Outputs. encoder layer 의 attention_score 추출 필요 for reasoning.
    attention score 추출할 수 있도록 추가해줄 예정
    
    Args:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length, 1)`):
            Regression scores.
        predictions (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length, 1)`):
            Probability of correct answer after sigmoid(logits).
        encoder_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length, config.hidden_size)`):
            Hidden-states of the encoder model at the output of last encoder layer
        decoder_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length, config.hidden_size)`):
            Hidden-states of the decoder model at the output of last decoder layer
    """
    logits: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.FloatTensor] = None
    encoder_output: Optional[Tuple[torch.FloatTensor]] = None
    decoder_output: Optional[Tuple[torch.FloatTensor]] = None