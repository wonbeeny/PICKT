# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-06-23

import os
import torch

import numpy as np

from overrides import overrides
from typing import Optional, Union, Dict

from .base import BaseDataset
from ..utils import pickt_logger, load_data


logger = pickt_logger(__name__)

class MilkTDataset(BaseDataset):
    @overrides
    def __init__(
        self,
        config,
        encoder_inputs: Dict[str, list] = None,
        decoder_inputs: Dict[str, list] = None,
        km_data: Dict[str, dict] = None,
    ):
        self.config = config
        max_seq_length = self.config.max_seq_length
        
        # 1. 각 학생의 응답을 max_seq_length 로 분할하고, 부족한 부분은 패딩값으로 채운 되 torch tensor 로 변환
        ### encoder input 변환
        self.encoder_inputs = dict()
        for data_type, student_seq in encoder_inputs.items():
            data_name = data_type.split('_')[0] + "2id"
            self.encoder_inputs[data_type] = self._split_and_pad_sequence_data(student_seq, max_seq_length, self.config[data_name]["pad_id"])

        ### decoder input 변환: 맨 앞에 start_id 추가 및 맨 마지막 index trim
        self.decoder_inputs = dict()
        for data_type, student_seq in decoder_inputs.items():
            data_name = data_type.split('_')[0] + "2id"
            batch_tensor = self._split_and_pad_sequence_data(student_seq, max_seq_length, self.config[data_name]["pad_id"])
            if data_type == "response_ids":
                self.decoder_inputs["labels"] = batch_tensor
                
            if self.config.model_name in ["pickt", "saint"]:
                self.decoder_inputs[data_type] = self._prepend_start_id_and_trim(batch_tensor, self.config[data_name]["start_id"])
            else:
                self.decoder_inputs[data_type] = batch_tensor

        if self.config.model_name in ["pickt","gkt"]:
            # 2. 이종 그래프(Heterogeneous Graph)를 구성을 위한 데이터 정의
            ### 개념과 개념 사이의 선후행 관계 정보
            source_concept_nodes = [edge['source'] for edge in km_data["concept2concept_edge"]]
            target_concept_nodes = [edge['target'] for edge in km_data["concept2concept_edge"]]
            self.concept2concept_edge = torch.tensor([source_concept_nodes, target_concept_nodes], dtype=torch.long)
    
            ### 개념과 문항 사이의 포함 관계 정보
            parent_concept_nodes = [edge['concept'] for edge in km_data["concept2question_edge"]]
            child_question_nodes = [edge['question'] for edge in km_data["concept2question_edge"]]
            self.concept2question_edge = torch.tensor([parent_concept_nodes, child_question_nodes], dtype=torch.long)
    
            ### text 정보: text embedding & 차원 축소한 vector
            self.concept_rel_embeds = torch.tensor(km_data["concept_embeds"])
            self.question_rel_embeds = torch.tensor(km_data["question_embeds"])
    
            # 3. 상삼각행렬 적용
            self.attention_mask = torch.triu(
                torch.ones(max_seq_length, max_seq_length), 
                diagonal=1
            ).to(dtype=torch.bool)

    @overrides
    def __len__(self):
        """
        `self.encoder_inputs["question_ids"]` shape: [batch_size, max_seq_length]
        """
        return len(self.encoder_inputs["question_ids"])

    @overrides
    def __getitem__(self, idx):
        if self.config.model_name == "pickt":
            return_dict = {
                "concept_rel_embeds": self.concept_rel_embeds,
                "question_rel_embeds": self.question_rel_embeds,
                "concept2concept_edge": self.concept2concept_edge,
                "concept2question_edge": self.concept2question_edge,
                "question_ids": self.encoder_inputs["question_ids"][idx],
                "concept_ids": self.encoder_inputs["concept_ids"][idx],
                "type_ids": self.encoder_inputs["type_ids"][idx],
                "difficulty_ids": self.encoder_inputs["difficulty_ids"][idx],
                "discriminate_ids": self.encoder_inputs["discriminate_ids"][idx],
                "content_ids": self.encoder_inputs["content_ids"][idx],
                "activity_ids": self.encoder_inputs["activity_ids"][idx],
                "response_ids": self.decoder_inputs["response_ids"][idx],
                "elapsed_ids": self.decoder_inputs["elapsed_ids"][idx],
                "lag_ids": self.decoder_inputs["lag_ids"][idx],
                "labels": self.decoder_inputs["labels"][idx],
                "attention_mask": self.attention_mask,
            }
        elif self.config.model_name == "saint":
            return_dict = {
                "question_ids": self.encoder_inputs["question_ids"][idx],
                "concept_ids": self.encoder_inputs["concept_ids"][idx],
                "response_ids": self.decoder_inputs["response_ids"][idx],
                "elapsed_ids": self.decoder_inputs["elapsed_ids"][idx],
                "lag_ids": self.decoder_inputs["lag_ids"][idx],
                "labels": self.decoder_inputs["labels"][idx],
            }
        elif self.config.model_name == "gkt":
            return_dict = {
                "concept2concept_edge": self.concept2concept_edge,
                "concept_ids": self.encoder_inputs["concept_ids"][idx],
                "response_ids": self.decoder_inputs["response_ids"][idx],
                "labels": self.decoder_inputs["labels"][idx],
            }
        elif self.config.model_name in ["dkt", "sakt", "dkvmn"]:
            return_dict = {
                "concept_ids": self.encoder_inputs["concept_ids"][idx],
                "response_ids": self.decoder_inputs["response_ids"][idx],
                "labels": self.decoder_inputs["labels"][idx],
            }
        elif self.config.model_name in ["akt", "dtransformer"]:
            return_dict = {
                "concept_ids": self.encoder_inputs["concept_ids"][idx],
                "response_ids": self.decoder_inputs["response_ids"][idx],
                "question_ids": self.encoder_inputs["question_ids"][idx],
                "labels": self.decoder_inputs["labels"][idx],
            }
        
        return return_dict

    def _split_and_pad_sequence_data(self, student_seq: list, max_seq_length: int, pad_id: int):
        """
        각 학생의 응답을 max_seq_length 로 분할하고, 부족한 부분은 패딩값으로 채움.
        길이가 max_seq_length 보다 긴 응답은 분할되고, 짧은 응답은 패딩됨.
        리스트에서 텐서로 변환 후 return.
        
        Args:
            student_seq: 한 학생의 전체 풀이 이력과 관련된 데이터 모음 (list type)
            max_seq_length: 입력 시퀀스의 최대 길이
            pad_id: PAD 토큰 ID 값
        Returns:
            result: [batch_size, max_seq_length]
        """
        padded_data = []
        for personal_seq in student_seq:
            for i in range(0, len(personal_seq), max_seq_length):    # # response를 max_seq_length 단위로 분할
                chunk = personal_seq[i:i+max_seq_length]
                if len(chunk) < max_seq_length:
                    chunk = chunk + [pad_id] * (max_seq_length - len(chunk))    # 패딩 적용
                padded_data.append(chunk)
        
        return torch.tensor(padded_data, dtype=torch.long)

    def _prepend_start_id_and_trim(self, tensor: torch.Tensor, start_id: int):
        """
        Args:
            tensor: Input tensor of shape [batch_size, max_seq_length]
            start_id: 시작 토큰 ID 값
        Returns:
            result: [batch_size, max_seq_length] 모양 유지
        """
        # 1. 시작 토큰 생성 (벡터화 연산)
        start_tokens = torch.full(
            size=(tensor.size(0), 1),
            fill_value=start_id,
            dtype=tensor.dtype,
            device=tensor.device
        )
        
        # 2. 원본 텐서에서 마지막 요소 제거
        trimmed = tensor[:, :-1]
        
        # 3. 두 텐서 결합 (메모리 효율적 연산)
        return torch.cat([start_tokens, trimmed], dim=1)