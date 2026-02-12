# coding : utf-8
# edit : 
# - author : lcn
# - date : 2025-08-14

import torch
import torch.nn as nn

from typing import Optional, Union, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, Linear
from transformers.activations import ACT2FN

from .modeling_outputs import KTModelOutput as PicktModelOutput
from .pickt_base import PicktEncoder, PicktDecoder, PicktClassificationHead


class HANEncoder(nn.Module):
    """
    지식맵 정보를 모델에 반영하기 위해 HANConv 인 GNN 사용.
    PicktQuestionEmbedding 에는 input 으로 문제의 text 로 HANEncoder Output 이 사용됨.
    PicktConceptEmbedding 에는 input 으로 문제의 개념 정보로 HANEncoder Output 이 사용됨.
    """
    def __init__(self, config):    # metadata, in_channels, hidden_channels, out_channels, num_heads=4):
        super().__init__()
        # 1st HAN 레이어: 기본 관계 패턴 학습
        self.han1 = HANConv(
            in_channels=config.han_in_channels, 
            out_channels=config.han_hidden_channels, 
            metadata=config.han_metadata, 
            heads=config.han_num_heads
        )
        self.han_act_fn = ACT2FN[config.hidden_act]
        # 2nd HAN 레이어: 고수준 관계 학습
        self.han2 = HANConv(
            in_channels=config.han_hidden_channels, 
            out_channels=config.han_hidden_channels,
            metadata=config.han_metadata, 
            heads=config.han_num_heads
        )

        # 1st 투영 레이어: 개념 노드 특화 처리
        self.concept_proj = Linear(config.han_hidden_channels, config.hidden_size)
        # 2nd 투영 레이어: 문제 노드 특화 처리
        self.question_proj = Linear(config.han_hidden_channels, config.hidden_size)

    def forward(
        self, 
        x_dict, 
        edge_index_dict,
        concept_node_nm='concept', 
        question_node_nm='question'
    ):
        # 1단계: 기본 그래프 구조 학습
        x = self.han1(x_dict, edge_index_dict)
        x = {k: self.han_act_fn(v) for k, v in x.items()}
        
        # 2단계: 추상적 관계 모델링
        x = self.han2(x, edge_index_dict)
        
        # 3단계: 개념 및 문제 노드 선택적 추출
        ### [num_concepts, out_channels], [num_questions, out_channels]
        return self.concept_proj(x[concept_node_nm]), self.question_proj(x[question_node_nm])


class PicktQuestionEmbedding(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltTextEmbeddings class 를 PICKT 에 맞게 수정.
    문제의 정보를 반영하는 Embedding Layer 로 구성하였음.
    """
    def __init__(self, config):
        super().__init__()
        self.question_embeddings = nn.Embedding(config.num_question, config.hidden_size, padding_idx=config.question2id["pad_id"])
        self.type_embeddings = nn.Embedding(config.num_type, config.hidden_size, padding_idx=config.type2id["pad_id"])
        self.difficulty_embeddings = nn.Embedding(config.num_difficulty, config.hidden_size, padding_idx=config.difficulty2id["pad_id"])
        self.discriminate_embeddings = nn.Embedding(config.num_discriminate, config.hidden_size, padding_idx=config.discriminate2id["pad_id"])
        # self.activity_embeddings = nn.Embedding(config.num_activity, config.hidden_size, padding_idx=config.activity2id["pad_id"])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_seq_length).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.question_pad_id = config.question2id["pad_id"]

    def forward(
        self,
        question_ids=None,
        type_ids=None,
        difficulty_ids=None,
        discriminate_ids=None,
        # activity_ids=None,
        question_rel_embeds=None,
        position_ids=None
    ):
        if position_ids is None:
            if question_ids is not None:
                position_ids = self.position_ids[:, 0:question_ids.shape[1]]
            else:
                raise AssertionError(
                    f"The question_ids is an absolutely necessary value."
                )
        
        question_embeds = self.question_embeddings(question_ids)
        type_embeds = self.type_embeddings(type_ids)
        difficulty_embeds = self.difficulty_embeddings(difficulty_ids)
        discriminate_embeds = self.discriminate_embeddings(discriminate_ids)
        # activity_embeds = self.activity_embeddings(activity_ids)

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)

        # question_rel_embeds: GNN outputs
        mask = (question_ids==self.question_pad_id)
        mask_expanded = mask.unsqueeze(-1).expand_as(question_rel_embeds)
        question_rel_embeds[mask_expanded] = 0.0
        
        # question_embeddings = question_embeds + type_embeds + difficulty_embeds + discriminate_embeds + \
        #                       activity_embeds + question_rel_embeds + position_embeddings
        question_embeddings = question_embeds + type_embeds + difficulty_embeds + discriminate_embeds + \
                              question_rel_embeds + position_embeddings
        
        question_embeddings = self.LayerNorm(question_embeddings)
        question_embeddings = self.dropout(question_embeddings)
        return question_embeddings, position_ids


class PicktConceptEmbedding(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltTextEmbeddings class 를 PICKT 에 맞게 수정.
    문제의 개념 정보를 반영하는 Embedding Layer 로 구성하였음.
    """
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.concept_embeddings = nn.Embedding(config.num_concept, config.hidden_size, padding_idx=config.concept2id["pad_id"])
        # self.content_embeddings = nn.Embedding(config.num_content, config.hidden_size, padding_idx=config.content2id["pad_id"])

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_seq_length).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.concept_pad_id = config.concept2id["pad_id"]

    def forward(
        self,
        concept_ids=None,
        content_ids=None,
        concept_rel_embeds=None,
        position_ids=None,
    ):
        if position_ids is None:
            raise AssertionError(
                f"The position_ids is an absolutely necessary value."
            )
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)

        concept_embeds = self.concept_embeddings(concept_ids)
        # content_embeds = self.content_embeddings(content_ids)
        
        # concept_rel_embeds: GNN outputs
        mask = (concept_ids==self.concept_pad_id)
        mask_expanded = mask.unsqueeze(-1).expand_as(concept_rel_embeds)
        concept_rel_embeds[mask_expanded] = 0.0

        # concept_embeddings = concept_embeds + content_embeds + concept_rel_embeds + position_embeddings
        concept_embeddings = concept_embeds + concept_rel_embeds + position_embeddings
        
        concept_embeddings = self.LayerNorm(concept_embeddings)
        concept_embeddings = self.dropout(concept_embeddings)
        return concept_embeddings


class PicktResponseEmbedding(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltTextEmbeddings class 를 PICKT 에 맞게 수정.
    학생의 응답 정보를 반영하는 Embedding Layer 로 구성하였음.
    """
    def __init__(self, config):
        super().__init__()
        self.response_embeddings = nn.Embedding(config.num_response, config.hidden_size, padding_idx=config.response2id["pad_id"])
        self.elapsed_embeddings = nn.Embedding(config.num_elapsed, config.hidden_size, padding_idx=config.elapsed2id["pad_id"])
        self.lag_embeddings = nn.Embedding(config.num_lag, config.hidden_size, padding_idx=config.lag2id["pad_id"])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        response_ids=None,
        elapsed_ids=None,
        lag_ids=None,
        position_ids=None,
    ):
        if position_ids is None:
            raise AssertionError(
                f"The position_ids is an absolutely necessary value."
            )
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)

        response_embeds = self.response_embeddings(response_ids)
        elapsed_embeds = self.elapsed_embeddings(elapsed_ids)
        lag_embeds = self.lag_embeddings(lag_ids)
        response_embeddings = response_embeds + elapsed_embeds + lag_embeds + position_embeddings
        
        response_embeddings = self.LayerNorm(response_embeddings)
        response_embeddings = self.dropout(response_embeddings)
        return response_embeddings


class PicktDbekt22Model(nn.Module):
    """
    최종 PICKT 모델 구조
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.gnn_han = HANEncoder(config)
        self.question_embeddings = PicktQuestionEmbedding(config)
        self.concept_embeddings = PicktConceptEmbedding(config)
        self.response_embeddings = PicktResponseEmbedding(config)
        
        self.encoder = PicktEncoder(config)
        self.decoder = PicktDecoder(config)

        self.classification_head = PicktClassificationHead(
            input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            pooler_dropout=config.hidden_dropout_prob,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        graph_data: HeteroData = None,
        question_ids: Optional[torch.Tensor] = None,
        concept_ids: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        difficulty_ids: Optional[torch.Tensor] = None,
        discriminate_ids: Optional[torch.Tensor] = None,
        # content_ids: Optional[torch.Tensor] = None,
        # activity_ids: Optional[torch.Tensor] = None,
        response_ids: Optional[torch.Tensor] = None,
        elapsed_ids: Optional[torch.Tensor] = None,
        lag_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], PicktModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        concept_proj, question_proj = self.gnn_han(graph_data.x_dict, graph_data.edge_index_dict)
        question_rel_embeds = question_proj[question_ids]
        concept_rel_embeds = concept_proj[concept_ids]

        # import pdb; pdb.set_trace()

        question_embeddings_output, position_ids_output = self.question_embeddings(
            question_ids=question_ids,
            type_ids=type_ids,
            difficulty_ids=difficulty_ids,
            discriminate_ids=discriminate_ids,
            # activity_ids=activity_ids,
            question_rel_embeds=question_rel_embeds,
        )
        concept_embeddings_output = self.concept_embeddings(
            concept_ids=concept_ids,
            # content_ids=content_ids,
            concept_rel_embeds=concept_rel_embeds,
            position_ids=position_ids_output,
        )
        response_embeddings_output = self.response_embeddings(
            response_ids=response_ids,
            elapsed_ids=elapsed_ids,
            lag_ids=lag_ids,
            position_ids=position_ids_output,
        )
        
        encoder_output = self.encoder(
            hidden_states=question_embeddings_output,
            concept_inputs=concept_embeddings_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        decoder_output = self.decoder(
            tgt=response_embeddings_output,
            memory=encoder_output,
            tgt_mask=attention_mask,
            memory_mask=attention_mask,
        )

        logits = self.classification_head(decoder_output)
        predictions = torch.sigmoid(logits)
        
        # # just test for encoder archi running
        # if not return_dict:
        #     return (None, None) + ??

        return PicktModelOutput(
            logits=logits,
            predictions=predictions,
            encoder_output=encoder_output,
            decoder_output=decoder_output,
        )