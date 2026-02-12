# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-05-28

import math
import torch
import torch.nn as nn
# import torch.nn.functional as F

from typing import Optional, Union, Tuple
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_outputs import BaseModelOutput as PicktEncoderOutput


class PicktEncoderSelfAttention(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltSelfAttention class 를 PICKT 에 맞게 수정.
    LILT 의 사상인 Text, Layout 을 상호작용하는 부분을 Question, Concept 을 상호작용 하는 걸로 수정.
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Question embedding 용 q, k, v
        self.question_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.question_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.question_value = nn.Linear(config.hidden_size, self.all_head_size)

        # Concept embedding 용 q, k, v
        ### LILT 의 channel_shrink_ratio 는 사용하지 않음 (Layout 과 사상이 다르기 때문)
        self.concept_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.concept_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.concept_value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # position_embedding_type=absolute 만 사용할 예정
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x, r=1):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size // r)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        concept_inputs,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        concept_key_layer = self.transpose_for_scores(self.concept_key(concept_inputs))
        concept_value_layer = self.transpose_for_scores(self.concept_value(concept_inputs))
        concept_query_layer = self.transpose_for_scores(self.concept_query(concept_inputs))

        mixed_question_query_layer = self.question_query(hidden_states)

        question_key_layer = self.transpose_for_scores(self.question_key(hidden_states))
        question_value_layer = self.transpose_for_scores(self.question_value(hidden_states))
        question_query_layer = self.transpose_for_scores(mixed_question_query_layer)

        question_attention_scores = torch.matmul(question_query_layer, question_key_layer.transpose(-1, -2))
        concept_attention_scores = torch.matmul(concept_query_layer, concept_key_layer.transpose(-1, -2))

        # Been add this: 상삼각 행렬을 적용해주기 위함
        if attention_mask is not None:
            attention_mask = attention_mask.float().masked_fill_(attention_mask, float('-inf'))
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
            question_attention_scores = question_attention_scores + attention_mask
            concept_attention_scores = concept_attention_scores + attention_mask

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=question_query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", question_query_layer, positional_embedding)
                question_attention_scores = question_attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", question_query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", question_key_layer, positional_embedding)
                question_attention_scores = question_attention_scores + relative_position_scores_query + relative_position_scores_key

        tmp_question_attention_scores = question_attention_scores / math.sqrt(self.attention_head_size)
        tmp_concept_attention_scores = concept_attention_scores / math.sqrt(self.attention_head_size)
        question_attention_scores = tmp_question_attention_scores + tmp_concept_attention_scores
        concept_attention_scores = tmp_concept_attention_scores + tmp_question_attention_scores

        # Normalize the attention scores to probabilities.
        concept_attention_probs = nn.Softmax(dim=-1)(concept_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        concept_attention_probs = self.dropout(concept_attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            concept_attention_probs = concept_attention_probs * head_mask

        concept_context_layer = torch.matmul(concept_attention_probs, concept_value_layer)

        concept_context_layer = concept_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = concept_context_layer.size()[:-2] + (self.all_head_size,)
        concept_context_layer = concept_context_layer.view(*new_context_layer_shape)

        # Normalize the attention scores to probabilities.
        question_attention_probs = nn.Softmax(dim=-1)(question_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        question_attention_probs = self.dropout(question_attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            question_attention_probs = question_attention_probs * head_mask

        question_context_layer = torch.matmul(question_attention_probs, question_value_layer)

        question_context_layer = question_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = question_context_layer.size()[:-2] + (self.all_head_size,)
        question_context_layer = question_context_layer.view(*new_context_layer_shape)

        outputs = (
            ((question_context_layer, concept_context_layer), question_attention_probs)
            if output_attentions
            else ((question_context_layer, concept_context_layer),)
        )

        return outputs


class PicktEncoderAttention(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltAttention class 를 PICKT 에 맞게 수정.
    LILT 의 사상인 Text, Layout 을 상호작용하는 부분을 Question, Concept 을 상호작용 하는 걸로 수정.
    """
    def __init__(self, config, position_embedding_type="absolute"):
        super().__init__()
        self.self = PicktEncoderSelfAttention(config, position_embedding_type=position_embedding_type)
        self.question_output = PicktSelfOutput(config)
        self.concept_output = PicktSelfOutput(config)
        # self.pruned_heads = set()

    # # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     heads, index = find_pruneable_heads_and_indices(
    #         heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    #     )

    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        concept_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            concept_inputs,
            attention_mask,
            head_mask,
            output_attentions,
        )
        question_attention_output = self.question_output(self_outputs[0][0], hidden_states)
        concept_attention_output = self.concept_output(self_outputs[0][1], concept_inputs)
        outputs = ((question_attention_output, concept_attention_output),) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PicktEncoderLayer(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltLayer class 를 PICKT 에 맞게 수정.
    PICKT 는 Encoder 와 Decoder 의 Attention 이 다르기 때문에 해당 사항을 반영.
    """
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PicktEncoderAttention(config)
        
        self.question_intermediate = PicktIntermediate(config)
        self.question_output = PicktOutput(config)
        self.concept_intermediate = PicktIntermediate(config)
        self.concept_output = PicktOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        concept_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            concept_inputs,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        question_attention_output = self_attention_outputs[0][0]
        concept_attention_output = self_attention_outputs[0][1]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        question_layer_output = apply_chunking_to_forward(
            self.question_feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, question_attention_output
        )
        concept_layer_output = apply_chunking_to_forward(
            self.concept_feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, concept_attention_output
        )
        outputs = ((question_layer_output, concept_layer_output),) + outputs

        return outputs

    # Copied from transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk
    def question_feed_forward_chunk(self, attention_output):
        intermediate_output = self.question_intermediate(attention_output)
        layer_output = self.question_output(intermediate_output, attention_output)
        return layer_output

    def concept_feed_forward_chunk(self, attention_output):
        intermediate_output = self.concept_intermediate(attention_output)
        layer_output = self.concept_output(intermediate_output, attention_output)
        return layer_output


class PicktEncoder(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltEncoder class 를 PICKT 에 맞게 수정.
    PICKT 는 Encoder 와 Decoder 의 Attention 이 다르기 때문에 해당 사항을 반영.
    """
    # Copied from transformers.models.bert.modeling_bert.BertEncoder.__init__ with Bert->Lilt
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PicktEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        concept_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], PicktEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    concept_inputs,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    concept_inputs,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0][0]
            concept_inputs = layer_outputs[0][1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return PicktEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class PicktSelfOutput(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltSelfOutput class 와 동일.
    class 명만 수정하였음.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class PicktIntermediate(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltIntermediate class 와 동일.
    class 명만 수정하였음.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class PicktOutput(nn.Module):
    """
    transformers modeling_lilt.py 에서 LiltOutput class 와 동일.
    class 명만 수정하였음.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class PicktDecoderLayer(nn.Module):
    """
    torch transformer.py 에서 TransformerDecoderLayer class 를 PICKT 에 맞게 수정.
    layer 별 dim 지정하는 __init__ method 를 transformers library 와 동일 Level 로 구성.
    """
    def __init__(self, config):
        super().__init__()
        self.norm_first = False
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
            bias=True
        )
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)
        
        self.multihead_attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
            bias=True
        )
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)
        
        # Implementation of Feedforward model(=intermediate)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)

    # def __setstate__(self, state):
    #     if "activation" not in state:
    #         state["activation"] = F.relu
    #     super().__setstate__(state)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.FloatTensor] = None,
        memory_mask: Optional[torch.FloatTensor] = None,
        tgt_key_padding_mask: Optional[torch.FloatTensor] = None,
        memory_key_padding_mask: Optional[torch.FloatTensor] = None,
        tgt_is_causal: Optional[bool] = False,
        memory_is_causal: Optional[bool] = False,
    ) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class PicktDecoder(nn.Module):
    """
    torch transformer.py 에서 TransformerDecoder class 를 PICKT 에 맞게 수정.
    DecoderLayer 지정하는 __init__ method 를 transformers library 와 동일 Level 로 구성.
    """
    # __constants__ = ["norm"]
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_decoder_layers
        self.num_attention_heads = config.num_attention_heads
        
        self.layer = nn.ModuleList([PicktDecoderLayer(config) for _ in range(self.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layer[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layer:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask[0],
                memory_mask=memory_mask[0],
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class PicktClassificationHead(nn.Module):
    """Head classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states