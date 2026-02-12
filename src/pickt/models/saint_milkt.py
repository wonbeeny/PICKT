# coding : utf-8
# edit : 
# - author : shivanandmn
# - date : 2025-05-22
# - link : https://github.com/shivanandmn/SAINT_plus-Knowledge-Tracing-/blob/main/saint.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling_outputs import KTModelOutput as SaintModelOutput


class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out


class EncoderEmbedding(nn.Module):
    def __init__(self, config):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = config.hidden_size
        self.max_seq_length = config.max_seq_length
        self.question_embed = nn.Embedding(config.num_question, self.n_dims, padding_idx=config.question2id["pad_id"])
        self.concept_embed = nn.Embedding(config.num_concept, self.n_dims, padding_idx=config.concept2id["pad_id"])
        self.position_embed = nn.Embedding(self.max_seq_length, self.n_dims)

    def forward(self, question_ids, concept_ids):
        q = self.question_embed(question_ids)
        c = self.concept_embed(concept_ids)
        seq = torch.arange(self.max_seq_length, device=q.device).unsqueeze(0)
        p = self.position_embed(seq)
        
        return p + c + q


class DecoderEmbedding(nn.Module):
    def __init__(self, config):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = config.hidden_size
        self.max_seq_length = config.max_seq_length
        self.response_embed = nn.Embedding(config.num_response, self.n_dims, padding_idx=config.response2id["pad_id"])
        self.elapsed_embed = nn.Embedding(config.num_elapsed, self.n_dims, padding_idx=config.elapsed2id["pad_id"])
        self.lag_embed = nn.Embedding(config.num_lag, self.n_dims, padding_idx=config.lag2id["pad_id"])
        self.position_embed = nn.Embedding(self.max_seq_length, self.n_dims)

    def forward(self, response_ids, elapsed_ids, lag_ids):
        r = self.response_embed(response_ids)
        e = self.elapsed_embed(elapsed_ids)
        l = self.lag_embed(lag_ids)
        seq = torch.arange(self.max_seq_length, device=r.device).unsqueeze(0)
        p = self.position_embed(seq)
        
        return p + r + e + l


class StackedNMultiHeadAttention(nn.Module):
    def __init__(self, config, n_stacks, n_multihead):
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_stacks = n_stacks
        self.n_multihead = n_multihead
        self.norm_layers = nn.LayerNorm(config.hidden_size)
        # n_stacks has n_multiheads each
        self.multihead_layers = nn.ModuleList(n_stacks*[nn.ModuleList(n_multihead*[nn.MultiheadAttention(embed_dim=config.hidden_size,
                                                                                                         num_heads=config.num_attention_heads,
                                                                                                         dropout=config.attention_probs_dropout_prob), ]), ])
        self.ffn = nn.ModuleList(n_stacks*[FFN(config)])
        self.mask = torch.triu(torch.ones(config.max_seq_length, config.max_seq_length),
                               diagonal=1).to(dtype=torch.bool)

    def forward(self, input_q, input_k, input_v, encoder_output=None, break_layer=None):
        for stack in range(self.n_stacks):
            for multihead in range(self.n_multihead):
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v)
                heads_output, _ = self.multihead_layers[stack][multihead](query=norm_q.permute(1, 0, 2),
                                                                          key=norm_k.permute(
                                                                              1, 0, 2),
                                                                          value=norm_v.permute(
                                                                              1, 0, 2),
                                                                          attn_mask=self.mask.to(input_q.device))
                heads_output = heads_output.permute(1, 0, 2)
                #assert encoder_output != None and break_layer is not None
                if encoder_output != None and multihead == break_layer:
                    assert break_layer <= multihead, " break layer should be less than multihead layers and postive integer"
                    input_k = input_v = encoder_output
                    input_q = input_q + heads_output
                else:
                    input_q = input_q + heads_output
                    input_k = input_k + heads_output
                    input_v = input_v + heads_output
            last_norm = self.norm_layers(heads_output)
            ffn_output = self.ffn[stack](last_norm)
            ffn_output = ffn_output + heads_output
        # after loops = input_q = input_k = input_v
        return ffn_output


class SaintMilktModel(nn.Module):
    def __init__(self, config):
        # n_encoder,n_detotal_responses,seq_len,max_time=300+1
        super(SaintMilktModel, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.encoder_layer = StackedNMultiHeadAttention(config=config,
                                                        n_stacks=config.num_encoder_layers,
                                                        n_multihead=1)
        self.decoder_layer = StackedNMultiHeadAttention(config=config,
                                                        n_stacks=config.num_decoder_layers,
                                                        n_multihead=2)
        self.encoder_embedding = EncoderEmbedding(config)
        self.decoder_embedding = DecoderEmbedding(config)
        self.head_layer = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        question_ids,
        concept_ids,
        response_ids,
        elapsed_ids,
        lag_ids
    ):
        enc = self.encoder_embedding(question_ids, concept_ids)
        dec = self.decoder_embedding(response_ids, elapsed_ids, lag_ids)
        # this encoder
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc,
                                            input_v=enc)
        #this is decoder
        decoder_output = self.decoder_layer(input_k=dec,
                                            input_q=dec,
                                            input_v=dec,
                                            encoder_output=encoder_output,
                                            break_layer=1)
        # fully connected layer
        logits = self.head_layer(decoder_output)
        predictions = torch.sigmoid(logits)

        return SaintModelOutput(
            logits=logits,
            predictions=predictions,
            encoder_output=encoder_output,
            decoder_output=decoder_output,
        )