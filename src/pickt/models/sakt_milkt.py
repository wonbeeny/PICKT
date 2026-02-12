import copy

import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout, Sequential, ReLU

from .modeling_outputs import KTModelOutput as SaktModelOutput

class SaktMilktModel(Module):
    def __init__(self, config, num_en=2, emb_type="qid"):
        super().__init__()

        self.emb_type = emb_type

        self.num_c = config.num_concept
        self.seq_len = config.max_seq_length
        self.emb_size = config.emb_size
        self.num_attn_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.num_en = num_en

        if emb_type.startswith("qid"):
            # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
            self.interaction_emb = Embedding(self.num_c * 2 + 1, self.emb_size) 
            self.exercise_emb = Embedding(self.num_c + 1, self.emb_size) 
            # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
        self.position_emb = Embedding(self.seq_len, self.emb_size)
        
        self.blocks = get_clones(Blocks(config), self.num_en)

        self.dropout_layer = Dropout(self.dropout)
        self.pred = Linear(self.emb_size, 1)

    def base_emb(self, q, r, qry):
        # x = q + self.num_c * r
        pad_idx = self.num_c * 2
        # x = torch.where((q >= 0), q + self.num_c * r, self.num_c * 2)
        
        x = torch.where((q >= 0) & (r <= 1), q + self.num_c * r, pad_idx)
        qry = torch.where(qry >= 0, qry, self.num_c)
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)

        posemb = self.position_emb(pos_encode(xemb.shape[1]).to(xemb.device))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, q, r, qry, qtest=False):
        emb_type = self.emb_type
        qshftemb, xemb = None, None
        if emb_type == "qid":
            qshftemb, xemb = self.base_emb(q, r, qry)
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)
        
        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        
        if not qtest:
            return SaktModelOutput(
                logits=None,
                predictions=p,
                encoder_output=None,
                decoder_output=None,
            )
        else:
            return p, xemb


class Blocks(Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.attn = MultiheadAttention(config.emb_size, config.num_attention_heads, dropout=config.dropout)
        self.attn_dropout = Dropout(config.dropout)
        self.attn_layer_norm = LayerNorm(config.emb_size)

        self.FFN = transformer_FFN(config)
        self.FFN_dropout = Dropout(config.dropout)
        self.FFN_layer_norm = LayerNorm(config.emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len=k.shape[0]).to(k.device)
        
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb


class transformer_FFN(Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.emb_size = config.emb_size
        self.dropout = config.dropout
        self.FFN = Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
            # Dropout(self.dropout),
        )

    def forward(self, in_fea):
        return self.FFN(in_fea)


def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool)  #.to(device)


def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0)  #.to(device)


def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
