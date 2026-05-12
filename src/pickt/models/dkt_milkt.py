# coding : utf-8
# edit : 
# - author : hcnoh
# - date : 2025-06-04
# - link : https://github.com/hcnoh/knowledge-tracing-collection-pytorch/blob/main/models/dkt.py

import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from .modeling_outputs import KTModelOutput as DktModelOutput


class DktMilktModel(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
    '''
    def __init__(self, config):
        super().__init__()
        self.num_q = config.num_concept
        self.emb_size = config.n_dims 
        self.hidden_size = config.hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, concept_ids, response_ids):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        response_ids[response_ids==2]=0        
        x = concept_ids + self.num_q * response_ids
        
        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        logits = self.dropout_layer(y)
        predictions = torch.sigmoid(logits)

        return DktModelOutput(
            logits=logits,
            predictions=predictions,
        )

