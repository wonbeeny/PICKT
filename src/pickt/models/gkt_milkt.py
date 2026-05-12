# coding : utf-8
# edit : 
# - author : jhljx
# - date : 2025-06-11
# - link : https://github.com/jhljx/GKT/blob/master/models.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj

from .modeling_outputs import KTModelOutput as GktModelOutput

class GktMilktModel(nn.Module):

    def __init__(self, config):
        super(GktMilktModel, self).__init__()
        self.concept_num = config.num_concept
        self.hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim
        self.edge_type_num = config.edge_type_num

        self.dropout = config.dropout

        self.bias = config.bias

        self.edge_type_num = config.edge_type_num

        self.res_len = 2 if config.binary else 12
        self.has_cuda = config.has_cuda

        # one-hot feature and question
        one_hot_feat = torch.eye(self.res_len * self.concept_num)
        self.one_hot_feat = one_hot_feat.cuda() if self.has_cuda else one_hot_feat
        self.one_hot_q = torch.eye(self.concept_num, device=self.one_hot_feat.device)
        zero_padding = torch.zeros(1, self.concept_num, device=self.one_hot_feat.device)
        self.one_hot_q = torch.cat((self.one_hot_q, zero_padding), dim=0)
        # concept and concept & response embeddings
        self.emb_x = nn.Embedding(self.res_len * self.concept_num, self.embedding_dim)
        # last embedding is used for padding, so dim + 1
        self.emb_c = nn.Embedding(self.concept_num + 1, self.embedding_dim, padding_idx=-1)

        # f_self function and f_neighbor functions
        mlp_input_dim = self.hidden_dim + self.embedding_dim
        self.f_self = MLP(mlp_input_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout, bias=self.bias)
        self.f_neighbor_list = nn.ModuleList()
        # f_in and f_out functions
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout, bias=self.bias))
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout, bias=self.bias))
        
        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(self.hidden_dim, self.concept_num)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim, bias=self.bias)
        # prediction layer
        self.predict = nn.Linear(self.hidden_dim, 1, bias=self.bias)

    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(self, xt, qt, ht, batch_size):
        r"""
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Shape:
            xt: [batch_size]
            qt: [batch_size]
            ht: [batch_size, concept_num, hidden_size]
            tmp_ht: [batch_size, concept_num, hidden_size + embedding_size]
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        x_idx_mat = torch.arange(self.res_len * self.concept_num, device=xt.device)
        x_embedding = self.emb_x(x_idx_mat)  # [res_len * concept_num, embedding_size]
        masked_feat = F.embedding(xt[qt_mask], self.one_hot_feat)  # [mask_num, res_len * concept_num]
        res_embedding = masked_feat.mm(x_embedding)  # [mask_num, embedding_size]
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.concept_num * torch.ones((batch_size, self.concept_num), device=xt.device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.concept_num, device=xt.device).expand(qt_mask.sum(), -1)
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, concept_num, embedding_size]

        index_tuple = (torch.arange(mask_num, device=xt.device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)  # [batch_size, concept_num, hidden_size + embedding_size]
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_size + embedding_size]
            qt: [batch_size]
            m_next: [batch_size, concept_num, hidden_size]
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        masked_qt = qt[qt_mask]  # [mask_num, ]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_size + embedding_size]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_size + embedding_size]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_size]
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.concept_num, 1)  #[mask_num, concept_num, hidden_size + embedding_size]
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)  #[mask_num, concept_num, 2 * (hidden_size + embedding_size)]
        
        adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
        reverse_adj = self.graph.permute(1, 0)[masked_qt.long()].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
        # self.f_neighbor_list[0](neigh_ht) shape: [mask_num, concept_num, hidden_dim]
        neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht)
        # neigh_features: [mask_num, concept_num, hidden_size]
        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_size + embedding_size]
            ht: [batch_size, concept_num, hidden_size]
            qt: [batch_size]
            h_next: [batch_size, concept_num, hidden_size]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_num = qt_mask.nonzero().shape[0]
        # GNN Aggregation
        m_next = self._agg_neighbors(tmp_ht, qt)  # [batch_size, concept_num, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim),
                       ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device),)
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.concept_num, self.hidden_dim))
        return h_next

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            h_next: [batch_size, concept_num, hidden_size]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        return y

    def _get_next_pred(self, yt, q_next):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, concept_num]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = q_next
        next_qt = torch.where(next_qt != -1, next_qt, self.concept_num * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, concept_num]
        # dot product between yt and one_hot_qt
        pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size, ]
        return pred

    def _convert_heterodata_to_adj(self, self_loops=True):
        # HeteroData에서 edge_index 추출
        edge_index = self.graph_data['concept', 'prereq', 'concept'].edge_index
        
        # 자가 루프 추가/제거
        if self_loops:
            edge_index, _ = add_remaining_self_loops(edge_index)
        else:
            edge_index, _ = remove_self_loops(edge_index)
        
        # dense adjacency matrix 변환
        adj = to_dense_adj(edge_index, max_num_nodes=self.concept_num).squeeze(0)
        
        ### GPU 설정 (선택사항)
        # if torch.cuda.is_available():
        #    adj = adj.cuda()
        
        return adj

    # def forward(self, graph_data, response_ids, concept_ids):
    def forward(self, graph_data, concept_ids, response_ids):
        r"""
        Parameters:
            response_ids: input one-hot matrix
            concept_ids: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            response_ids: [batch_size, seq_len]
            concept_ids: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of concept_ids answered at the next timestamp
        """
        self.graph_data = graph_data
        self.graph = self._convert_heterodata_to_adj()
        
        batch_size, seq_len = response_ids.shape
        response_ids = torch.where(concept_ids >= 0, concept_ids * self.res_len + response_ids, self.concept_num * self.res_len)
        ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device=response_ids.device))
        pred_list = []
        for i in range(seq_len):
            xt = response_ids[:, i]  # [batch_size]
            qt = concept_ids[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)  # [batch_size, concept_num, hidden_dim + embedding_dim]
            h_next = self._update(tmp_ht, ht, qt)  # [batch_size, concept_num, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # update new ht
            yt = self._predict(h_next, qt)  # [batch_size, concept_num]
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, concept_ids[:, i + 1])
                pred_list.append(pred)
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]

        return GktModelOutput(
            logits=None,
            predictions=pred_res,
            encoder_output=None,
            decoder_output=None,
        )

class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        # the paper said they added Batch Normalization for the output of MLPs, as shown in Section 4.2
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 or 0 will cause BatchNorm error, so return the input directly
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:  # len(input_size()) == 2
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)  # pay attention to add training=self.training
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)

class EraseAddGate(nn.Module):
    """
    Erase & Add Gate module
    NOTE: this erase & add gate is a bit different from that in DKVMN.
    For more information about Erase & Add gate, please refer to the paper "Dynamic Key-Value Memory Networks for Knowledge Tracing"
    The paper can be found in https://arxiv.org/abs/1611.08108
    """

    def __init__(self, feature_dim, concept_num, bias=True):
        super(EraseAddGate, self).__init__()
        # weight
        self.weight = nn.Parameter(torch.rand(concept_num))
        self.reset_parameters()
        # erase gate
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        # add gate
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        r"""
        Params:
            x: input feature matrix
        Shape:
            x: [batch_size, concept_num, feature_dim]
            res: [batch_size, concept_num, feature_dim]
        Return:
            res: returned feature matrix with old information erased and new information added
        The GKT paper didn't provide detailed explanation about this erase-add gate. As the erase-add gate in the GKT only has one input parameter,
        this gate is different with that of the DKVMN. We used the input matrix to build the erase and add gates, rather than $\mathbf{v}_{t}$ vector in the DKVMN.
        """
        erase_gate = torch.sigmoid(self.erase(x))  # [batch_size, concept_num, feature_dim]
        # self.weight.unsqueeze(dim=1) shape: [concept_num, 1]
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))  # [batch_size, concept_num, feature_dim]
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res
