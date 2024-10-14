# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:50:04 2020

@author: a
"""
import math
from src.models.layers import LinkAttention
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from src.utils import pack_sequences, pack_pre_sequences, unpack_sequences, split_text, load_protvec, graph_pad
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from src.models.transformer import Transformer
from torch_geometric.utils import to_dense_adj
from torch.nn.parameter import Parameter
import time
device_ids = [0, 1, 2, 3]


class DAT3(nn.Module):
    def __init__(self, embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout_rate,
                 alpha, n_heads, graph_input_dim=78, rnn_layers=2,
                 attn_type='dotproduct', vocab=26, smile_vocab=63, is_pretrain=True,
                 is_drug_pretrain=False, n_extend=1, num_feature_xd=156,output_dim=128,
                 dropout=0.2, feature_dim=1024, DENSE_DIM=16, ATTENTION_HEADS=4,):
        super(DAT3, self).__init__()

        # attention部分
        self.dropout = nn.Dropout(dropout_rate)  # 丢弃率
        self.leakyrelu = nn.LeakyReLU(alpha)  # LeakyReLU激活函数
        self.relu = nn.ReLU()  # ReLU激活函数

        # drug-1D
        self.smiles_vocab = smile_vocab  # SMILES词汇表大小
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=smile_vocab)  # SMILES嵌入层
        self.rnn_layers = 2  # RNN层数
        self.is_bidirectional = True  # 是否双向
#         self.encoder = Transformer(256, 256)  # ////////////// MACCS 指纹的长度=2
        # 输入维度128，输出维度128，LSTM层数2，输入数据的维度顺序，是否使用双向LSTM，丢弃率
        self.smiles_rnn = nn.LSTM(rnn_dim, rnn_dim, self.rnn_layers, batch_first=True
                                  , bidirectional=self.is_bidirectional, dropout=dropout_rate)  # SMILES循环神经网络
        self.smiles_out_fc = nn.Linear(rnn_dim * 2, rnn_dim)  # SMILES输出全连接层
        self.concat_input_fc = nn.Linear(output_dim, output_dim * 2)  # 句子输入全连接层
        self.fc_x = nn.Linear(166 * 256, output_dim)

        # drug-2D
        self.Conv1 = GCNConv(num_feature_xd, num_feature_xd)
        self.fc_g1 = nn.Linear(num_feature_xd, num_feature_xd * 2)
        self.Conv2 = GCNConv(num_feature_xd * 2, num_feature_xd * 2)
        self.fc_g2 = nn.Linear(num_feature_xd * 2, output_dim)

        # protein-1D
        self.is_pretrain = is_pretrain  # 是否预训练
        if not is_pretrain:
            self.vocab = vocab  # 词汇表大小
            self.embed = nn.Embedding(vocab + 1, embedding_dim, padding_idx=vocab)  # 嵌入层
        self.rnn_layers = 2  # RNN层数
        self.is_bidirectional = True  # 是否双向
        self.sentence_input_fc = nn.Linear(embedding_dim, rnn_dim)  # 句子输入全连接层
        self.encode_rnn = nn.LSTM(rnn_dim, rnn_dim, self.rnn_layers, batch_first=True
                                  , bidirectional=self.is_bidirectional, dropout=dropout_rate)  # 编码RNN
        self.rnn_out_fc = nn.Linear(rnn_dim * 2, rnn_dim)  # RNN输出全连接层
        self.fc_xt = nn.Linear(1024 * 256, output_dim)

        # protein-2D
        self.gc1 = GraphConvolution(feature_dim, hidden_dim)
        self.relu_p = nn.LeakyReLU(dropout, inplace=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(256, 16 * 1024)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.attention_p = Attention_p(output_dim, DENSE_DIM, ATTENTION_HEADS)
        # 连接部分
        self.attention_layer = nn.MultiheadAttention(128, 1)
        self.out_fc1 = nn.Linear(hidden_dim, hidden_dim * 4)  # 全连接层1
        self.out_fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)  # 全连接层2
        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)  # 全连接层3
        self.layer_norm = nn.LayerNorm(rnn_dim * 2)  # 层归一化

    def forward(self, protein, smiles, node_drug, edge_drug, batch, vae, node_proteins, edge_proteins):
        # Drug-1D
        batchsize = len(protein)
        #256
        smiles_lengths = np.array([len(x) for x in smiles])
        temp = (torch.zeros(batchsize, max(smiles_lengths)) * 63).long()
        for i in range(batchsize):
            temp[i, :len(smiles[i])] = smiles[i]
        smiles = temp.cuda()
        smiles, _ = self.smiles_rnn(smiles)
        smiles = smiles.contiguous().view(-1, 166 * 256)
        smiles = self.fc_x(smiles)  # (128, 128)

        # Drug-2D
        X_list = get_augmented_features(node_drug, vae)
        X_list = torch.cat((X_list, node_drug), dim=1)
        drug = self.Conv1(X_list, edge_drug)
        drug = self.relu(self.fc_g1(drug))
        drug = self.Conv2(drug, edge_drug)
        drug = self.relu(self.fc_g2(drug))
        drug = gmp(drug, batch)   # (128, 128)

        #  proteins-1D
        protein = torch.stack(protein)
        target = self.sentence_input_fc(protein)  # smiles序列进入nn.Linear(256, 128)，输出维度为128
        target, _ = self.encode_rnn(target)  # 使用双向LSTM，输出维度128
        target = target.contiguous().view(-1, 1024 * 256)
        target = self.fc_xt(target)     # (128, 128)

        #  proteins-2D
        node_proteins = torch.stack(node_proteins)
        edge_proteins = torch.stack(edge_proteins)
        node = self.gc1(node_proteins, edge_proteins)
        node = self.relu_p(self.ln1(node))
        node = self.gc2(node, edge_proteins)
        node = self.relu_p(self.ln2(node))
        att = self.attention_p(node)
        node_feature_embedding = att @ node
        node = torch.sum(node_feature_embedding, 1) / self.attention_p.n_heads
        #  concat
        x_att, _ = self.attention_layer(drug, smiles, smiles)
        xt_att, _ = self.attention_layer(node, target, target)

        x_cat = smiles * 0.5 + x_att * 0.5
        xt_cat = target * 0.5 + xt_att * 0.5

        out = torch.cat((x_cat, xt_cat), dim=1)
        d_block = self.dropout(self.relu(self.out_fc1(out)))  # 256*3->256*8
        #256,256*8
        out = self.dropout(self.relu(self.out_fc2(d_block)))  # 256*8->256*2
       #256,512
        out = self.out_fc3(out).squeeze()  # 256*2->1
        #256,1
        return d_block, out

def get_augmented_features(x, vae):
    z = torch.randn([x.size(0), vae.latent_size]).cuda()
    augmented_features = vae.inference(z, x).detach()  # 3933 10 3933 78--->4164 128
    return augmented_features  # 4036 128


class GraphConvolution(nn.Module):

    def __init__(self, input_rep, output_rep, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_rep = input_rep
        self.output_rep = output_rep
        self.weight = Parameter(torch.FloatTensor(input_rep, output_rep))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_rep))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        adj = adj.to(torch.float32)
        support = input @ self.weight
        output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Attention_p(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention_p, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = self.fc2(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        return attention
