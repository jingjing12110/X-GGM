# @File : gcn.py 
# @Time : 2020/8/17 
# @Email : jingjingjiang2017@gmail.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxrt.modeling import GeLU


class GCNConv(nn.Module):
    def __init__(self, dim_hidden, dropout=0.0):
        super(GCNConv, self).__init__()
        self.ctx_layer = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.layer_norm = nn.LayerNorm(dim_hidden)
        self.dropout = nn.Dropout(p=dropout)

        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.ctx_layer.weight)

    def forward(self, x, adj):
        """
        @param x: (bs, num_nodes, embed_size)
        @param adj: (bs, num_nodes, num_nodes)
        @return:
        """
        node_embeds = x + self.dropout(self.ctx_layer(torch.bmm(adj, x)))
        return self.layer_norm(node_embeds)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_layers, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout_p = dropout

        self.gnn_layers = nn.ModuleList()
        self.linear_prediction = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.gnn_layers.append(
                    GCNConv(input_dim)
                )
                self.linear_prediction.append(
                    nn.Sequential(nn.Linear(input_dim, hidden_dims[i]),
                                  GeLU(),
                                  nn.LayerNorm(hidden_dims[i]),
                                  ))
            else:
                self.gnn_layers.append(
                    GCNConv(hidden_dims[i - 1])
                )
                self.linear_prediction.append(
                    nn.Sequential(nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                                  GeLU(),
                                  nn.LayerNorm(hidden_dims[i]),
                                  ))
        self.linear_prediction.append(
            nn.Sequential(nn.Linear(hidden_dims[-2], hidden_dims[-1]),
                          GeLU(),
                          nn.LayerNorm(hidden_dims[-1]),
                          ))

    def forward(self, x, adj):
        hidden_states = [x]
        for layer in self.gnn_layers:
            x = layer(x, adj)
            hidden_states.append(x)

        ret = 0.
        for layer, h in enumerate(hidden_states):
            ret = ret + F.dropout(
                self.linear_prediction[layer](h),
                self.dropout_p,
                training=self.training
            )
        return ret

# class GCN(nn.Module):
#     def __init__(self, input_dim, hidden_dims, n_layers, dropout=0.5):
#         super(GCN, self).__init__()
#         self.dropout_p = dropout
#
#         self.gnn_layers = nn.ModuleList()
#         self.linear_prediction = nn.ModuleList()
#         for i in range(n_layers):
#             if i == 0:
#                 self.gnn_layers.append(
#                     GCNConv(input_dim)
#                 )
#                 self.linear_prediction.append(
#                     nn.Sequential(nn.Linear(input_dim, hidden_dims[i]),
#                                   nn.LayerNorm(hidden_dims[i]),
#                                   nn.ReLU(inplace=True)))
#             else:
#                 self.gnn_layers.append(
#                     GCNConv(hidden_dims[i - 1])
#                 )
#                 self.linear_prediction.append(
#                     nn.Sequential(nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
#                                   nn.LayerNorm(hidden_dims[i]),
#                                   nn.ReLU(inplace=True)))
#         self.linear_prediction.append(
#             nn.Sequential(nn.Linear(hidden_dims[-2], hidden_dims[-1]),
#                           nn.LayerNorm(hidden_dims[-1]),
#                           nn.ReLU(inplace=True)))
#
#     def forward(self, x, adj):
#         hidden_states = [x]
#         for layer in self.gnn_layers:
#             x = layer(x, adj)
#             hidden_states.append(x)
#
#         ret = 0.
#         for layer, h in enumerate(hidden_states):
#             ret = ret + F.dropout(
#                 self.linear_prediction[layer](h),
#                 self.dropout_p,
#                 training=self.training
#             )
#         return ret

