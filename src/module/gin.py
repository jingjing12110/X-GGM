# @File  :gin.py
# @Time  :2021/1/12
# @Desc  :
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxrt.modeling import GeLU


class GINConv(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GeLU(),
            nn.LayerNorm(hidden_dim),
            # nn.ReLU(inplace=True),
        )
    
    def forward(self, X, A):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix

        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        X = X + (1 + self.eps) * A @ X
        X = self.linear(X)
        return X


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_layers, dropout=0.5):
        super().__init__()
        self.dropout_p = dropout
        
        self.gnn_convs = nn.ModuleList()
        self.linear_prediction = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.gnn_convs.append(
                    GINConv(input_dim, hidden_dims[i]))
                self.linear_prediction.append(
                    nn.Sequential(nn.Linear(input_dim, hidden_dims[i]),
                                  GeLU(),
                                  nn.LayerNorm(hidden_dims[i]),
                                  ))
            else:
                self.gnn_convs.append(
                    GINConv(hidden_dims[i - 1], hidden_dims[i]))
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
    
    def forward(self, X, A):
        """
        :param X: [bs, node, dim]
        :param A: [bs, node, node]
        :return:
        """
        hidden_states = [X]
        
        for layer in self.gnn_convs:
            X = layer(X, A)
            hidden_states.append(X)
        
        ret = 0.
        for layer, h in enumerate(hidden_states):
            ret = ret + F.dropout(
                self.linear_prediction[layer](h),
                self.dropout_p,
                training=self.training
            )
        return ret  # B x N x F_out
    

# class GINConv(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.eps = nn.Parameter(torch.zeros(1))
#         self.linear = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, X, A):
#         """
#         Params
#         ------
#         A [batch x nodes x nodes]: adjacency matrix
#         X [batch x nodes x features]: node features matrix
#
#         Returns
#         -------
#         X' [batch x nodes x features]: updated node features matrix
#         """
#         X = X + (1 + self.eps) * A @ X
#         X = self.linear(X)
#         return X
#
#
# class GIN(nn.Module):
#     def __init__(self, input_dim, hidden_dims, n_layers, dropout=0.5):
#         super().__init__()
#         self.dropout_p = dropout
#
#         self.gnn_convs = nn.ModuleList()
#         self.linear_prediction = nn.ModuleList()
#         for i in range(n_layers):
#             if i == 0:
#                 self.gnn_convs.append(
#                     GINConv(input_dim, hidden_dims[i]))
#                 self.linear_prediction.append(
#                     nn.Sequential(nn.Linear(input_dim, hidden_dims[i]),
#                                   nn.LayerNorm(hidden_dims[i]),
#                                   nn.ReLU(inplace=True)))
#             else:
#                 self.gnn_convs.append(
#                     GINConv(hidden_dims[i - 1], hidden_dims[i]))
#                 self.linear_prediction.append(
#                     nn.Sequential(nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
#                                   nn.LayerNorm(hidden_dims[i]),
#                                   nn.ReLU(inplace=True)))
#
#         self.linear_prediction.append(
#             nn.Sequential(nn.Linear(hidden_dims[-2], hidden_dims[-1]),
#                           nn.LayerNorm(hidden_dims[-1]),
#                           nn.ReLU(inplace=True)))
#
#     def forward(self, X, A):
#         """
#         :param X: [bs, node, dim]
#         :param A: [bs, node, node]
#         :return:
#         """
#         hidden_states = [X]
#
#         for layer in self.gnn_convs:
#             X = layer(X, A)
#             hidden_states.append(X)
#
#         ret = 0.
#         for layer, h in enumerate(hidden_states):
#             ret = ret + F.dropout(
#                 self.linear_prediction[layer](h),
#                 self.dropout_p,
#                 training=self.training
#             )
#         return ret  # B x N x F_out


