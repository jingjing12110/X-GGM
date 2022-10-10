# @File  :edge_gnn.py
# @Time  :2021/1/13
# @Desc  :
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.gin import GIN
from module.graph_utils import node_feature_to_matrix


class MLP(nn.Module):
    def __init__(self, dims, n_layers, use_bn=True):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        assert len(dims) == (self.n_layers + 1)
        
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_bn:
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU())
                )
            else:
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU())
                )
    
    def forward(self, x):
        """
        :param x: [bs, *, dim]
        :return:
        """
        for layer in range(self.n_layers):
            x = self.mlp_layers[layer](x)
        return x


class MultiConv1x1(nn.Module):
    def __init__(self, channels, n_layers, use_bn=True):
        super(MultiConv1x1, self).__init__()
        self.n_layers = n_layers
        assert len(channels) == (self.n_layers + 1)
        
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_bn:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True))
                )
            else:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=1),
                    nn.ReLU(inplace=True))
                )
    
    def forward(self, x):
        """
        :param x: [bs, C, W, H]
        :return:
        """
        for layer in range(self.n_layers):
            x = self.conv_layers[layer](x)
        return x


class EdgeGNN(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout=0.5):
        super(EdgeGNN, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers
        
        # encoding node feature and edge adjacency matrix
        self.gnn_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GIN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=2))
            self.edge_layers.append(nn.Linear(
                2 * hidden_dim, 1))
            
    def forward(self, x, adj):
        """
        :param x: [bs, n_node, hidden_dim]
        :param adj: [bs, n_node, n_node]
        :return:
        """
        n_node = x.shape[1]
        hidden_x, hidden_adj = [x], [adj]
        
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            hidden_x.append(x)
            
            adj = node_feature_to_matrix(
                x)  # [bs, n_node, n_node, 2hidden_dim]
            adj = self.edge_layers[layer](adj.view(
                -1, n_node * n_node, adj.shape[-1]))
            adj = adj.squeeze().view(-1, n_node, n_node)
            hidden_adj.append(adj)
        
        x = torch.stack(hidden_x, dim=-2).sum(dim=-2)
        adj = torch.stack(hidden_adj, dim=1).sum(dim=1)
        return x, adj


class EdgeConvGNN(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout=0.5,
                 node_norm=False, gpool=False):
        super(EdgeConvGNN, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers
        self.node_norm = node_norm
        self.gpool = gpool
        if self.gpool:
            self.act = nn.Tanh()
        
        # encoding node feature and edge adjacency matrix
        self.gnn_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GIN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=2))
            self.edge_layers.append(nn.Sequential(
                nn.Linear(2 * hidden_dim, 2),
                nn.Tanh()))
            self.conv_layers.append(MultiConv1x1([4, 1], 1))
    
    def forward(self, x, adj):
        """
        :param x: [bs, n_node, hidden_dim]
        :param adj: [bs, n_node, n_node]
        :return:
        """
        n_node = x.shape[1]
        hidden_x, hidden_adj = [x], [adj]
        
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            hidden_x.append(x)
            
            adj_node = node_feature_to_matrix(
                x)  # [bs, n_node, n_node, 2hidden_dim -> 2]
            adj_node = self.edge_layers[layer](adj_node.view(
                -1, n_node * n_node, adj_node.shape[-1]))
            if self.node_norm:
                adj_node = F.normalize(adj_node)
            adj_node = adj_node.view(-1, n_node, n_node, 2).permute(0, 3, 1, 2)
            
            # adjacency Conv
            adj = torch.cat([adj.unsqueeze(1), 1. - adj.unsqueeze(1),
                             adj_node], dim=1)  # [bs, 4, N, N]
            # adj_feat.append(adj)
            adj = self.conv_layers[layer](adj).squeeze()
            
            hidden_adj.append(adj)
        
        x = torch.stack(hidden_x, dim=-2).sum(dim=-2)
        adj = torch.stack(hidden_adj, dim=1).sum(dim=1)
        
        return x, adj
    
    def graph_pool(self, x):
        return self.act(x)
