# @File  :graph_generative_modeling.py
# @Time  :2021/1/26
# @Desc  :
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.gin import GIN
from module.gcn import GCN
from module.gat import GAT

from lxrt.modeling import GeLU


class GinPlainEncoder(nn.Module):
    def __init__(self, hidden_dim, n_layers=2, dropout=0.5):
        super(GinPlainEncoder, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GIN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=1))
    
    def forward(self, x, adj):
        """
        :param x: [bs, n_node, hidden_dim]
        :param adj: [bs, n_node, n_node]
        :return:
        """
        # hidden_x = [x]
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            # hidden_x.append(x)
        # x = torch.stack(hidden_x, dim=-2).sum(dim=-2)
        # return hidden_x[-1]
        return x
    

class GCNPlainEncoder(nn.Module):
    def __init__(self, hidden_dim, n_layers=2, dropout=0.5):
        super(GCNPlainEncoder, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers

        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GCN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=1))

    def forward(self, x, adj):
        """
        :param x: [bs, n_node, hidden_dim]
        :param adj: [bs, n_node, n_node]
        :return:
        """
        # hidden_x = [x]
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            # hidden_x.append(x)
        # x = torch.stack(hidden_x, dim=-2).sum(dim=-2)
        # return hidden_x[-1]
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            GeLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))


class DiscriminatorV2(nn.Module):
    def __init__(self, hidden_dim):
        super(DiscriminatorV2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))
    
    
class EdgeGenerator(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout=0.5):
        super(EdgeGenerator, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GIN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=1,
                dropout=self.dropout_p))
    
    def forward(self, x, adj):
        """
        :param x: [bs, n_node, hidden_dim]
        :param adj: [bs, n_node, n_node] sampling from noise
        :return:
        """
        # hidden_adj = [adj]
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            
            adj = torch.bmm(x, x.transpose(1, 2))
            adj = torch.div(adj, adj.max(dim=1)[0].unsqueeze(-1))
            adj = adj.triu(1) + adj.tril(-1)
            # hidden_adj.append(adj)
        # adj = torch.stack(hidden_adj, dim=1).sum(dim=1)
        # return hidden_adj[-1]
        return adj


class NodeGenerator(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout=0.5):
        super(NodeGenerator, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GIN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=1,
                dropout=self.dropout_p))
    
    def forward(self, x, adj):
        """
        :param x: [bs, n_node, hidden_dim]  sampling from noise
        :param adj: [bs, n_node, n_node]
        :return:
        """
        # hidden_x = [x]
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            # hidden_x.append(x)
        # x = torch.stack(hidden_x, dim=-2).sum(dim=-2)
        # return hidden_x[-1]
        return x


class GINGenerator(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout=0.5):
        super(GINGenerator, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers
        self.act = nn.Sigmoid()
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GIN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=1,
                dropout=self.dropout_p))
    
    def forward(self, x, adj):
        """ (x or adj) sampling from noise
        :param x: [bs, n_node, hidden_dim]
        :param adj: [bs, n_node, n_node]
        :return:
        """
        # hidden_x, hidden_adj = [x], [adj]
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            # hidden_x.append(x)
            
            adj = torch.bmm(x, x.transpose(1, 2))
            adj = torch.div(adj, adj.max(dim=1)[0].unsqueeze(-1))
            adj = self.act(adj)
            adj = adj.triu(1) + adj.tril(-1)
            # hidden_adj.append(adj)
        # adj = torch.stack(hidden_adj, dim=1).sum(dim=1)
        # x = torch.stack(hidden_x, dim=-2).sum(dim=-2)
        # return hidden_x[-1], hidden_adj[-1]
        return x, adj
    
    
class GCNGenerator(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout=0.5):
        super(GCNGenerator, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers
        self.act = nn.Sigmoid()
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GCN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=2,
                dropout=self.dropout_p))
    
    def forward(self, x, adj):
        """ (x or adj) sampling from noise
        :param x: [bs, n_node, hidden_dim]
        :param adj: [bs, n_node, n_node]
        :return:
        """
        # hidden_x, hidden_adj = [x], [adj]
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            # hidden_x.append(x)
            
            adj = torch.bmm(x, x.transpose(1, 2))
            adj = torch.div(adj, adj.max(dim=1)[0].unsqueeze(-1))
            adj = self.act(adj)
            adj = adj.triu(1) + adj.tril(-1)
            # hidden_adj.append(adj)
        # adj = torch.stack(hidden_adj, dim=1).sum(dim=1)
        # x = torch.stack(hidden_x, dim=-2).sum(dim=-2)
        # return hidden_x[-1], hidden_adj[-1]
        return x, adj


class GATGenerator(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout=0.5):
        super(GATGenerator, self).__init__()
        self.dropout_p = dropout
        self.n_layers = n_layers
        self.act = nn.Sigmoid()
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GAT(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim, n_head=2)
            )
    
    def forward(self, x, adj):
        """ (x or adj) sampling from noise
        :param x: [bs, n_node, hidden_dim]
        :param adj: [bs, n_node, n_node]
        :return:
        """
        # hidden_x, hidden_adj = [x], [adj]
        for layer in range(self.n_layers):
            x = self.gnn_layers[layer](x, adj)
            # hidden_x.append(x)
            
            adj = torch.bmm(x, x.transpose(1, 2))
            adj = torch.div(adj, adj.max(dim=1)[0].unsqueeze(-1))
            adj = self.act(adj)
            adj = adj.triu(1) + adj.tril(-1)
            # hidden_adj.append(adj)
        # adj = torch.stack(hidden_adj, dim=1).sum(dim=1)
        # x = torch.stack(hidden_x, dim=-2).sum(dim=-2)
        # return hidden_x[-1], hidden_adj[-1]
        return x, adj
    
    
class MixGenerator(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout=0.5):
        super(MixGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 6 * hidden_dim),
            nn.LayerNorm(6 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(6 * hidden_dim, 36 * hidden_dim),
        )
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GIN(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                n_layers=1,
                dropout=dropout))
    
    def forward(self, x, adj, obj_feats):
        """ (x or adj) sampling from noise
        :param x: [bs, hidden_dim]
        :param adj: [bs, n_node, n_node]
        :param obj_feats [bs, n_node, hidden_dim]
        :return:
        """
        node_feats, kl_div_loss = self.generate_node(x)
        rec_loss = F.binary_cross_entropy_with_logits(
            node_feats, obj_feats) * 768
        hidden_x = [node_feats]
        for layer in range(self.n_layers):
            node_feats = self.gnn_layers[layer](node_feats, adj)
            hidden_x.append(node_feats)
        return hidden_x[-1], rec_loss + kl_div_loss
    
    def generate_node(self, x):
        mu, log_var = self.fc1(x), self.fc2(x)
        z = self.re_parameterize(mu, log_var)
        
        z = self.decoder(z).view(-1, 36, self.hidden_dim)
        kl_div_loss = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return z, kl_div_loss
    
    @staticmethod
    def re_parameterize(mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
