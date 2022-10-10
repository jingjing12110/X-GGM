# @File  :graph_vae.py
# @Time  :2021/1/16
# @Desc  :
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.gin import GIN


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.5):
        super(MLPDecoder, self).__init__()
        self.dropout = dropout
        self.decode = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim // 2))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x):
        x = F.dropout(x, p=self.dropout)
        x = self.decode(x)
        return x


class GraphVAE(nn.Module):
    def __init__(self, hidden_dim=300):
        super(GraphVAE, self).__init__()
        self.pool = 'max'
        self.act = nn.Sigmoid()
        self.hid_dim = hidden_dim
        
        self.encoder_mu = GIN(input_dim=768, hidden_dims=[300, 300], n_layers=2)
        self.encoder_var = GIN(input_dim=768, hidden_dims=[300, 300], n_layers=2)
        
        self.decoder = MLPDecoder(
            input_dim=hidden_dim,
            hidden_dim=36 * 35
        )
    
    def forward(self, x, adj):
        """
        :param x: [bs, node, 768]
        :param adj: [bs, node, node]
        :return: rec_adj -> [bs, node, node]
        """
        mu = self.encoder_mu(x, adj)  # [bs, node, 300]
        log_var = self.encoder_var(x, adj)  # [bs, node, 300]
        
        mu = self.pool_graph(F.normalize(mu, dim=-1))  # [bs, 300]
        log_var = self.pool_graph(F.normalize(log_var, dim=-1))  # [bs, 300]
        # mu = self.pool_graph(mu)  # [bs, 300]
        # log_var = self.pool_graph(log_var)  # [bs, 300]
       
        z = self.re_parameterize(mu, log_var)  # [bs, 300]
        
        z = self.decoder(z)  # [bs, 630]
        rec_adj = self.recover_adj(self.act(z))  # [bs, 36, 36]
        
        # compute KL divergence
        # mu, log_var = self.act(mu), self.act(log_var)
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / 1296  # 36 *36
        
        temp = torch.triu(torch.ones(z.shape[0], 36, 36).to(z.device)) - \
            torch.eye(36).to(z.device).unsqueeze(0).repeat(z.shape[0], 1, 1, )
        z_true = adj[temp == 1].view(z.shape[0], -1)
        rec_loss = self.adj_recon_loss(z, z_true)
        
        if self.training:
            return rec_adj, kl_loss, rec_loss
        else:
            return rec_adj
    
    @staticmethod
    def recover_adj(l):
        adj = torch.zeros((l.shape[0], 36, 36), dtype=l.dtype).to(l.device)
        adj_temp = torch.triu(torch.ones(l.shape[0], 36, 36).to(l.device))
        adj_temp = adj_temp - torch.eye(36).to(l.device).unsqueeze(
            0).repeat(l.shape[0], 1, 1, )
        adj[adj_temp == 1] = l.view(-1)
        
        adj = adj + adj.transpose(1, 2)
        return adj
    
    @staticmethod
    def adj_recon_loss(adj_pred, adj_truth):
        return F.binary_cross_entropy_with_logits(adj_pred, adj_truth)
    
    def pool_graph(self, x):
        if self.pool == 'max':
            x, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
            x = torch.sum(x, dim=1, keepdim=False)
        return x
    
    def re_parameterize(self, mu, log_var):
        if self.training:
            # eps = torch.randn_like(log_var)
            # return eps.mul(log_var).add_(mu)
            std = torch.exp(log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
