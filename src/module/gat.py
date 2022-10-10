import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    def __init__(self, dim_input, dim_hidden, dropout=0.5,
                 alpha=0.2, concat=True):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.dim_hidden = dim_hidden

        self.linear_layer = nn.Linear(dim_input, dim_hidden, bias=False)
        self.attn_layer = nn.Linear(2 * dim_hidden, 1, bias=False)
        self.reset_parameters()

        self.leaky_relu = nn.LeakyReLU(alpha)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_layer.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_layer.weight, gain=gain)

    def forward(self, x, adj):
        """
        @param x: [bs, num_node, feat_dim]
        @param adj: [bs, num_node, num_node]
        @return: [bs, num_node, feat_dim]
        """
        bs, N, _ = x.shape
        h = self.linear_layer(x)  # [bs, N, dim_hidden]

        h_self = h.repeat(1, 1, N).view(bs, N * N, -1)
        h_neighbor = h.repeat(1, N, 1)
        a_input = torch.cat([h_self, h_neighbor], dim=2).view(
            bs, N, -1, 2 * self.dim_hidden)
        # [bs, N, N]
        attention = self.leaky_relu(self.attn_layer(a_input)).squeeze(dim=-1)

        # Masked Attention
        attention = attention.masked_fill(adj == 0, -9e15)
        attention = F.softmax(attention, dim=-1)  # [bs, N, N]

        # [bs, N, C]
        if self.concat:
            return F.elu(torch.bmm(attention, h))
        else:
            return torch.bmm(attention, h)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_head, dropout=0.5,
                 alpha=0.2, merge='cat'):
        """
        @param input_dim: [num_node, in_dim]
        @param hidden_dim: [num_node, hidden_dim]
        @param dropout:
        @param alpha:
        @param n_head:
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.merge = merge

        self.gat_layers = nn.ModuleList()
        for _ in range(n_head):
            self.gat_layers.append(GATConv(
                input_dim, hidden_dim,
                dropout=dropout, alpha=alpha, concat=True))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.merge == 'cat':
            x = torch.cat([att(x, adj) for att in self.gat_layers], dim=2)
        else:
            x = torch.mean(torch.stack([att(x, adj) for att in self.gat_layers]))

        return x

