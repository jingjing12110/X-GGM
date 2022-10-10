import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(
            torch.FloatTensor(input_dim, output_dim).cuda())
    
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        return y


# a deterministic linear output (update: add noise)
class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms
        
        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        # make edge prediction (reconstruct)
        self.decode_2 = nn.Linear(embedding_size, y_size)
        self.relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).cuda()
        z = eps * z_sgm + z_mu
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms


class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes,
                 pool='sum'):
        """
        :param input_dim: input feature dimension for node.
        :param hidden_dim: hidden dim for 2-layer gcn.
        :param latent_dim: dimension of the latent representation of graph.
        :param max_num_nodes:
        :param pool:
        """
        super(GraphVAE, self).__init__()
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()
        
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.vae = MLP_VAE_plain(input_dim * input_dim, latent_dim, output_dim)
        
        self.max_num_nodes = max_num_nodes
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.pool = pool

    def forward(self, input_features, adj):
        graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes)
        # vae
        h_decode, z_mu, z_lsgms = self.vae(graph_h)
        out = F.sigmoid(h_decode)
        out_tensor = out.cpu().data
        recon_adj_lower = self.recover_adj_lower(out_tensor)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)
    
        # set matching features be degree
        out_features = torch.sum(recon_adj_tensor, 1)
    
        adj_data = adj.cpu().data[0]
        adj_features = torch.sum(adj_data, 1)
    
        S = self.edge_similarity_matrix(adj_data, recon_adj_tensor, adj_features,
                                        out_features,
                                        self.deg_feature_similarity)
    
        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes,
                                     self.max_num_nodes) * init_corr
    
        assignment = self.mpm(init_assignment, S)
        # print('Assignment: ', assignment)
    
        # matching
        # use negative of the assignment score since the alg finds min cost flow
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            -assignment.numpy())
        print('row: ', row_ind)
        print('col: ', col_ind)
        # order row index according to col index
        adj_permuted = adj_data
        adj_vectorized = adj_permuted[torch.triu(
            torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1].squeeze_()
        adj_vectorized_var = Variable(adj_vectorized).cuda()
    
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
        print('recon: ', adj_recon_loss)
        print(adj_vectorized_var)
        print(out[0])
    
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes  # normalize
        print('kl: ', loss_kl)
    
        loss = adj_recon_loss + loss_kl
    
        return loss
    
    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(
            torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj
    
    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag
    
    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                               matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                        sim_func(matching_features[i],
                                                 matching_features_recon[a])
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                            adj_recon[a, b] * adj_recon[a, a] * \
                                            adj_recon[b, b]
        return S
    
    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :])
                              for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x
    
    @staticmethod
    def deg_feature_similarity(f1, f2):
        return 1 / (abs(f1 - f2) + 1)
    
    def permute_adj(self, adj, curr_ind, target_ind):
        """ Permute adjacency matrix. The target_ind (connectivity) should be
        permuted to the curr_ind position.
        """
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted
    
    def pool_graph(self, x):
        if self.pool == 'max':
            x, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
            x = torch.sum(x, dim=1, keepdim=False)
        return x
    
    def forward_test(self, input_features, adj):
        self.max_num_nodes = 4
        adj_data = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data[:4, :4] = torch.FloatTensor(
            [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
        adj_features = torch.Tensor([2, 3, 3, 2])
        
        adj_data1 = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data1 = torch.FloatTensor(
            [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        adj_features1 = torch.Tensor([3, 3, 2, 2])
        S = self.edge_similarity_matrix(adj_data, adj_data1, adj_features,
                                        adj_features1,
                                        self.deg_feature_similarity)
        
        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes,
                                     self.max_num_nodes) * init_corr

        assignment = self.mpm(init_assignment, S)
        # print('Assignment: ', assignment)
        
        # matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            -assignment.numpy())
        print('row: ', row_ind)
        print('col: ', col_ind)
        
        permuted_adj = self.permute_adj(adj_data, row_ind, col_ind)
        print('permuted: ', permuted_adj)
        
        adj_recon_loss = self.adj_recon_loss(permuted_adj, adj_data1)
        print(adj_data1)
        print('diff: ', adj_recon_loss)
    
    @staticmethod
    def adj_recon_loss(adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_truth, adj_pred)

