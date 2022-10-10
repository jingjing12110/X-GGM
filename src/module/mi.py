# @File :mi.py
# @Time :2021/7/2
# @Desc :
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        """Mutual Information I(X,Y) Contrastive Learning Upper Bound
        :param x_dim:
        :param y_dim:
        :param hidden_size: the dimension of the hidden layer of the
        approximation network q(Y|X)
        """
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )
    
    def forward(self, x_samples, y_samples):
        # [sample_size, x_dim/y_dim]
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()
        
        prediction_1 = mu.unsqueeze(1)  # [sample_size, 1, dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1, sample_size, dim]
        
        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(
            dim=1) / 2. / logvar.exp()
        
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound


class CLUBSample(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        """Sampled version of the CLUB.
        """
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )
    
    def forward(self, x_samples, y_samples):
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.


class CLUBv2(nn.Module):
    def __init__(self, beta=1e-3):
        """using in InfoBERT
        """
        super(CLUBv2, self).__init__()
        self.beta = beta
    
    @staticmethod
    def mi_est_org(y_samples):
        positive = torch.zeros_like(y_samples)
        
        # [sample_size, 1, dim]
        # prediction_1 = y_samples.unsqueeze(1)
        # [1, sample_size, dim]
        # y_samples_1 = y_samples.unsqueeze(0)
        # [sample_size, dim]
        negative = - ((y_samples.unsqueeze(0) - y_samples.unsqueeze(1)
                       ) ** 2).mean(dim=1) / 2.
        
        # upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    
    @staticmethod
    def mi_est_sample(y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()
        # random_index = torch.randperm(sample_size).long()
        
        positive = torch.zeros_like(y_samples)
        negative = - (y_samples - y_samples[random_index]) ** 2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound
    
    def update(self, y_samples, steps=None, mi_mode=None):
        # no performance improvement, not enabled
        # if steps:
        #     # beta anealing
        #     beta = self.beta if steps > 1000 else self.beta * steps / 1000
        # else:
        #     beta = self.beta
        
        return self.mi_est_org(y_samples) * self.beta


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x_samples, y_samples):
        # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        y_shuffle = y_samples[random_index]
        
        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))
        
        lower_bound = T0.mean() - torch.log(T1.exp().mean())
        
        # compute the negative loss (maximise loss == minimise -loss)
        return -lower_bound


class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        
        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        # shape [sample_size, sample_size, 1]
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.
        
        lower_bound = T0.mean() - (
                T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return -lower_bound


class VarUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        """variational upper bound
        """
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )
    
    def forward(self, x_samples, y_samples):  # [sample_size, 1]
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)
        return 1. / 2. * (mu ** 2 + logvar.exp() - 1. - logvar).mean()


class L1OutUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        """naive upper bound
        """
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )
    
    @staticmethod
    def log_sum_exp(value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        import math
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            if isinstance(sum_exp, Number):
                return m + math.log(sum_exp)
            else:
                return m + torch.log(sum_exp)
    
    def forward(self, x_samples, y_samples):
        batch_size = y_samples.shape[0]
        
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)
        
        # [sample_size]
        positive = (- (mu - y_samples) ** 2 / 2. / logvar.exp() - logvar / 2.
                    ).sum(dim=-1)
        
        # [sample_size, 1, dim]
        mu_1 = mu.unsqueeze(1)
        logvar_1 = logvar.unsqueeze(1)
        # [1, sample_size, dim]
        y_samples_1 = y_samples.unsqueeze(0)
        # [sample_size, sample_size]
        all_probs = (- (y_samples_1 - mu_1
                        ) ** 2 / 2. / logvar_1.exp() - logvar_1 / 2.).sum(dim=-1)
        
        diag_mask = torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        # [sample_size]
        
        negative = self.log_sum_exp(all_probs + diag_mask, dim=0) - np.log(
            batch_size - 1.)
        
        return (positive - negative).mean()


class InfoNCE(nn.Module):
    def __init__(self, x_dim=768, y_dim=768, hidden_size=300):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
    
    def forward(self, x_samples, y_samples):
        # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples[random_index].unsqueeze(1).repeat((1, sample_size, 1))
        
        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(
            torch.cat([x_tile, y_tile], dim=-1))  # [s_size, s_size, 1]
        
        lower_bound = T0.mean() - (
                T1.logsumexp(dim=1).mean() - np.log(sample_size))
        
        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound


# ******************************************************************************
# Modifying for CIB
# ******************************************************************************
class MIUpperBound(nn.Module):
    def __init__(self, variational=False,
                 x_dim=None, y_dim=None, hidden_size=None):
        """Modified version of CLUBv2
        """
        super(MIUpperBound, self).__init__()
        self.variational = variational
        if self.variational:
            # p(y|x) is unknown, using variational approximation
            self.p_mu = nn.Sequential(
                nn.Linear(x_dim, hidden_size // 2),
                # nn.ReLU(),
                nn.GELU(),
                nn.Linear(hidden_size // 2, y_dim)
            )
            
            self.p_logvar = nn.Sequential(
                nn.Linear(x_dim, hidden_size // 2),
                # nn.ReLU(),
                nn.GELU(),
                nn.Linear(hidden_size // 2, y_dim),
                nn.Tanh()
            )
    
    def mi_est_sample_variational(self, x_samples, y_samples):
        # [sample_size, x_dim/y_dim]
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.
    
    def mi_est_org_variational(self, x_samples, y_samples):
        mu, logvar = self.p_mu(x_samples), self.p_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()
        
        prediction_1 = mu.unsqueeze(1)  # [sample_size, 1, dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1, sample_size, dim]
        
        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(
            dim=1) / 2. / logvar.exp()
        
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound
    
    def variational_update(self, x_samples, y_samples, mi_mode='original'):
        if mi_mode == 'original':
            return self.mi_est_org_variational(
                x_samples, y_samples)
        else:
            return self.mi_est_sample_variational(
                x_samples, y_samples)
    
    @staticmethod
    def mi_est_sample(y_samples):
        # mu, logvar = 0., 0.
        sample_size = y_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        positive = torch.zeros_like(y_samples)
        negative = - (y_samples - y_samples[random_index]) ** 2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound
    
    @staticmethod
    def mi_est_org(y_samples):
        positive = torch.zeros_like(y_samples)
        
        # [bs, sample_size, 1, dim]
        prediction_1 = y_samples.unsqueeze(dim=-2)
        # [bs, 1, sample_size, dim]
        y_samples_1 = y_samples.unsqueeze(dim=-3)
        # [sample_size, dim]
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=-2) / 2.
        
        # upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    
    def update(self, y_samples, mi_mode='original'):
        if mi_mode == 'original':
            return self.mi_est_org(y_samples)
        else:
            return self.mi_est_sample(y_samples)


class InfoNCEv2(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCEv2, self).__init__()
        # joint distribution: p(x, y)
        self.F_func = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
    
    def forward(self, x_samples, y_samples):
        # samples have shape [sample_size, dim]
        sample_size = y_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        # [sample_size, 1]
        t0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        # [sample_size, sample_size, 1]
        t1 = self.F_func(torch.cat([
            x_samples.unsqueeze(0).repeat((sample_size, 1, 1)),
            y_samples[random_index].unsqueeze(1).repeat((1, sample_size, 1))
        ], dim=-1)
        ).squeeze()
        
        lower_bound = np.log(sample_size) + t0.mean() - t1.sum(1).mean(0)
        return -lower_bound


# Auxiliary network for mutual information estimation
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1),
        )
    
    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(
            neg).mean(), pos.mean() - neg.exp().mean() + 1


class MVMIEstimator(nn.Module):
    def __init__(self, hidden_size=384, x1_dim=768, x2_dim=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.mi_estimator = MIEstimator(x1_dim // 2, x2_dim // 2)
    
    def forward(self, p_z1_given_x1_samples, p_z2_given_x2_samples):
        mu, sigma = p_z1_given_x1_samples[:, :self.hidden_size], \
                    p_z1_given_x1_samples[:, self.hidden_size:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        # a factorized Normal distribution
        p_z1_given_x1 = Independent(Normal(loc=mu, scale=sigma), 1)
        
        mu, sigma = p_z2_given_x2_samples[:, :self.hidden_size], \
                    p_z2_given_x2_samples[:, self.hidden_size:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        # a factorized Normal distribution
        p_z2_given_x2 = Independent(Normal(loc=mu, scale=sigma), 1)
        
        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_x1.rsample()
        z2 = p_z2_given_x2.rsample()
        
        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        # mi_estimation = mi_estimation.mean()
        
        # Symmetrized Kullback-Leibler divergence
        kl_1_2 = p_z1_given_x1.log_prob(z1) - p_z2_given_x2.log_prob(z1)
        kl_2_1 = p_z2_given_x2.log_prob(z2) - p_z1_given_x1.log_prob(z2)
        d_skl_2 = (kl_1_2 + kl_2_1).mean()
        
        return d_skl_2 - 2 * mi_gradient
    
    def compute_2variable_mi(self, p_z1_given_x1_samples, p_z2_given_x2_samples):
        mu, sigma = p_z1_given_x1_samples[:, :self.hidden_size], \
                    p_z1_given_x1_samples[:, self.hidden_size:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        # a factorized Normal distribution
        p_z1_given_x1 = Independent(Normal(loc=mu, scale=sigma), 1)
        
        mu, sigma = p_z2_given_x2_samples[:, :self.hidden_size], \
                    p_z2_given_x2_samples[:, self.hidden_size:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        # a factorized Normal distribution
        p_z2_given_x2 = Independent(Normal(loc=mu, scale=sigma), 1)
        
        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_x1.rsample()
        z2 = p_z2_given_x2.rsample()
        
        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        # mi_estimation = mi_estimation.mean()
        return mi_gradient


class JointMIEstimator(nn.Module):
    def __init__(self, hidden_size=384, x1_dim=768, x2_dim=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # self.mi_estimator = MIEstimator(x1_dim // 2, x2_dim // 2)
        self.mi_estimator = InfoNCE(x1_dim // 2, x2_dim // 2)
        
        self.fc_l = nn.Linear(20, 1)
        self.fc_v = nn.Linear(36, 1)
    
    def forward(self, zl, zv):
        z1 = self.fc_l(zl.permute(0, 2, 1)).squeeze()
        z2 = self.fc_v(zv.permute(0, 2, 1)).squeeze()
        
        mu, sigma = z1[:, :self.hidden_size], z1[:, self.hidden_size:]
        # Make sigma always positive
        sigma = softplus(sigma) + 1e-7
        # a factorized Normal distribution
        p_z1_given_x1 = Independent(Normal(loc=mu, scale=sigma), 1)
        
        mu, sigma = z2[:, :self.hidden_size], z2[:, self.hidden_size:]
        sigma = softplus(sigma) + 1e-7
        # a factorized Normal distribution
        p_z2_given_x2 = Independent(Normal(loc=mu, scale=sigma), 1)
        
        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_x1.rsample()  # [bs, 384]
        z2 = p_z2_given_x2.rsample()  # [bs, 384]
        
        # Symmetrized Kullback-Leibler divergence
        kl_1_2 = p_z1_given_x1.log_prob(z1) - p_z2_given_x2.log_prob(z1)
        kl_2_1 = p_z2_given_x2.log_prob(z2) - p_z1_given_x1.log_prob(z2)
        d_skl = (kl_1_2 + kl_2_1).mean() / 2.
        
        # Mutual information estimation
        # mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_estimation = self.mi_estimator(z1, z2)
        if mi_estimation > 0:
            print("infonce is positive")
        
        return d_skl - mi_estimation


def compute_skl_divergence(xl, xv):
    """样本个数可以不一样
    :param xl: [num_l, dim]
    :param xv: [num_v, dim]
    :return:
    """
    kl_1_2 = naive_estimator(xl, xv)
    kl_2_1 = naive_estimator(xv, xl)
    return kl_1_2 + kl_2_1


def naive_estimator(s1, s2, k=1):
    """ KL-Divergence estimator using brute-force (numpy) k-NN
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
    """
    assert (len(s1.shape) == len(s2.shape) == 2)
    assert (s1.shape[1] == s2.shape[1])
    
    n, m = len(s1), len(s2)
    D = np.log(m / (n - 1))
    d = float(s1.shape[1])
    
    # combined knn_distance
    # [n, m, 768]
    nu_norms = s2.unsqueeze(0).repeat(n, 1, 1) - s1.unsqueeze(1)
    nu_norms = torch.norm(nu_norms, dim=-1, p=2)
    nu_norms = torch.sort(nu_norms, dim=-1)[0][:, k - 1]  # [n]
    # [n, n, 768]
    rho_norms = s1.unsqueeze(1).repeat(1, n, 1) - s1.unsqueeze(0)
    rho_norms = torch.norm(rho_norms, dim=-1, p=2)
    rho_norms = torch.sort(rho_norms, dim=-1)[0][:, k]  # [n]
    
    D += (d / n) * torch.sum(torch.log(nu_norms / rho_norms))
    return D


def knn_distance(point, sample, k):
    """ Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample` """
    norms = torch.norm(sample - point, dim=1, p=2)  # [288]
    return torch.sort(norms)[0][k]

# def compute_skl_divergence(x, y):
#     px = F.softmax(x, dim=-1)
#     log_px = F.log_softmax(x, dim=-1)
#     py = F.softmax(y, dim=-1)
#     log_py = F.log_softmax(y, dim=-1)
#     d_skl_2 = F.kl_div(log_px, py, reduction='none') + F.kl_div(
#         log_py, px, reduction='none')
#     return d_skl_2.mean()
