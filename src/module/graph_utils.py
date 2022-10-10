# @Github : https://github.com/ermongroup/GraphScoreMatching
import numpy as np
import torch


def node_feature_to_matrix(x):
    """
    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(-1, -1, x.size(1), -1)  # BS x N x N x F
    x_b_t = x_b.transpose(1, 2)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b_t], dim=-1)  # BS x N x N x 2F
    return x_pair


def mask_adjs(adjs, node_flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param node_flags: B x N
    :return:
    """
    if len(adjs.shape) == 4:
        node_flags = node_flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * node_flags.unsqueeze(-1)
    adjs = adjs * node_flags.unsqueeze(-2)
    return adjs


def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])],
                       axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])],
                       axis=0)
    return a


def get_corrupt_k(min_k=0, max_k=None, p=0.5):
    ret = np.random.geometric(p) + min_k - 1
    if max_k is not None:
        ret = min(ret, max_k)
    return ret


def remove_self_loop_if_exists(adjs):
    return (adjs - torch.eye(adjs.size()[-1]).unsqueeze(0).to(adjs.device)
            ).clamp(min=0.0)


def add_self_loop_if_not_exists(adjs):
    if len(adjs.shape) == 4:
        return adjs + torch.eye(adjs.size()[-1]
                                ).unsqueeze(0).unsqueeze(0).to(adjs.device)
    return adjs + torch.eye(adjs.size()[-1]).unsqueeze(0).to(adjs.device)


def toggle_edge_np(adj, count=1):
    """
    uniformly toggle `count` edges of the graph,
    suppose that the vertex number is fixed

    :param adj: N x N
    :param count: int
    :return: new adjs and node_flags
    """
    count = min(count, adj.shape[-1])
    x = np.random.choice(adj.shape[0], count)
    y = np.random.choice(adj.shape[1], count)
    change = 1. - adj[x, y]
    adj[x, y] = change
    adj[y, x] = change
    return adj


def check_adjs_symmetry(adjs, do_check_adjs_symmetry=False):
    if not do_check_adjs_symmetry:
        return
    tr_adjs = adjs.transpose(-1, -2)
    assert (adjs - tr_adjs).abs().sum([0, 1, 2]) < 1e-2


def gen_list_of_data(train_x_b, train_adj_b, sigma_list):
    """
    :param train_x_b: [bs, N, F_in], batch of feature vectors of nodes
    :param train_adj_b: [bs, N, N], batch of original adjacency matrices
    :param sigma_list: list of noise levels
    :return:
        train_x_b: [len(sigma_list) * bs, N, F_in],
            batch of feature vectors of nodes (w.r.t. `train_noise_adj_b`)
        train_noise_adj_b: [len(sigma_list) * bs, N, N],
            batch of perturbed adjacency matrices
        train_node_flag_b: [len(sigma_list) * bs, N], the flags for
            the existence of nodes (w.r.t. `train_noise_adj_b`)
        grad_log_q_noise_list: [len(sigma_list) * bs, N, N],
            the ground truth gradient (w.r.t. `train_noise_adj_b`)
    """
    assert isinstance(sigma_list, list)
    train_noise_adj_b_list = []
    grad_log_q_noise_list = []
    for sigma_i in sigma_list:
        train_noise_adj_b, grad_log_q_noise = add_gaussian_noise(
            train_adj_b,
            sigma=sigma_i
        )
        train_noise_adj_b_list.append(train_noise_adj_b)
        grad_log_q_noise_list.append(grad_log_q_noise)
    
    train_noise_adj_b = torch.cat(train_noise_adj_b_list, dim=0)
    train_x_b = train_x_b.repeat(len(sigma_list), 1, 1)
    
    return train_x_b, train_noise_adj_b, grad_log_q_noise_list


def add_gaussian_noise(adjs, sigma, is_half=False):
    assert isinstance(adjs, torch.Tensor)
    noise = torch.randn_like(adjs).triu(diagonal=1) * sigma
    if is_half:
        noise = noise.abs()
    noise_s = noise + noise.transpose(-1, -2)
    check_adjs_symmetry(noise_s)
    grad_log_noise = - noise_s / (sigma ** 2)
    ret_adjs = adjs + noise_s
    
    return ret_adjs, grad_log_noise


def add_feature_noise(feats, sigma=0.2):
    assert isinstance(feats, torch.Tensor)
    ret_feats, grad_log_noise = [], []
    for i in range(36):
        noise = torch.randn_like(feats[:, i, :]) * sigma
        grad_log_noise.append(- noise / (sigma ** 2))
        ret_feats.append(feats[:, i, :] + noise)
    return torch.stack(ret_feats, dim=1), torch.stack(grad_log_noise, dim=1)


def add_feature_noise_v2(feats, sigma=0.2):
    assert isinstance(feats, torch.Tensor)
    noise = torch.randn_like(feats) * sigma
    feats = feats + noise
    grad_log_noise = - noise / (sigma ** 2)
    return feats, grad_log_noise


def add_edge_noise(adjs, sigma=0.2):
    noise = torch.zeros_like(adjs)
    temp = torch.randn((adjs.shape[0], 630), device=adjs.device) * sigma
    noise[torch.ones_like(adjs).triu(1) == 1] = temp.view(-1)
    noise = noise + noise.transpose(-1, -2)
    grad_log_noise = - noise / (sigma ** 2)
    adjs = adjs + noise
    return adjs, grad_log_noise


def add_edge_noise_v2(adjs, sigma=0.2):
    assert isinstance(adjs, torch.Tensor)
    noise = torch.randn_like(adjs).triu(diagonal=1) * sigma
    noise = noise + noise.transpose(-1, -2)
    grad_log_noise = - noise / (sigma ** 2)
    adjs = adjs + noise
    return adjs, grad_log_noise

# def add_edge_noise(adjs, sigma=0.2):
#     noise = torch.zeros_like(adjs)
#     temp = torch.randn((adjs.shape[0], 630), device=adjs.device) * sigma
#     noise[torch.ones_like(adjs).triu(1) == 1] = temp.view(-1)
#
#     noise = noise + noise.transpose(-1, -2)
#     grad_log_noise = - noise / (sigma ** 2)
#     adjs = adjs + noise
#     adjs = torch.div(adjs, adjs.max(1)[0].unsqueeze(-1))
#     return adjs, grad_log_noise
