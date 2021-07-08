# This code is modified from https://github.com/jakesnell/prototypical-networks

from methods.ours import euclidean_distance
from torch.functional import einsum
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils


class GMMProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, finetune=False, gaussian='mixture'):
        """gaussian =
        mixture: Use Gaussian mixture to form a class distribution
        single: Use one Gaussian for a class
        """
        super(GMMProtoNet, self).__init__(model_func,  n_way, n_support)
        assert gaussian in ['mixture', 'single']
        self.mixture = gaussian == 'mixture'

        self.loss_fn = nn.CrossEntropyLoss()

        self.finetune = finetune

        self.vars = nn.parameter.Parameter(torch.Tensor(512))
        # nn.init.constant_(self.vars, 0.1)
        nn.init.normal_(self.vars, mean=0.1, std=0.01)

    def set_forward(self, x, is_feature=False):
        # (n_way, n_support), (n_way, n_query)
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous().view(self.n_way*self.n_query, -1)

        center_loss = euclidean_distance(z_support,
            z_support.mean(dim=1, keepdim=True).expand(-1, self.n_support, -1), dim=-1).mean()

        if self.mixture:
            scores = self.predict_prob(z_query, z_support)
        else:
            z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)
            scores = self.predict_class_prob_single(z_query, z_proto, z_support.var(dim=1))

        if self.training:
            return scores, center_loss
        else:
            return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores, center_loss = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)

        return loss + center_loss

    def predict_class_prob_single(self, z_query, mu, var):
        """Use only one Gaussian to form a class distribution
        """
        zq = z_query.unsqueeze(1).expand(-1, self.n_way, -1)
        mus = mu.unsqueeze(0).expand(len(z_query), -1, -1)
        z = (zq - mus).unsqueeze(2) # (n_query, n_way, 1, D)
        ivar = torch.stack([torch.diag(1/v) for v in var]) # (n_way, D, D)

        scores = []
        N = z_query.size(0)
        for i in range(self.n_way):
            x = torch.bmm(z[:,i], ivar[i].unsqueeze(0).expand(N, -1, -1)) # (n_query, 1, D)
            if torch.isnan(x).sum():
                print('a', x)
                exit()
            score = torch.exp(-1/2 * torch.bmm(x, z[:,i].transpose(1, 2))).squeeze()
            if torch.isnan(score).sum():
                print('b', score)
                exit()
            scores.append(score)

        return torch.stack(scores).transpose(0, 1)

    def predict_class_prob(self, z_query, z_support):
        """Use Gaussian mixture to form a class distribution
        """
        N = len(z_query)
        z_q = z_query.unsqueeze(1).expand(-1, self.n_support, -1)
        z_s = z_support.unsqueeze(0).expand(N, -1, -1)
        z = z_q - z_s # (n_query, n_support, D)
        var = torch.diag(1/self.vars) # (D, D)

        x = torch.bmm(z, var.unsqueeze(0).expand(N, -1, -1)) # (n_query, n_support, D)
        e = torch.exp(-1/2 * (x * z).sum(dim=2)) # (n_query, n_support)

        # Ignore the following line because it is constant if the variance is shared
        # p = (2*np.pi)**(-512/2) * 1/torch.sqrt(torch.diag(self.vars).det()) * e

        p = e.mean(dim=1)

        return p # (n_query)

    def predict_prob(self, z_query, z_support):
        x = torch.stack([self.predict_class_prob(z_query, zs) for zs in z_support])
        return x.transpose(0, 1)


def distance(x, y, method='l2'):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if method == 'l2':
        return euclidean_distance(x, y)
    elif method == 'vec':
        return (x - y)
    elif method == 'cos':
        return -cos_similarity(x, y)
    else:
        raise NotImplementedError(method)


def euclidean_distance(x, y, dim=-1):
    return torch.pow(x-y, 2).sum(dim)


def cos_similarity(x, y, dim=-1):
    return nn.functional.cosine_similarity(x, y, dim=dim)
