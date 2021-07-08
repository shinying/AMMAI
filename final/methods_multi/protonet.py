# This code is modified from https://github.com/jakesnell/prototypical-networks

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils


class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, feat=0):
        """feat
        0: vanilla ProtoNet
        1: self-attention
        2: weighted mean
        3: residual similarity
        4: pushing away
        5: center loss
        """

        super(ProtoNet, self).__init__(model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.feat = feat

        if self.feat == 1:
            self.down = nn.Linear(512, 128)
            self.fc_mu = nn.Linear(128, 128)
            self.fc_var = nn.Linear(128, 128)
            self.up = nn.Linear(128, 512)

            self.attention = nn.MultiheadAttention(512, 8, bias=False)

    def set_forward(self, x, is_feature=False):
        # (n_way, n_support), (n_way, n_query)
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous().view(self.n_way*self.n_query, -1)

        if self.feat == 0:
            z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)

        if self.feat == 1:  # self attention
            x = z_support.transpose(0, 1)
            x, _ = self.attention(x, x, x) # (seq, batch, dim)
            z_proto = x.mean(0)

        if self.feat == 2: # weighted mean
            zs1 = z_support.unsqueeze(1).expand(-1, self.n_support, -1, -1) # -> ABC ABC ...
            zs2 = z_support.unsqueeze(2).expand(-1, -1, self.n_support, -1) # -> AAA BBB ...
            dist = euclidean_distance(zs1, zs2, dim=3).sum(dim=2, keepdim=True) # -> (n_way, n_support, 1)
            w = nn.functional.softmax(torch.sqrt(dist), dim=1)
            z_proto = (z_support * w).sum(dim=1)

        dist = distance(z_query, z_proto, method='l2')
        scores = -dist

        if self.feat == 3: # residual similarity
            # (n_way, n_support, D)
            support_resid = z_support - z_proto.unsqueeze(1).expand(-1, self.n_support, -1)
            support_resid = support_resid.unsqueeze(0).expand(self.n_query*self.n_way, -1, -1, -1)

            # (n_query, n_way, D)
            query_resid = distance(z_query, z_proto, method='vec')
            query_resid = query_resid.unsqueeze(2).expand(-1, -1, self.n_support, -1)

            # (n_query, n_way, n_support, D) -> (n_query, n_way)
            sim = cos_similarity(support_resid, query_resid, dim=3).sum(2)
            scores += sim

        if self.feat == 4: # pushing away
            zp1 = z_proto.unsqueeze(1).expand(-1, self.n_way, -1)
            zp2 = z_proto.unsqueeze(0).expand(self.n_way, -1, -1)
            proto_dist = -euclidean_distance(zp1, zp2).mean()
            return scores, proto_dist

        if self.feat == 5:
            center_loss = euclidean_distance(z_support,
                z_proto.unsqueeze(1).expand(-1, self.n_support, -1)).mean()
            return scores, center_loss

        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        if self.feat == 4:
            scores, proto_dist = self.set_forwatd(x)
        elif self.feat == 5:
            scores, center_loss = self.set_forward(x)
        else:
            scores = self.set_forward(x)

        loss = self.loss_fn(scores, y_query)

        if self.feat == 4:
            return loss + 0.1 * proto_dist
        elif self.feat == 5:
            return loss + center_loss

        return loss


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
