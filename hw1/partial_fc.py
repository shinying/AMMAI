import logging
import os

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter


class PartialFC(Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @torch.no_grad()
    def __init__(self, batch_size, resume, margin_softmax, num_classes,
                 rank=0, local_rank=0, world_size=1, sample_rate=1.0, embedding_size=512, prefix="./"):
        super(PartialFC, self).__init__()
        #
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.local_rank: int = local_rank
        self.device: torch.device = torch.device("cuda:{}".format(self.local_rank))
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.margin_softmax: callable = margin_softmax
        self.sample_rate: float = sample_rate
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix
        self.num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
        self.class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)

        self.weight_name = os.path.join(self.prefix, "rank:{}_softmax_weight.pt".format(self.rank))
        self.weight_mom_name = os.path.join(self.prefix, "rank:{}_softmax_weight_mom.pt".format(self.rank))

        if resume:
            try:
                self.weight: torch.Tensor = torch.load(self.weight_name)
                logging.info("softmax weight resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
                logging.info("softmax weight resume fail!")

            try:
                self.weight_mom: torch.Tensor = torch.load(self.weight_mom_name)
                logging.info("softmax weight mom resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
                logging.info("softmax weight mom resume fail!")
        else:
            self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
            self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
            logging.info("softmax weight init successfully!")
            logging.info("softmax weight mom init successfully!")
        self.stream: torch.cuda.Stream = torch.cuda.Stream(local_rank)

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = Parameter(torch.empty((0, 0)).cuda(local_rank))

    def save_params(self):
        torch.save(self.weight.data, self.weight_name)
        torch.save(self.weight_mom, self.weight_mom_name)

    @torch.no_grad()
    def sample(self, total_label):
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start
        if int(self.sample_rate) != 1:
            positive = torch.unique(total_label[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=self.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.index = index
            total_label[index_positive] = torch.searchsorted(index, total_label[index_positive])
            self.sub_weight = Parameter(self.weight[index])
            self.sub_weight_mom = self.weight_mom[index]

    def forward(self, total_features, norm_weight):
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = linear(total_features, norm_weight)
        return logits

    @torch.no_grad()
    def update(self):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        with torch.cuda.stream(self.stream):
            total_label = label.to(self.device)
            self.sample(total_label)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            return total_label, norm_weight

    def forward_backward(self, label, features, optimizer, mixup_labels=None):
        total_label = torch.cat([label, mixup_labels[:,0], mixup_labels[:,1]])
        total_label, norm_weight = self.prepare(label, optimizer)
        total_features = features.data.to(self.device)
        total_features.requires_grad = True

        logits = self.forward(total_features, norm_weight)
        # logits = torch.cat([self.margin_softmax(logits[:len(label)], total_label),
                            # self.margin_softmax.forward_mixup(logits[len(label):], mixup_labels)])
        logits = self.margin_softmax(logits, total_label)

        with torch.no_grad():
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]

            # calculate exp(logits) and all-reduce
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # calculate loss
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_label[index, None])
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

            # calculate grad
            grad[index] -= one_hot
            valid_batch_size = self.batch_size - len(mixup_labels)
            # grad.div_(self.batch_size * self.world_size)
            grad.div_(valid_batch_size)
            x = torch.tensor([1]*len(label)+[0.5]*len(mixup_labels)*2, device=grad.device).float().unsqueeze(1)
            grad = grad * x


            # mixup
            # grad2 = logits_exp[len(label):]
            # assert grad2.size(0) == mixup_labels.size(0)
            # half_hot = torch.zeros(size=[mixup_labels.size(0), grad2.size(1)], device=grad2.device)
            # for i in range(mixup_labels.size(1)):
            #     half_hot.scatter_(1, mixup_labels[:,i].unsqueeze(1), 1./mixup_labels.size(1))

            # loss2 = torch.zeros(grad2.size(0), 1, device=grad2.device)
            # loss2 = grad2.gather(1, mixup_labels[:, 0].unsqueeze(1)) + grad2.gather(1, mixup_labels[:, 1].unsqueeze(1))
            # loss2_v = loss2.clamp_min_(1e-30).log_().mean() * -1

            # grad2 -= half_hot
            # grad2.div_(len(mixup_labels))

        # grad = torch.cat([grad, grad2])
        # loss_v = torch.cat([loss, loss2]).clamp_min_(1e-30).log_().mean() * -1

        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()

        x_grad = total_features.grad
        x_grad = x_grad * self.world_size

        return x_grad, loss_v, logits

    def evaluate(self, label, features):
        total_label = label.to(self.device)
        norm_weight = normalize(self.sub_weight)

        with torch.no_grad():
            logits = self.forward(features, norm_weight)
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]

            # calculate exp(logits) and all-reduce
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # calculate loss
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_label[index, None])
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

        return logits.cpu().argmax(dim=1), loss_v

    def predict(self, features):
        norm_weight = normalize(self.sub_weight)
        with torch.no_grad():
            logits = self.forward(features, norm_weight)

        return logits.cpu().argmax(dim=1)
