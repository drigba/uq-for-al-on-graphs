import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from tqdm import tqdm
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from copy import deepcopy
import random
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj, is_undirected, to_undirected
from graph_al.data.config import DatasetSplit
from graph_al.acquisition.enum import AdaptationStrategy

class FeatAgentVariational:

    def __init__(self, data_all, model, config, kl_weight):
        self.device = 'cuda'
        self.config = config
        self.data_all = data_all
        self.model = model
        self.kl_weight = kl_weight

    def initialize_as_ori_feat(self, feat):
        self.delta_feat_mu.data.copy_(feat)
        self.delta_feat_logvar.data.fill_(-5)

    def learn_graph(self, dataset):
        data = dataset.data
        data = data.to(self.device)
        config = self.config
        self.data = data
        
        nnodes = dataset.num_nodes
        d = dataset.num_input_features
        
        delta_feat_mu = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        delta_feat_logvar = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        self.delta_feat_mu = delta_feat_mu
        self.delta_feat_logvar = delta_feat_logvar
        self.initialize_as_ori_feat(data.x)
        self.optimizer_feat = torch.optim.Adam([delta_feat_mu, delta_feat_logvar], lr=config.lr_feat)

        model = self.model
        for param in model.parameters():
            param.requires_grad = False
        model.eval() # should set to eval
        
        self.feat, self.edge_index = data.x, data.edge_index
        feat, edge_index = self.feat, self.edge_index
        
        for it in range(config.epochs):
            self.optimizer_feat.zero_grad()
            loss = self.test_time_loss(feat, edge_index)
            loss.backward()
            self.optimizer_feat.step()

        with torch.no_grad():
            loss = self.test_time_loss(feat, edge_index)
        # print(f'Epoch {it+1}: {loss}')
        
        output = model.forward_impl(feat + self.sample_delta_feat(), edge_index, acquisition=True)[0]
        
        return self.sample_delta_feat(), output

    def sample_delta_feat(self):
        std = torch.exp(0.5 * self.delta_feat_logvar)
        eps = torch.randn_like(std)
        return self.delta_feat_mu + eps * std

    def augment(self, feat, edge_index=None, edge_weight=None, strategy='dropedge', p=0.5):
        model = self.model
        
        if hasattr(self, 'delta_feat_mu'):
            x = feat + self.sample_delta_feat()
        else:
            x = feat
        if strategy == AdaptationStrategy.SHUFFLE:
            idx = np.random.permutation(x.shape[0])
            shuf_fts = x[idx, :]
            x = shuf_fts
        if strategy == AdaptationStrategy.DROPEDGE:
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
        if strategy == AdaptationStrategy.DROPNODE:
            mask = torch.cuda.FloatTensor(len(x)).uniform_() > p
            x = x * mask.view(-1, 1)
        if strategy == AdaptationStrategy.DROPMIX:
            mask = torch.cuda.FloatTensor(len(x)).uniform_() > p
            x = x * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
        if strategy == AdaptationStrategy.DROPFEAT:
            x = F.dropout(x, p=p) + self.sample_delta_feat()
        if strategy == AdaptationStrategy.FEATNOISE:
            mean, std = 0, p
            noise = torch.randn(x.size()) * std + mean
            x = x + noise.to(x.device)
        if edge_weight is not None:
            output = model.forward_impl(x, edge_index, edge_weight.unsqueeze(-1), acquisition=True)[0]
        else:
            output = model.forward_impl(x, edge_index, acquisition=True)[0]
        return output

    def test_time_loss(self, feat, edge_index, edge_weight=None, mode='train'):
        config = self.config
        loss = 0
        if mode == 'eval': # random seed setting
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
        output1 = self.augment(feat, edge_index, edge_weight, strategy=config.strategy, p=0.05)
        output2 = self.augment(feat, edge_index, edge_weight, strategy=AdaptationStrategy.DROPEDGE, p=0.0)
        output3 = self.augment(feat, edge_index, edge_weight, strategy=AdaptationStrategy.SHUFFLE)

        if config.margin != -1:
            loss = inner(output1, output2) - inner_margin(output2, output3, margin=config.margin)
        else:
            loss = inner(output1, output2) - inner(output2, output3)
        
        # Add KL divergence term
        kl_div = -0.5 * torch.sum(1 + self.delta_feat_logvar - self.delta_feat_mu.pow(2) - self.delta_feat_logvar.exp())
        loss += self.kl_weight * kl_div

        return loss


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(x * torch.log(x+1e-15)).sum(1)

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def sim(t1, t2):
    # cosine similarity
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (t1 * t2).sum(1)

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def inner_margin(t1, t2, margin):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return F.relu(1-(t1 * t2).sum(1)-margin).mean()

def diff(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return 0.5*((t1-t2)**2).sum(1).mean()