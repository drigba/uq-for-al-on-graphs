import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from tqdm import tqdm
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from copy import deepcopy
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj, is_undirected, to_undirected
from graph_al.test_time_adaptation.edge_agent import EdgeAgent, grad_with_checkpoint
from graph_al.test_time_adaptation.feat_agent import FeatAgent
from graph_al.acquisition.enum import AdaptationStrategy

class GraphAgent(EdgeAgent):

    def __init__(self, data_all, model, config):
        self.device = 'cuda'
        self.config = config
        self.data_all = data_all
        self.model = model

    def learn_graph(self, dataset):
        print('====learning on this graph===')
        config = self.config
        data= dataset.data.cuda()
        import torch_geometric.nn as tgnn
        
        # MODIFY
        self.setup_params(dataset)
        
        config = self.config
        model = self.model
        model.eval() # should set to eval

        self.max_final_samples = 5

        config = self.config
        self.data = data
        
        nnodes = dataset.num_nodes
        d = dataset.num_input_features

        delta_feat = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        self.delta_feat = delta_feat
        delta_feat.data.fill_(1e-7)
        self.optimizer_feat = torch.optim.Adam([delta_feat], lr=config.lr_feat)

        model = self.model
        
        # MAY CAUSE PROBLEMS
        for param in model.parameters():
            param.requires_grad = False
        model.eval() # should set to eval


        edge_index,feat,labels = dataset.data.edge_index ,dataset.data.x, dataset.data.y
        
        self.edge_index, self.feat, self.labels = edge_index, feat, labels
        self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.device)

        n_perturbations = int(config.ratio * self.edge_index.shape[1] //2)
        print('n_perturbations:', n_perturbations)
        
        # MAYBE MODIFY
        self.sample_random_block(n_perturbations)
        
            
        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=config.lr_adj)
        edge_index, edge_weight = edge_index, None
        model = self.model
        
        for it in range(config.epochs//(config.loop_feat+config.loop_adj)):
            for loop_feat in range(config.loop_feat):
                self.optimizer_feat.zero_grad()
                loss = self.test_time_loss(feat, edge_index, edge_weight)
                loss.backward()
                # if loop_feat == 0:
                #     print(f'Epoch {it}, Loop Feat {loop_feat}: {loss.item()}')
                self.optimizer_feat.step()
            new_feat = (feat+delta_feat).detach()
            
            for loop_adj in range(config.loop_adj):
                self.perturbed_edge_weight.requires_grad = True
                edge_index, edge_weight  = self.get_modified_adj()
                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                loss = self.test_time_loss(new_feat, edge_index, edge_weight)
                gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
                if not config.existing_space:
                    if torch.cuda.is_available() and self.do_synchronize:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                # if loop_adj == 0:
                #     print(f'Epoch {it}, Loop Adj {loop_adj}: {loss.item()}')    

                with torch.no_grad():
                    self.update_edge_weights(n_perturbations, it, gradient)
                    self.perturbed_edge_weight = self.project(
                        n_perturbations, self.perturbed_edge_weight, self.eps)
                    del edge_index, edge_weight #, logits

                if it < self.epochs_resampling - 1:
                    self.perturbed_edge_weight.requires_grad = True
                    self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=config.lr_adj)

            # edge_index, edge_weight = self.sample_final_edges(n_perturbations, data)
            if config.loop_adj != 0:
                edge_index, edge_weight  = self.get_modified_adj()
                edge_weight = edge_weight.detach()

        print(f'Epoch {it+1}: {loss}')

        if config.loop_adj != 0:
            edge_index, edge_weight = self.sample_final_edges(n_perturbations, data)
            
        for param in model.parameters():
            param.requires_grad = True

        return delta_feat, edge_index, edge_weight


    def augment(self,feat, edge_index=None, edge_weight=None, strategy='dropedge', p=0.5):
        model = self.model
        if hasattr(self, 'delta_feat'):
            x = feat + self.delta_feat
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
        if strategy == AdaptationStrategy.RWSAMPLE:
            import augmentor as A
            walk_length = 10
            aug = A.RWSampling(num_seeds=1000, walk_length=walk_length)
            x = self.feat + self.delta_feat
            x, edge_index, edge_weight = aug(x, edge_index, edge_weight)

        if strategy == AdaptationStrategy.DROPMIX:
            mask = torch.cuda.FloatTensor(len(x)).uniform_() > p
            x = x * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
        if strategy == AdaptationStrategy.DROPFEAT:
            x = F.dropout(x, p=p)
        if strategy == AdaptationStrategy.FEATNOISE:
            mean, std = 0, p
            noise = torch.randn(x.size()) * std + mean
            x = x + noise.to(x.device)
        
        data_tmp = deepcopy(self.data)
        data_tmp.x = x
        data_tmp.edge_index = edge_index
        if edge_weight is not None:
            data_tmp.edge_attr = edge_weight.unsqueeze(-1)
        output = model.predict(data_tmp, acquisition=True).embeddings[0]
        return output

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def diff(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return 0.5*((t1-t2)**2).sum(1).mean()


