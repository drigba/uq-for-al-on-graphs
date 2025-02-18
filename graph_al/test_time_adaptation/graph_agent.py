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
        
        for it in tqdm(range(config.epochs//(config.loop_feat+config.loop_adj))):
            for loop_feat in range(config.loop_feat):
                self.optimizer_feat.zero_grad()
                data_tmp = deepcopy(data)

                data_tmp = data_tmp.to(self.device)
                data_tmp.x = data_tmp.x + delta_feat
                
                loss = self.test_time_loss(self.model,data_tmp)
                loss = delta_feat.sum()
                loss.backward()
                print("loss", loss, loss.grad)
                print("delta grad", delta_feat.grad.mean(), delta_feat.grad.std(), delta_feat.grad.max(), delta_feat.grad.min())
                if loop_feat == 0:
                    print(f'Epoch {it}, Loop Feat {loop_feat}: {loss.item()}')

                self.optimizer_feat.step()
                print(","*30)
                print("delta feat", delta_feat.mean(), delta_feat.std(), delta_feat.max(), delta_feat.min())
                print("data", data.x.mean(), data.x.std(), data.x.max(), data.x.min())
                print("data+delta", (data.x+delta_feat).mean())  
                print(","*30)

            new_feat = (feat+delta_feat).detach()
            
            # for loop_adj in range(config.loop_adj):
            #     self.perturbed_edge_weight.requires_grad = True
            #     edge_index, edge_weight  = self.get_modified_adj()
            #     if torch.cuda.is_available() and self.do_synchronize:
            #         torch.cuda.empty_cache()
            #         torch.cuda.synchronize()
            #     data_tmp = deepcopy(data)
            #     data_tmp.x,data_tmp.edge_index, data_tmp.edge_attr = new_feat,edge_index, edge_weight
            #     # loss = self.test_time_loss(data_tmp)
                
            #     from  torch_geometric.nn import GCNConv
            #     l = GCNConv(2879,64).cuda()
            #     for param in l.parameters():
            #         param.requires_grad = False
                
            #     output = self.model.forward_impl(new_feat,edge_index, edge_weight.unsqueeze(-1), acquisition=True)[0]
            #     loss = output.sum()

            #     gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]
            #     if not config.existing_space:
            #         if torch.cuda.is_available() and self.do_synchronize:
            #             torch.cuda.empty_cache()
            #             torch.cuda.synchronize()

            #     if loop_adj == 0:
            #         print(f'Epoch {it}, Loop Adj {loop_adj}: {loss.item()}')    

            #     with torch.no_grad():
            #         self.update_edge_weights(n_perturbations, it, gradient)
            #         self.perturbed_edge_weight = self.project(
            #             n_perturbations, self.perturbed_edge_weight, self.eps)
            #         del edge_index, edge_weight #, logits

            #     if it < self.epochs_resampling - 1:
            #         self.perturbed_edge_weight.requires_grad = True
            #         self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=config.lr_adj)

            # # edge_index, edge_weight = self.sample_final_edges(n_perturbations, data)
            # if config.loop_adj != 0:
            #     edge_index, edge_weight  = self.get_modified_adj()
            #     edge_weight = edge_weight.detach()

        print(f'Epoch {it+1}: {loss}')

        if config.loop_adj != 0:
            edge_index, edge_weight = self.sample_final_edges(n_perturbations, data)

        data_tmp = deepcopy(data)
        
        
        
        data_tmp.x,data_tmp.edge_index, data_tmp.edge_attr = data_tmp.x + delta_feat, edge_index, edge_weight
        # TODO MODIFY
        # output = model.predict(data_tmp, acquisition=True)
        output = model.forward_impl(data_tmp.x, data_tmp.edge_index, data_tmp.edge_attr.unsqueeze(-1), acquisition=True)[0]
        del data_tmp.edge_attr
        return data_tmp, output


    def augment(self,data, strategy=AdaptationStrategy.DROPEDGE, p=0.5 ):
        model = self.model
        data_tmp = deepcopy(data)
        edge_index, edge_weight = data_tmp.edge_index, data_tmp.edge_attr
        
            
        if strategy == AdaptationStrategy.SHUFFLE:
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            #TODO DONE
            data_tmp.x = shuf_fts
            
        if strategy == AdaptationStrategy.DROPEDGE:
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            #TODO DONE
            data_tmp.edge_index, data_tmp.edge_attr = edge_index, edge_weight
        if strategy == AdaptationStrategy.DROPNODE:
            feat = data_tmp.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            #TODO DONE
            data_tmp.x = feat
        if strategy == AdaptationStrategy.RWSAMPLE:
            import augmentor as A
            walk_length = 10
            aug = A.RWSampling(num_seeds=1000, walk_length=walk_length)
            x = self.feat + self.delta_feat
            x2, edge_index2, edge_weight2 = aug(x, edge_index, edge_weight)
            # TODO MODIFY
            data_tmp.x, data_tmp.edge_index, data_tmp.edge_attr = x2, edge_index2, edge_weight2

        if strategy == AdaptationStrategy.DROPMIX:
            feat = data_tmp.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            #TODO DONE
            data_tmp.x, data_tmp.edge_index, data_tmp.edge_attr = feat, edge_index, edge_weight
        if strategy == AdaptationStrategy.DROPFEAT:
            feat = F.dropout(data_tmp.x, p=p) + self.delta_feat
            #TODO DONE
            data_tmp.x = feat
        if strategy == AdaptationStrategy.FEATNOISE:
            mean, std = 0, p
            noise = torch.randn(feat.size()) * std + mean
            feat = feat + noise.to(feat.device)
            #TODO DONE
            data_tmp.x = feat
        
        if data_tmp.edge_attr is not None:
            data_tmp.edge_attr = data_tmp.edge_attr.unsqueeze(-1)
        # output = model.predict(data_tmp, acquisition=True)
        output = model.forward_impl(data_tmp.x, data_tmp.edge_index, data_tmp.edge_attr, acquisition=True)[0]
        if data_tmp.edge_attr is not None:
            data_tmp.edge_attr = data_tmp.edge_attr.squeeze(-1)
        return output

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def diff(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return 0.5*((t1-t2)**2).sum(1).mean()


