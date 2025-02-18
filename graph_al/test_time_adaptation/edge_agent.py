"""learn edge indices"""
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from tqdm import tqdm
from graph_al.test_time_adaptation.feat_agent import FeatAgent
import torch_sparse
from torch_sparse import coalesce
import math
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj, is_undirected, to_undirected
from copy import deepcopy
from graph_al.acquisition.enum import *

class EdgeAgent(FeatAgent):

    def __init__(self, data_all,model, config):
        self.device = 'cuda'
        self.config = config
        self.data_all = data_all
        self.model = model

    def setup_params(self, dataset):
        config = self.config
        for param in self.model.parameters():
            param.requires_grad = False

        nnodes = dataset.num_nodes
        d = dataset.num_input_features

        self.n, self.d = nnodes, nnodes

        self.make_undirected = True
        self.max_final_samples = 20
        self.search_space_size = 10_000_000
        self.eps = 1e-7

        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later

        lr_factor = config.lr_adj
        self.lr_factor = lr_factor * max(math.log2(self.n_possible_edges / self.search_space_size), 1.)
        self.epochs_resampling = self.config.epochs
        self.with_early_stopping = True
        self.do_synchronize = True

    def learn_graph(self, dataset):
        self.setup_params(dataset)
        data = dataset.data
        config = self.config
        model = self.model
        model.eval() # should set to eval

        feat, labels = data.x, data.y
        self.edge_index = data.edge_index
        self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.device)
        self.feat = feat

        n_perturbations = int(config.ratio * self.edge_index.shape[1] //2)
        print('n_perturbations:', n_perturbations)
        self.sample_random_block(n_perturbations)

        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=config.lr_adj)
        for it in tqdm(range(config.epochs)):
            self.perturbed_edge_weight.requires_grad = True

            edge_index, edge_weight  = self.get_modified_adj()

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            data_tmp = deepcopy(data)
            data_tmp.edge_index, data_tmp.edge_attr = edge_index, edge_weight
            loss = self.test_time_loss(data_tmp)
            gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

            if torch.cuda.is_available() and self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if it == 0:
                print(f'Epoch {it}: {loss}')

            with torch.no_grad():
                self.update_edge_weights(n_perturbations, it, gradient)
                self.perturbed_edge_weight = self.project(
                    n_perturbations, self.perturbed_edge_weight, self.eps)
                del edge_index, edge_weight #, logits
            if it < self.epochs_resampling - 1:
                self.perturbed_edge_weight.requires_grad = True
                self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=config.lr_adj)

        print(f'Epoch {it}: {loss}')
        edge_index, edge_weight = self.sample_final_edges(n_perturbations, data)
        data_tmp = deepcopy(data)
        data_tmp.x, data_tmp.edge_index, data_tmp.edge_attr = feat, edge_index, edge_weight
        loss = self.test_time_loss(data_tmp)
        print('final loss:', loss.item())

        output = model.predict(data_tmp, acquisition=True)
        print('Test:')
        return output, data_tmp


    def augment(self,data_tmp, strategy=AdaptationStrategy.DROPEDGE, p=0.5 ):
        model = self.model
        feat = self.feat
        if strategy == AdaptationStrategy.SHUFFLE:
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            #TODO DONE
            data_tmp.x = shuf_fts
        if strategy == AdaptationStrategy.DROPEDGE:
            edge_index, edge_weight = dropout_adj(data_tmp.edge_index, data_tmp.edge_attr, p=p)
            #TODO DONE
            data_tmp.edge_index, data_tmp.edge_attr = edge_index, edge_weight

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
        output = model.predict(data_tmp, acquisition=True)
        return output

    def sample_random_block(self, n_perturbations):
        if self.config.existing_space:
            edge_index = self.edge_index.clone()
            edge_index = edge_index[:, edge_index[0] < edge_index[1]]
            row, col = edge_index[0], edge_index[1]
            edge_index_id = (2*self.n - row-1)*row//2 + col - row -1 # // is important to get the correct result
            edge_index_id = edge_index_id.long()
            self.current_search_space = edge_index_id
            self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            self.perturbed_edge_weight = Parameter(torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            ))
            return
        
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    @torch.no_grad()
    def sample_final_edges(self, n_perturbations, data):
        best_loss = float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0
        feat, labels = data.x, data.y
        for i in range(self.max_final_samples):
            if best_loss == float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                if self.config.debug ==2:
                    print(f'{i}-th sampling: too many samples {n_samples}')
                continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            data_tmp = deepcopy(data)
            data_tmp.edge_index, data_tmp.edge_attr = edge_index, edge_weight
            with torch.no_grad():
                # output = self.model.forward(feat, edge_index, edge_weight)
                loss = self.test_time_loss(self.model,data_tmp, mode='eval')
            # Save best sample
            if best_loss > loss:
                best_loss = loss
                print('best_loss:', best_loss)
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))
        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = self.edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    

    def project(self, n_perturbations, values, eps, inplace=False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    def get_modified_adj(self):
        if self.make_undirected:
            modified_edge_index, modified_edge_weight = to_symmetric(
                self.modified_edge_index, self.perturbed_edge_weight, self.n
            )
        else:
            modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight
        
        edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
        

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]

        return edge_index, edge_weight





    def _update_edge_weights(self, n_perturbations, epoch, gradient):
        lr_factor = n_perturbations / self.n / 2 * self.lr_factor
        lr = lr_factor / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        self.perturbed_edge_weight.data.add_(-lr * gradient)
        # We require for technical reasons that all edges in the block have at least a small positive value
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

    def update_edge_weights(self, n_perturbations, epoch, gradient):
        self.optimizer_adj.zero_grad()
        self.perturbed_edge_weight.grad = gradient
        self.optimizer_adj.step()
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(torch.exp(x) * x).sum(1)


def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )
    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight

def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1 - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))


def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)
    grad_outputs = []
    for input in inputs:

        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs

def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu

def homophily(adj, labels):
    edge_index = adj.nonzero()
    homo = (labels[edge_index[0]] == labels[edge_index[1]])
    return np.mean(homo.numpy())

