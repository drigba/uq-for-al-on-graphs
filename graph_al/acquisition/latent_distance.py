from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.model.sgc import SGC
from graph_al.model.build import get_model
from graph_al.acquisition.config import AcquisitionStrategyLatentDistanceConfig


from jaxtyping import jaxtyped, Bool, Int, Float
from typeguard import typechecked
from torch import Generator, Tensor
from torch_geometric.utils import to_dense_adj, dropout_edge, mask_feature

from copy import deepcopy

import itertools
import torch
import numpy as np
from tqdm import tqdm
from graph_al.utils.logging import get_logger
from sklearn.linear_model import LogisticRegression
import scipy.special

from graph_al.utils.timer import Timer

class AcquisitionStrategyLatentDistance(AcquisitionStrategyByAttribute):
    
    """ Acquisition strategy that uses approximation to the ground truth uncertainty. 
    
    """
    
    def __init__(self, config: AcquisitionStrategyLatentDistanceConfig):
        super().__init__(config)
        self.tta_strat_node = config.tta.strat_node
        self.tta_strat_edge = config.tta.strat_edge
        self.num = config.tta.num

    
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator,
                        model_config: ModelConfig) -> Tensor:
        
        embedding_orig = prediction.embeddings
        scores = torch.zeros(dataset.num_nodes, dtype=torch.float32).to(embedding_orig.device)
        for _ in range(self.num):
            data_clone = self.augment_data(dataset.data, generator)
            p_tmp = model.predict(data_clone, acquisition=True)
            embedding_tmp = p_tmp.embeddings            
            rowwise_distance = torch.norm(embedding_orig - embedding_tmp, dim=2)
            scores += rowwise_distance.squeeze(dim = 0)

        mask_predict = dataset.data.get_mask(DatasetSplit.TRAIN_POOL)
        scores[~mask_predict] = float('inf')
        return scores
    
    def augment_data_node(self, data, generator):
        """
        Augment the data using the model and generator.

        Args:
            data (torch_geometric.data.Data): The input data.
            generator (torch.Generator): The random number generator.

        Returns:
            torch_geometric.data.Data: The augmented data.
        """
        x = data.x
        match self.tta_strat_node:
            case "dropout":
                return torch.nn.functional.dropout(x, p=0.3, training=True)
            case "mask":
                return mask_feature(x, p=0.3, mode='col')
            case "noise":
                noise_mask = torch.randn_like(x) * 0.3
                return x + noise_mask

    def augment_data_edge(self, data, generator):
        """
        Augment the data using the model and generator.

        Args:
            data (torch_geometric.data.Data): The input data.
            generator (torch.Generator): The random number generator.

        Returns:
            torch_geometric.data.Data: The augmented data.
        """
        edge_index = data.edge_index
        match self.tta_strat_edge:
            case "mask":
                edge_index, edge_mask = dropout_edge(edge_index, p=0.3)
            case "train_connection":
                train_mask = data.mask_train[edge_index[0]] | data.mask_train[edge_index[1]]
                edge_probs = torch.full((edge_index.size(1),), 0.3, device=edge_index.device)
                edge_probs[train_mask] = 0.8
                edge_to_drop = torch.bernoulli(edge_probs).to(torch.bool)
                edge_index = edge_index[:, ~edge_to_drop]
        return edge_index
    
    
    def augment_data(self, data, generator):
        """
        Augment the data using the model and generator.

        Args:
            data (torch_geometric.data.Data): The input data.
            generator (torch.Generator): The random number generator.

        Returns:
            torch_geometric.data.Data: The augmented data.
        """
        data_clone = deepcopy(data)
        data_clone.x = self.augment_data_node(data_clone, generator)
        data_clone.edge_index = self.augment_data_edge(data_clone, generator)
        return data_clone
       
