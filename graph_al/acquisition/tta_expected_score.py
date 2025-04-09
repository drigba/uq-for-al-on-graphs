from copy import deepcopy
import itertools
import numpy as np
import torch
from torch import Generator, Tensor
from torch_geometric.utils import to_dense_adj, dropout_edge, mask_feature
from sklearn.linear_model import LogisticRegression
import scipy.special

from jaxtyping import jaxtyped, Bool, Int, Float
from typeguard import typechecked

from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.model.sgc import SGC
from graph_al.model.build import get_model
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.acquisition.config import AcquisitionStrategyTTAExpectedQueryScoreConfig
from graph_al.utils.logging import get_logger
from graph_al.utils.timer import Timer

class AcquisitionStrategyTTAExpectedQueryScore(AcquisitionStrategyByAttribute):
    
    """ Acquisition strategy that uses approximation to the ground truth uncertainty. 
    
    """
    
    def __init__(self, config: AcquisitionStrategyTTAExpectedQueryScoreConfig):
        super().__init__(config)
        self.config = config
        print("TTA ENABLED")
        self.tta_strat_node = config.strat_node
        self.tta_strat_edge = config.strat_edge
        self.num = config.num
        self.drop_weights = None
        self.feature_weights = None
        self.tta_filter = config.filter
        self.p_edge = config.p_edge
        self.p_node = config.p_node
        print("p_node: ", self.p_node)
        print("p_edge: ", self.p_edge)
        from graph_al.acquisition.build import get_acquisition_strategy

        self.embedded_strategy: AcquisitionStrategyByAttribute = get_acquisition_strategy(config.embedded_strategy, None)

    
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator,
                        model_config: ModelConfig) -> Tensor:
        
        proxy = self.embedded_strategy.get_attribute(prediction, model, dataset, generator, model_config)
        pred_o = prediction.get_probabilities(propagated=True).argmax(dim=-1)
        for i in range(self.num):
            prediction_tta = self.tta_predict_single(model, model_config, dataset, generator,pred_o)
            
            score = self.embedded_strategy.get_attribute(prediction_tta, model, dataset, generator, model_config)
            proxy += score
            
        return proxy
    
    
    
    def tta_predict_single(self, model, model_config,dataset, generator,pred_o):
        
        data_clone = self.augment_data(dataset.data, generator)
        from graph_al.model.sgc import SGC
        if isinstance(model, SGC):
            x = model.get_diffused_node_features(data_clone, cache= False).cpu().numpy()        
            probs= model.logistic_regression.predict_proba(x)
            probs_unprop= model.logistic_regression.predict_proba(data_clone.x.cpu().numpy())
            logits = model.logistic_regression.decision_function(x)
            logits_unprop= model.logistic_regression.decision_function(data_clone.x.cpu().numpy())
            p_tmp = Prediction(probabilities=probs, probabilities_unpropagated=probs_unprop, logits=logits, logits_unpropagated=logits_unprop)

        else:
            with torch.no_grad():
                p_tmp = model.predict(data_clone, acquisition=True)
                p_tmp.probabilities = p_tmp.get_probabilities(propagated=True)
                p_tmp.probabilities_unpropagated = p_tmp.get_probabilities(propagated=False)
            
            
        if self.tta_filter:
            pred = p_tmp.get_probabilities(propagated=True).argmax(dim=-1)
            mask = pred != pred_o
            self.filter_results.append(mask.sum().item())
            p_tmp.logits[mask] = 0
            p_tmp.probabilities[mask] = 0
            p_tmp.logits_unpropagated[mask] = 0
            p_tmp.probabilities_unpropagated[mask] = 0

        return p_tmp
        
        
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
                edge_index, edge_mask = dropout_edge(edge_index, p=self.p_edge)
            case "adaptive":
                edge_index = self.drop_edge_weighted(edge_index, self.drop_weights, p=self.p_edge, threshold=0.7).to(device=edge_index.device)
            case "train_connection":
                train_mask = data.mask_train[edge_index[0]] | data.mask_train[edge_index[1]]
                edge_probs = torch.full((edge_index.size(1),), self.p_edge, device=edge_index.device)
                edge_probs[train_mask] = 0.8
                edge_to_drop = torch.bernoulli(edge_probs).to(torch.bool)
                edge_index = edge_index[:, ~edge_to_drop]
            case _:
                edge_index = edge_index
        return edge_index
    
    
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
                return torch.nn.functional.dropout(x, p=self.p_node, training=True)
            case "mask":
                return mask_feature(x, p=self.p_node, mode='col')[0]
            case "noise":
                noise_mask = torch.randn_like(x) * self.p_node
                return x + noise_mask
            case "adaptive":
                return self.drop_feature_weighted_2(x, self.feature_weights, p=self.p_node).to(device=x.device)
            case _:
                return x
    
    def augment_data(self, data, generator):
        """
        Augment the data using the model and generator.

        Args:
            data (torch_geometric.data.Data): The input data.
            generator (torch.Generator): The random number generator.

        Returns:
            torch_geometric.data.Data: The augmented data.
        """
        
        if self.tta_strat_node == "adaptive" and self.feature_weights is None:
            self.feature_weights = torch.load('feature_weights.pt')
        if self.tta_strat_edge == "adaptive" and self.drop_weights is None:
            self.drop_weights = torch.load('drop_weights.pt')
        
        data_clone = deepcopy(data)
        data_clone.x = self.augment_data_node(data_clone, generator)
        data_clone.edge_index = self.augment_data_edge(data_clone, generator)
        return data_clone
       
