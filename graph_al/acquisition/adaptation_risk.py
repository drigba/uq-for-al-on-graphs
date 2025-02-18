from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.model.sgc import SGC
from graph_al.model.build import get_model
from graph_al.acquisition.config import AcquisitionStrategyAdaptationRiskConfig
from graph_al.test_time_adaptation.feat_agent_disable import FeatAgentDisable
from graph_al.test_time_adaptation.feat_agent import FeatAgent

from jaxtyping import jaxtyped, Bool, Int, Float
from typeguard import typechecked
from torch import Generator, Tensor
from torch_geometric.utils import to_dense_adj, dropout_edge, mask_feature

from copy import deepcopy

import itertools
import torch
import numpy as np
from graph_al.utils.logging import get_logger
from sklearn.linear_model import LogisticRegression
import scipy.special
from graph_al.utils.timer import Timer

class AcquisitionStrategyAdaptationRisk(AcquisitionStrategyByAttribute):
    
    """ Acquisition strategy that uses approximation to the ground truth uncertainty. 
    
    """
    
    def __init__(self, config: AcquisitionStrategyAdaptationRiskConfig):
        super().__init__(config)
        print("init AcquisitionStrategyAdaptationRisk")
        self.config = config

        

    
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator,
                        model_config: ModelConfig) -> Tensor:
        agent = FeatAgentDisable(dataset.data, model, self.config)
        # agent = FeatAgent(dataset.data, model, self.config)
        probs = prediction.get_probabilities(propagated=True)[0]
        proxy = [float('inf') for i in range(dataset.num_nodes)]
        train_pool_indexes = torch.where(dataset.data.get_mask(DatasetSplit.TRAIN_POOL) == 1)[0].tolist()
        for i in train_pool_indexes:
            with torch.enable_grad():
                delta_feat,output, loss = agent.learn_graph(dataset, i)
                output = model.forward_impl(dataset.data.x+delta_feat,dataset.data.edge_index, acquisition=True)[1]
                # delta_feat,output, loss = agent.learn_graph(dataset)
            p = 1 - torch.nn.functional.softmax(output, dim=1)[i].max()
            # r = 1 - torch.nn.functional.softmax(output, dim=1)[i]
            # p = (r*probs[i]).sum()
            # proxy.append(p)
            proxy[i] = p
        proxy = torch.tensor(proxy).to(dataset.data.x.device)
        mask_predict = dataset.data.get_mask(DatasetSplit.TRAIN_POOL)
        proxy[~mask_predict] = float('inf')
        return proxy
        
       
