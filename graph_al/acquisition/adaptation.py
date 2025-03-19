from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.model.sgc import SGC
from graph_al.model.build import get_model
from graph_al.acquisition.config import AcquisitionStrategyAdaptationConfig
from graph_al.test_time_adaptation.feat_agent_disable import FeatAgentDisable
from graph_al.test_time_adaptation.feat_agent import FeatAgent
from graph_al.test_time_adaptation.feat_agent_variational import FeatAgentVariational
from graph_al.test_time_adaptation.edge_agent import EdgeAgent

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

class AcquisitionStrategyAdaptation(AcquisitionStrategyByAttribute):
    
    """ Acquisition strategy that uses approximation to the ground truth uncertainty. 
    
    """
    
    def __init__(self, config: AcquisitionStrategyAdaptationConfig):
        super().__init__(config)
        self.config = config

        

    
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator,
                        model_config: ModelConfig) -> Tensor:
        # agent = FeatAgentVariational(dataset, model, self.config, 0.1)
        agent = FeatAgent(dataset, model, self.config)
        x_original = deepcopy(dataset.data.x.detach())
        with torch.enable_grad():
                delta_feat, loss = agent.learn_graph(dataset)
        dataset.data.x = dataset.data.x + delta_feat
        output = model.predict(dataset.data, acquisition=True)
        dataset.data.x = x_original
        max_scores = output.get_max_score(propagated=True)        
    
        return max_scores
        
       
