from graph_al.acquisition.config import AcquisitionStrategyLeaveOutConfig
from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.model.sgc import SGC
from graph_al.model.build import get_model

from jaxtyping import jaxtyped, Bool, Int, Float
from typeguard import typechecked
from torch import Generator, Tensor

from copy import deepcopy

import itertools
import torch
import numpy as np
from tqdm import tqdm
from graph_al.utils.logging import get_logger
from sklearn.linear_model import LogisticRegression
import scipy.special

from graph_al.utils.timer import Timer

class AcquisitionStrategyLeaveOut(AcquisitionStrategyByAttribute):
    
    """ Acquisition strategy that uses approximation to the ground truth uncertainty. 
    
    """
    
    def __init__(self, config: AcquisitionStrategyLeaveOutConfig):
        super().__init__(config)
    
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator,
                        model_config: ModelConfig) -> Tensor:
        node_risks = []
        for node in range(dataset.num_nodes):
            data_clone = deepcopy(dataset.data)
            edges = data_clone.edge_index
            mask = (edges[0] != node) & (edges[1] != node)
            data_clone.edge_index = edges[:, mask]
            pred_tmp = model.predict(data_clone,acquisition=True)
            risk = 1 - pred_tmp.get_probabilities(propagated=True)[0].max(dim=1)[0].mean()
            node_risks.append(risk)
        node_risks = torch.tensor(node_risks, device=dataset.data.x.device)
        mask_predict = dataset.data.get_mask(DatasetSplit.TRAIN_POOL)
        node_risks[~mask_predict] = 0
        return node_risks
        
       
