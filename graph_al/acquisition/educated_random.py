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
from graph_al.acquisition.config import AcquisitionStrategyEducatedRandomConfig
from graph_al.utils.logging import get_logger
from graph_al.utils.timer import Timer

class AcquisitionStrategyEducatedRandom(AcquisitionStrategyByAttribute):
    
    """ Acquisition strategy that uses approximation to the ground truth uncertainty. 
    
    """
    
    def __init__(self, config: AcquisitionStrategyEducatedRandomConfig):
        super().__init__(config)
        self.config = config
        self.top_percent = config.top_percent
        self.low_percent = config.low_percent
        from graph_al.acquisition.build import get_acquisition_strategy

        self.embedded_strategy: AcquisitionStrategyByAttribute = get_acquisition_strategy(config.embedded_strategy, None)

    
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator,
                        model_config: ModelConfig) -> Tensor:
        
    

        proxy = self.embedded_strategy.get_attribute(prediction, model, dataset, generator, model_config)
        proxy = -proxy if self.embedded_strategy.higher_is_better else proxy
    
        mask_predict_indices = torch.where(dataset.data.get_mask(DatasetSplit.TRAIN_POOL))[0]
        
        num_samples = proxy.shape[0]
        num_top = int(num_samples * (self.top_percent / 100))
        num_worst = int(num_samples * (self.low_percent / 100))

        top_indices = torch.topk(proxy, num_top, largest=True).indices
        worst_indices = torch.topk(proxy, num_worst, largest=False).indices

        remaining_indices = torch.tensor([i for i in mask_predict_indices if 
                          i not in top_indices and 
                          i not in worst_indices])
        
        random_index = remaining_indices[torch.randint(len(remaining_indices), (1,))].item()
        max_scores = torch.ones_like(proxy)
        max_scores[random_index] = -1.0
        
        return max_scores
        
       
