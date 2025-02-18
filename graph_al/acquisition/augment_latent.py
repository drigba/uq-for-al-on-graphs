from graph_al.data.base import Data, Dataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import ModelConfig
from graph_al.model.prediction import Prediction
from graph_al.acquisition.attribute import AcquisitionStrategyByAttribute
from graph_al.model.sgc import SGC
from graph_al.model.build import get_model
from graph_al.acquisition.config import AcquisitionStrategyAugmentLatentConfig


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

class AcquisitionStrategyAugmentLatent(AcquisitionStrategyByAttribute):
    
    """ Acquisition strategy that uses approximation to the ground truth uncertainty. 
    
    """
    
    def __init__(self, config: AcquisitionStrategyAugmentLatentConfig):
        super().__init__(config)
        self.num = config.num_tta

    
    @jaxtyped(typechecker=typechecked)
    def get_attribute(self, prediction: Prediction | None, model: BaseModel, dataset: Dataset, generator: Generator,
                        model_config: ModelConfig) -> Tensor:
        
        embedding_orig = prediction.embeddings
        edge_index = dataset.data.edge_index
        edge_weight = dataset.data.edge_attr
        prediction.probabilities = prediction.get_probabilities()
        pred_o = prediction.probabilities.argmax(dim=-1)
        cnt = torch.ones_like(pred_o, dtype=torch.float)
        for _ in range(self.num):
            embedding_clone = deepcopy(embedding_orig)
            embedding_tmp = self.augment_embedding(embedding_clone, generator)
            logit_tmp = model.layers[-1](embedding_tmp, edge_index,edge_weight)
            prob_tmp = torch.softmax(logit_tmp, dim=-1)
            pred = prob_tmp.argmax(dim=-1)
            
            prob_tmp[pred != pred_o] = 0
            cnt[pred == pred_o] += 1
            prediction.probabilities += prob_tmp

        prediction.probabilities /= cnt.unsqueeze(-1)
        scores = prediction.get_max_score(propagated=True)
        mask_predict = dataset.data.get_mask(DatasetSplit.TRAIN_POOL)
        scores[~mask_predict] = float('inf')
        return scores
    
    def augment_embedding(self,embedding,generator):
        center_embedding = embedding.mean(dim=0)
        avg_distance = torch.norm(embedding - center_embedding, p=2, dim=-1).mean()
        epsilon = avg_distance*0.1

        noise = torch.randn_like(embedding) - 0.5
        noise_unit_vector = noise/torch.norm(noise,p=2,dim=-1,keepdim=True)
        
        final_noise = torch.normal(mean=epsilon,std = epsilon*0.04)*noise_unit_vector
        new_embedding = embedding + final_noise

        return new_embedding
        
       
