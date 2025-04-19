from graph_al.data.base import Dataset
from graph_al.data.generative import GenerativeDataset
from graph_al.data.config import DatasetSplit
from graph_al.model.base import BaseModel
from graph_al.model.config import SGCConfig
from graph_al.model.prediction import Prediction
from graph_al.utils.logging import get_logger
from graph_al.data.base import Data
from graph_al.model.config import ApproximationType

import numpy as np
import torch
import torch_scatter
import torch_geometric.nn as tgnn
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

import itertools
from scipy.special import logsumexp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import math
from copy import deepcopy

from jaxtyping import jaxtyped, Float, Int, Bool, UInt64
from typeguard import typechecked
from typing import Tuple, Any

from graph_al.utils.utils import batched


class SGC(BaseModel):
    """ Uses the simplified graph convolution framework: p = sigma(A^(k)XW) """

    def __init__(self, config: SGCConfig, dataset: Dataset, generator: torch.Generator):
        super().__init__(config, dataset)
        self.inverse_regularization_strength = config.inverse_regularization_strength
        self.cached = config.cached
        self.balanced = config.balanced
        self.normalize = True
        self.add_self_loops = config.add_self_loops
        self.improved = config.improved
        self.k = config.k
        self.solver = config.solver
        self.model = MulticlassLogisticRegression(dataset.base.num_input_features, dataset.num_classes)
        self.model_set = False
        self.reset_cache()
        self._frozen_prediction = None
        self.reset_parameters(generator)
        
    def reset_cache(self):
        self._cached_node_features = None



    def reset_parameters(self, generator: torch.Generator):
        self.logistic_regression = LogisticRegression(C=self.inverse_regularization_strength, solver=self.solver,
            class_weight='balanced' if self.balanced else None)
        self._frozen_prediction = None
        self.model_set = False

    @torch.no_grad()
    def predict(self, batch: Data, acquisition: bool = False) -> Prediction:
        if isinstance(self._frozen_prediction, int): # Prediction is frozen to this one class
            probs = np.zeros((batch.num_nodes, batch.num_classes), dtype=float) # type: ignore
            probs[:, self._frozen_prediction] = 1.0
            probs, probs_unpropagated = probs, probs
            logits, logits_unpropagated = probs, probs_unpropagated # a bit arbitrary...
            x = probs
        else:
            if self.logistic_regression is None:
                raise RuntimeError(f'No regression model was fitted for SGC')
            try:
                x = self.get_diffused_node_features(batch, cache=self.cached)
                probs = self.predict_proba(x)
                probs_unpropagated = self.predict_proba(batch.x)
                logits = self.decision_function(x)
                logits_unpropagated = self.decision_function(batch.x)
            except NotFittedError:
                get_logger().warn(f'Predictions with a non-fitted regression model: Fall back to uniform predictions')
                probs = np.ones((batch.num_nodes, batch.num_classes), dtype=float) / batch.num_classes # type: ignore
                probs, probs_unpropagated = probs, probs
                logits, logits_unpropagated = probs, probs_unpropagated # a bit arbitrary...
        return Prediction(probabilities=probs.unsqueeze(0), 
                          probabilities_unpropagated=probs_unpropagated.unsqueeze(0),
                          logits=logits.unsqueeze(0),
                          logits_unpropagated=logits_unpropagated.unsqueeze(0),
                          # they are not really embeddings, this is a bit iffy...
                          embeddings=logits.unsqueeze(0),
                          embeddings_unpropagated=logits_unpropagated.unsqueeze(0),
                          )
    
    def set_model(self):
        """ Sets the model to be used for predictions. """
        if self.model_set:
            return
        
        with torch.no_grad():
            self.model.linear.weight.copy_(torch.tensor(self.logistic_regression.coef_).float())
            self.model.linear.bias.copy_(torch.tensor(self.logistic_regression.intercept_).float())
        self.model =  self.model.cuda()
        self.model_set = True
    
    def predict_proba(self, batch: Data) -> Prediction: 
        self.set_model()
        self.model.eval()
        with torch.no_grad():
            logits, probs = self.model(batch)
        return probs
    
    def decision_function(self, batch: Data) -> Prediction:
        self.set_model()
        self.model.eval()
        with torch.no_grad():
            logits, probs = self.model(batch)
        return logits

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def get_diffused_node_features(self, batch: Data, cache: bool = None) -> Float[torch.Tensor, 'num_nodes num_features']:
        """ Gets the diffused node features. """
        if cache is None:
            cache = self.cached
        return batch.get_diffused_nodes_features(self.k, normalize=self.normalize, improved=self.improved,
                    add_self_loops=self.add_self_loops, cache=cache)

    def freeze_predictions(self, class_idx: int):
        self._frozen_prediction = class_idx
    
    def unfreeze_predictions(self):
        self._frozen_prediction = None


import torch.nn as nn
import torch.nn.functional as F
class MulticlassLogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        logits = self.linear(x)             # same as decision_function
        probs = F.softmax(logits, dim=1)    # same as predict_proba
        return logits, probs
                

