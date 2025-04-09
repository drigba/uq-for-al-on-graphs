from dataclasses import dataclass, field
from graph_al.acquisition.enum import *

@dataclass
class TTAConfig:
    strat_node: str | None = NodeAugmentation.NOISE # which node augmentation strategy to use
    strat_edge: str | None = EdgeAugmentation.MASK # which edge augmentation strategy to use
    norm: bool | None = True
    num: int = 100 # number of tta samples
    filter: bool = False # whether to filter the tta 
    probs: bool = True # whether to use the probabilities of the tta samples or logits
    p_edge: float = 0.3 # probability of edge dropout
    p_node: float = 0.3