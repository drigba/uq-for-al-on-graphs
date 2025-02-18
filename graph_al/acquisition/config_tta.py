from dataclasses import dataclass, field
from graph_al.acquisition.enum import *

@dataclass
class TTAConfig:
    strat_node: str | None = NodeAugmentation.NOISE # which node augmentation strategy to use
    strat_edge: str | None = EdgeAugmentation.MASK # which edge augmentation strategy to use
    norm: bool | None = True
    retrain_model: bool | None = False
    num: int = 100 # number of tta samples
    filter: bool = False # whether to filter the tta 