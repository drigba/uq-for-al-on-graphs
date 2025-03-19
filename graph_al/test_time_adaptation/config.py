from dataclasses import dataclass, field
from graph_al.acquisition.enum import AdaptationStrategy, AdaptationMode, AdaptationIntegration

@dataclass
class AdaptationConfig:
    lr_feat: float = 0.0005 # learning rate for feature adaptation
    lr_adj: float = 0.1 # learning rate for structure adaptation
    epochs: int = 20 # number of epochs for feature adaptation
    seed: int = 0 # seed for reproducibility
    strategy: str = AdaptationStrategy.DROPEDGE # strategy for augmenting the graph
    margin: float = -1 # margin for the loss function
    ratio: float = 0.1 # budget for changing the graph structure
    existing_space: bool = True # whether to enable removing edges from the graph
    loop_adj: int = 1 # number of loops for optimizing structure
    loop_feat: int = 4 # number of loops for optimizing features
    debug:int = 0 # debug flag
    mode: str = AdaptationMode.FEATURE # mode for adaptation
    integration: str = AdaptationIntegration.QUERY # integration for adaptation
