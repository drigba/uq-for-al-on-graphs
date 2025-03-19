import os
import pandas as pd
from graph_al.test_time_adaptation.config import AdaptationConfig
import plotly.graph_objects as go


def update_progress(df, path):
    for dataset in os.listdir(path):
        for model in os.listdir(os.path.join(path, dataset)):
            for strategy in os.listdir(os.path.join(path, dataset, model)):
                df.loc[(dataset, model, strategy), "progress"] = True
                strat_path = os.path.join(path, dataset, model, strategy)
                cnt = 0
                for seed in os.listdir(strat_path):
                    seed_dir = os.path.join(strat_path, seed)
                    for run_dir in os.listdir(seed_dir):
                        run_path = os.path.join(seed_dir, run_dir)
                        if run_dir != "hydra-outputs":
                            if strategy == "geem":
                                if os.path.exists(os.path.join(run_path, "acquisition_metrics-4-4-24.pt")):
                                    cnt = 25
                                    break
                                elif os.path.exists(os.path.join(run_path, "acquisition_curve_metrics.pt")):
                                    cnt += 1
                            else:
                                cnt = max(cnt, sum(1 for file in os.listdir(run_path) if file.startswith("acquisition_metrics")))
                df.loc[(dataset, model, strategy), "progress_percentage"] = cnt / 25
                df.loc[(dataset, model, strategy), "progress_count"] = cnt
                if cnt >= 25:
                    df.loc[(dataset, model, strategy), "done"] = True
    return df

def plot_progress(expected_datasets, df):
    for dataset in expected_datasets:
        df_reset = df.reset_index()
        df_reset = df_reset[df_reset["dataset"] == dataset]
        df_reset["strategy"] = df_reset["strategy"].apply(lambda x: x.replace("approximate_uncertainty_", ""))
        df_reset.loc[df_reset["strategy"] == "aleatoric_propagated","strategy"] = "aleatoric"
        fig = go.Figure()
        fig.add_bar(x=df_reset.transpose().loc[["model","strategy"]],y=df_reset["progress_percentage"].transpose())
        fig.update_layout(title=dataset.capitalize())
        fig.show()
        
        
def generate_prompt(dataset, model, strategy):
    s = (
        f"nohup python main.py model={model} data={dataset} "
        f"acquisition_strategy={strategy} data.num_splits=5 "
        f"model.num_inits=5 print_summary=True "
        f"acquisition_strategy.tta=null > logs_new/{dataset}_{model}_{strategy}.log &"
    )
    return s


def generate_prompt_geem(dataset, model, strategy, seed):
    s = (
        f"nohup python main.py model={model} data={dataset} "
        f"acquisition_strategy={strategy} data.num_splits=1 "
        f"acquisition_strategy.adaptation_enabled=False "
        f"model.num_inits=1 print_summary=True "
        f"seed={seed} "
        f"model.cached=True "
        f"wandb.name={seed} "
        f"acquisition_strategy.tta_enabled=False > logs_new/{dataset}_{model}_{strategy}_{seed}.log &"
    )
    return s


from graph_al.acquisition.enum import *

def generate_adaptation_name(adaptation_config: AdaptationConfig):
    match adaptation_config.mode:
        case AdaptationMode.FEATURE:
            return f"feature_lr{adaptation_config.lr_feat}_epochs{adaptation_config.epochs}_i{adaptation_config.integration}"
        case AdaptationMode.STRUCTURE:
            return f"adj_lr{adaptation_config.lr_adj}_epochs{adaptation_config.epochs}_i{adaptation_config.integration}"
        case AdaptationMode.BOTH:
            return f"graph_lra{adaptation_config.lr_adj}_lrf{adaptation_config.lr_feat}_epochs{adaptation_config.epochs}_i{adaptation_config.integration}"
        
        
        
def generate_prompt_adaptation(dataset,model,strategy,adaptation_config:AdaptationConfig,scale, seed):
    wn = generate_adaptation_name(adaptation_config)
    s = (
        f"nohup python main.py model={model} data={dataset} "
        f"acquisition_strategy={strategy} data.num_splits=5 "
        f"model.num_inits=5 print_summary=True "
        f"model.cached=False "
        f"seed={seed} "
        f"acquisition_strategy.tta_enabled=False "
        f"acquisition_strategy.adaptation_enabled=True "
        f"acquisition_strategy.scale={scale} "
        f"acquisition_strategy.adaptation.lr_feat={adaptation_config.lr_feat} "
        f"acquisition_strategy.adaptation.lr_adj={adaptation_config.lr_adj} "
        f"acquisition_strategy.adaptation.epochs={adaptation_config.epochs} "
        f"acquisition_strategy.adaptation.seed={adaptation_config.seed} "
        f"acquisition_strategy.adaptation.mode={adaptation_config.mode} "
        f"acquisition_strategy.adaptation.integration={adaptation_config.integration} "
        f"wandb.name={wn} "
        f"> ../logs_new/{dataset}_{model}_{strategy}_adaptation_{wn}.log &"
    )
    p = f"{dataset}/{model}/{strategy}/{wn}"
    return s,p


def generate_prompt_tta(dataset,model,strategy,strat_node, strat_edge,num, filter,probs,scale, seed):
    f = "filter" if filter else "nofilter"
    p = "probs" if probs else "logits"
    s = (
        f"nohup python main.py model={model} data={dataset} "
        f"acquisition_strategy={strategy} data.num_splits=5 "
        f"model.num_inits=5 print_summary=True "
        f"model.cached=False "
        f"seed={seed} "
        f"acquisition_strategy.adaptation_enabled=False "
        f"acquisition_strategy.tta_enabled=True "
        f"acquisition_strategy.scale={scale} "
        f"acquisition_strategy.tta.strat_node={strat_node} "
        f"acquisition_strategy.tta.strat_edge={strat_edge} "
        f"acquisition_strategy.tta.num={num} "
        f"acquisition_strategy.tta.filter={filter} "
        f"acquisition_strategy.tta.probs={probs} "
        f"wandb.name=f{strat_node}_e{strat_edge}_{num}_{f}_{p} "
        f"> ../logs_new/{dataset}_{model}_{strategy}_tta_f{strat_node}_e{strat_edge}_{num}_{f}_{p}.log &"
    )
    return s