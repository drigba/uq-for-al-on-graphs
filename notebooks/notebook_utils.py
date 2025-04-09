import os
import pandas as pd
from graph_al.test_time_adaptation.config import AdaptationConfig
import plotly.graph_objects as go
import torch
import numpy as np

def load_results(dataset, model, strategies_names,save = False, cached = False, cache_path = None):
    
    if cached and cache_path is not None:
        print("Loading cached metrics")
        if not os.path.exists(cache_path):
            raise ValueError(f"Cache path {cache_path} does not exist.")
        metrics_path = os.path.join(cache_path, "metrics_dict.pt")
        metrics_dict = torch.load(metrics_path)
        return metrics_dict
    
    prefix = f"../output2/runs/{dataset}/{model}/"
    # strategies_names = ['aleatoric_propagated', "educated_random", "augment_latent"]
    # strategies_names = [ "educated_random"]

    strategies_paths = [os.path.join(prefix, strategy) for strategy in strategies_names]
    metrics_dict = {}
    # plt.figure(figsize=(20,10))
    print("Loading metrics")
    for ix,strategies_path in enumerate(strategies_paths):
        strategies = os.listdir(strategies_path)
        for strategy in strategies:
            path = os.path.join(strategies_path, strategy)
            for run in os.listdir(path):
                if os.path.exists(os.path.join(path, run, "acquisition_curve_metrics.pt")):
                    metrics_path = os.path.join(path, run, "acquisition_curve_metrics.pt")
                    print(metrics_path)
                    metrics = torch.load(metrics_path, weights_only=True)
                    accuracy = np.array(metrics["accuracy/test"])*100
                    accuracy_mean, accuracy_std = np.mean(accuracy, axis=0), np.std(accuracy, axis=0)
                    metrics_dict[strategies_names[ix] + "_" + strategy] = (accuracy_mean,accuracy_std,accuracy,metrics)
    if save and cache_path is not None:
        print("Saving metrics to cache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        torch.save(metrics_dict, os.path.join(cache_path, "metrics_dict.pt"))
    return metrics_dict


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
        f"> logs_new/{dataset}_{model}_{strategy}_adaptation_{wn}.log &"
    )
    p = f"{dataset}/{model}/{strategy}/{wn}"
    return s,p


def generate_prompt_tta(dataset,model,strategy,strat_node, strat_edge,num, filter,probs,scale, seed, p_node = "none", p_edge = "none"):
    f = "filter" if filter else "nofilter"
    pl = "probs" if probs else "logits"
    node_prob = f"acquisition_strategy.tta.p_node={p_node}" if p_node != "none" else ""
    edge_prob = f"acquisition_strategy.tta.p_edge={p_edge}" if p_edge != "none" else ""
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
        f"{node_prob} "
        f"{edge_prob} "
        f"acquisition_strategy.tta.filter={filter} "
        f"acquisition_strategy.tta.probs={probs} "
        f"wandb.name=f{strat_node}_e{strat_edge}_{num}_{f}_{pl}_{p_node}_{p_edge} "
        f"> logs_new/{dataset}_{model}_{strategy}_tta_f{strat_node}_e{strat_edge}_{num}_{f}_{pl}_{p_node}_{p_edge}.log &"
    )
    return s


def augmentation_name(x):
    if x == "fmask":
        return "Feature Mask"
    elif x == "fnoise":
        return "Feature Noise"
    elif x == "emask":
        return "Edge Mask"
    elif x == "fmask,emask":
        return "Feature & Edge Mask"
    elif x == "fnoise,emask":
        return "Feature Noise & Edge Mask"
    return None



# AGGREGATE GEEM METRICS
def combine_geem_metrics(dataset):
    metrics = []
    seed_path = os.path.join("..","output2/runs", dataset, "sgc", "geem")
    for seed in os.listdir(seed_path):
        if seed != "None":
            run_path = os.path.join(seed_path, seed)
            for run_dir in os.listdir(run_path):
                if run_dir != "hydra-outputs" and os.path.exists(os.path.join(run_path, run_dir, "acquisition_curve_metrics.pt")):
                    metrics_path = os.path.join(run_path, run_dir, "acquisition_curve_metrics.pt")
                    metric = torch.load(metrics_path, weights_only=True)
                    metrics.append(metric)
    aggregated_metrics = {k:np.array([m[k][0] for m in metrics]) if k != "acquired_idxs" else [m[k][0] for m in metrics]  for k in metrics[0].keys()}
    return aggregated_metrics
    
def aggregate_geem_metrics(dataset):
    metrics = combine_geem_metrics(dataset)
    metrics["accuracy/test"] *=100
    mean_metrics = {k:np.mean(v, axis=0) if k != "acquired_idxs" else v for k,v in metrics.items()}
    std_metrics = {k:np.std(v, axis=0) if k != "acquired_idxs" else v for k,v in metrics.items()}
    return mean_metrics, std_metrics

