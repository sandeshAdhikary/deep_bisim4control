from train import run_train
import yaml
import optuna
from functools import partial
import os
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import numpy as np

def objective(trial, hyperparams_config, log_dir):

    # Delete synced wandb files to prevent storage overflow
    os.system(f'echo "y" | wandb sync --clean --clean-old-hours 1')

    # config = yaml.safe_load(open(hyperparams_config['base_config'], 'r'))
    config = hyperparams_config['base_config']

    # Update config with trial params
    hyperparams = hyperparams_config['hyperparams']
    for param, param_config in hyperparams.items():
        if param_config['type'] == 'float':
            log = param_config.get('log', False)
            config[param] = trial.suggest_float(param, param_config['low'], param_config['high'], log=log)
        elif param_config['type'] == 'int':
            step = param_config.get('step', 1)
            log = param_config.get('log', False)
            config[param] = trial.suggest_int(param, param_config['low'], param_config['high'], step=step, log=log)
        elif param_config['type'] == 'categorical':
            config[param] = trial.suggest_categorical(param, choices=param_config['options'])
        elif param_config['type'] == 'fixed':
            # Update base config with fixed value
            config[param] = param_config['value']

    config['logger_project'] = hyperparams_config['project_name']
    config['work_dir'] = os.path.join(log_dir, f"trial_{str(trial._trial_id)}")
    try:
        avg_ep_reward = run_train(config)    
    except (Exception, ValueError, AssertionError) as e:
        print(e)
        avg_ep_reward = np.nan

    return avg_ep_reward


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help="Path to hyperparams config file")
    args = argparser.parse_args()

    # hyperparam_config_file = 'tune_hyperparams_config.yaml
    hyperparams_config = yaml.safe_load(open(args.config, 'r'))
    project_name = hyperparams_config['project_name']
    log_dir = os.path.join(hyperparams_config['log_dir'], project_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    partial_objective = partial(objective, hyperparams_config=hyperparams_config, log_dir=log_dir)
    
    # Set the sampler
    sampler_type = hyperparams_config['sampler']
    if sampler_type == 'random':
        sampler = optuna.samplers.RandomSampler(seed=123)
    elif sampler_type == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=123)
    else:
        raise ValueError("Unknown Sampler")

    study = optuna.create_study(direction='maximize', 
                                study_name=project_name,
                                storage=hyperparams_config['study_storage_url'],
                                load_if_exists=True)
    
    # Set up tracking callback with wandb
    wandbc = WeightsAndBiasesCallback(wandb_kwargs={"project": project_name,})

    study.optimize(partial_objective, n_trials=hyperparams_config['n_trials'], callbacks=[wandbc])