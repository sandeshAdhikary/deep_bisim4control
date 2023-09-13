from train import run_train
import yaml
import optuna
from functools import partial
import os
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import numpy as np
from optuna.pruners import BasePruner
from optuna.trial import TrialState

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


class RepeatPruner(BasePruner):
    """
    https://stackoverflow.com/questions/58820574/how-to-sample-parameters-without-duplicates-in-optuna
    """
    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        trials = study.get_trials(deepcopy=False)
        
        numbers=np.array([t.number for t in trials])
        bool_params= np.array([trial.params==t.params for t in trials]).astype(bool)
        #Don´t evaluate function if another with same params has been/is being evaluated before this one
        if np.sum(bool_params)>1:
            if trial.number>np.min(numbers[bool_params]):
                return True

        return False

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
                                load_if_exists=True,
                                pruner=RepeatPruner(),
                                )
    
    # Set up tracking callback with wandb
    # wandbc = WeightsAndBiasesCallback(wandb_kwargs={"project": project_name,})

    # study.optimize(partial_objective, n_trials=hyperparams_config['n_trials'], callbacks=[wandbc])
    study.optimize(partial_objective, n_trials=hyperparams_config['n_trials'])