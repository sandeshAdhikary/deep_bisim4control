from src.train.train import run_train
import yaml
import optuna
from functools import partial
import numpy as np
from optuna.pruners import BasePruner
from optuna.trial import TrialState

def objective(trial, hyperparams_config, callback=None):

    if callback is not None:
        callback.before_trial()

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

    if trial.should_prune():
        raise optuna.TrialPruned()
    config['log_dir'] = 'logdir'
    config['logger_project'] = hyperparams_config['project_name']
    config['work_dir'] = config['log_dir']
    try:
        avg_ep_reward = run_train(config)    
    except (Exception, ValueError, AssertionError) as e:
        print(e)
        avg_ep_reward = np.nan

    if callback is not None:
        callback.after_trial()

    return avg_ep_reward

class TuningCallback():
    def __init__(self, config):
        self.project_name = config.get('project_name', None)

    def before_trial(self):
        pass

    def after_trial(self):
        pass        
        # # Delete synced wandb files to prevent storage overflow
        # if self.delete_old_studies and (self.project_name is not None):
        #     os.system(f'echo "y" | wandb sync --project {project_name} --clean --clean-old-hours 1')

        


class RepeatPruner(BasePruner):
    """
    https://stackoverflow.com/questions/58820574/how-to-sample-parameters-without-duplicates-in-optuna
    """
    
    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        trials = study.get_trials(deepcopy=False)
        
        numbers=np.array([t.number for t in trials if t.state in [TrialState.COMPLETE, 
                                                                                   TrialState.PRUNED,
                                                                                   TrialState.RUNNING,
                                                                                   ]])

        # Check if there exists trials with params from complete,pruned or running trials
        successful_trials = [trial.params==t.params for t in trials if t.state in [TrialState.COMPLETE, 
                                                                                   TrialState.PRUNED,
                                                                                   TrialState.RUNNING,
                                                                                   ]]
      
        # Don´t evaluate function if another with same params has been/is being evaluated before this one
        bool_params= np.array(successful_trials).astype(bool)
        if np.sum(bool_params)>0:
            if trial.number>np.min(numbers[bool_params]):
                return True

        return False




if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help="Path to hyperparams config file")
    argparser.add_argument('--delete-old-studies', action='store_true', help="Delete old studies")
    args = argparser.parse_args()

    # hyperparam_config_file = 'tune_hyperparams_config.yaml
    hyperparams_config = yaml.safe_load(open(args.config, 'r'))
    project_name = hyperparams_config['project_name']
    
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
                                pruner=RepeatPruner()
                                )
    
    tuning_callback = TuningCallback(config={
        'project_name': project_name,
    })

    partial_objective = partial(objective, hyperparams_config=hyperparams_config, callback=tuning_callback)
    study.optimize(partial_objective, n_trials=hyperparams_config['n_trials'])