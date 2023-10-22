import numpy as np
import optuna
import pytz
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
tz = pytz.timezone('America/Los_Angeles')
from src.utils.optuna_utils.storage import CustomRDBStorage
from src.train.train import run_train
import yaml
import optuna
from functools import partial
import numpy as np
from optuna.trial import TrialState
from mysql import connector
from src.defaults import DEFAULTS
from utils.utils import get_hash_id
import argparse
from src.sweeper.sweeper_old import Sweeper

DEFAULT_OPTUNA_STORAGE = DEFAULTS['OPTUNA_STORAGE']

with open('src/studies/default_config.yaml', 'r') as f:
    DEFAULT_HYPERPARAMS = yaml.safe_load(f)


def objective(trial, hyperparams_config, callback=None):

    # Set sweep metadata
    for key, value in hyperparams_config['sweep_info'].items():
        trial.set_user_attr(key, value)


    # Load the default config from default_config
    config = DEFAULT_HYPERPARAMS.copy()

    # Update with any base_config params provided in the hyperparams_config file
    base_config = hyperparams_config.get('base_config')
    if base_config is not None:
        config.update(base_config)



    # Update config with trial params
    hyperparams = hyperparams_config['hyperparams']
    run_name_string = f"{trial.study._study_id}_{trial.number}"
    for param, param_config in hyperparams.items():
        suggestion = None
        if param_config['type'] == 'float':
            log = param_config.get('log', False)
            suggestion = trial.suggest_float(param, param_config['low'], param_config['high'], log=log)
        elif param_config['type'] == 'int':
            step = param_config.get('step', 1)
            log = param_config.get('log', False)
            suggestion = trial.suggest_int(param, param_config['low'], param_config['high'], step=step, log=log)
        elif param_config['type'] == 'categorical':
            suggestion = trial.suggest_categorical(param, choices=param_config['options'])
        elif param_config['type'] == 'fixed':
            # Update base config with fixed value
            suggestion = param_config['value']

        if suggestion is not None:
            config[param] = suggestion
            run_name_string = "-".join((run_name_string, f"{param}_{suggestion}"))

    # Get run_id
    project_name = hyperparams_config['project_name']
    sweep_name = hyperparams_config['sweep_info']['sweep_name']
    run_id = get_hash_id(run_name_string)
    config['run_id'] = run_id

    avg_ep_reward = np.nan
    try:        

        if callback is not None:
            callback.before_trial(trial)

        config['log_dir'] = f"logdir/{project_name}/{sweep_name}/run_{run_id}"
        config['logger_project'] = project_name

        # Log the sweep metadata
        config['sweep_info'] = hyperparams_config['sweep_info']

        # Add optuna trial info to trainer
        config['sweep_config'] = {
            'study_name': project_name,
            'sweep_name': sweep_name,
            'optuna_trial': trial,
            'optuna_storage': hyperparams_config['study_storage_url'],
        }

        avg_ep_reward, exception = run_train(config)

        if exception is not None:
            raise exception
        
    except (Exception, ValueError, AssertionError, KeyboardInterrupt) as e:
        error_msg = e.__class__.__name__
        if len(e.args) > 0:
            error_args = '_'.join([x for x in e.args])
            error_msg = '_'.join([error_msg, error_args])
        trial.set_user_attr("fail_reason", error_msg)
        if isinstance(e, optuna.TrialPruned):
            raise e

    if callback is not None:
        callback.after_trial()

    return avg_ep_reward

class TuningCallback():
    def __init__(self, config):
        self.project_name = config.get('project_name', None)
        self.run_backup_script = config.get('run_backup_script', False)
        self.pruner = config.get('pruner', None)
        
    def before_trial(self, trial=None):
        if (trial is not None) and self.is_trial_repeated(trial):
            # Check if trial should be pruned based on params (e.g. repeated config already in database)
            trial.set_user_attr("repeated_trial", True)
            raise Exception("Repeated Trial")

    def after_trial(self, trial=None):
        # Update sweeper info
        pass

    def is_trial_repeated(trial):
        """
        https://stackoverflow.com/questions/58820574/how-to-sample-parameters-without-duplicates-in-optuna
        """

        trials = trial.study.get_trials(deepcopy=False)
        
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


def dry_run_args_update(hyperparams_config):
    hyperparams_config['project_name'] = hyperparams_config['project_name'] + '_dryrun'

    hyperparams_config['base_config']['episode_length'] = 5
    hyperparams_config['base_config']['init_steps'] = 5
    hyperparams_config['base_config']['eval_freq'] = 3
    hyperparams_config['base_config']['num_train_steps'] = 20

    print(f"Doing a dry run of the sweep")

    return hyperparams_config


def study_exists(study_name, optuna_storage):
    # Check if study exists in the storage
    username, storage_str = optuna_storage.split("://")[1].split(":")
    password, storage_str = storage_str.split("@")
    host, database = storage_str.split("/")
    mysql_config = {
        'user': username,
        'password': password,
        'host': host,
        'database': database,
    }
    cnx = connector.connect(**mysql_config)
    cursor = cnx.cursor()

    query = f"SELECT * FROM studies WHERE study_name='{study_name}'"
    cursor.execute(query)
    tables = cursor.fetchall()
    return len(tables) > 0

def create_sampler(hyperparams_config):
    # Set the sampler
    sampler_type = hyperparams_config['sweep_info']['sampler']
    if sampler_type == 'random':
        sampler = optuna.samplers.RandomSampler()
    elif sampler_type == 'grid':
        # Define search space
        search_space = {}
        for param, param_config in hyperparams_config['hyperparams'].items():
            if param_config['type'] == 'categorical':
                param_options = param_config['options']
                search_space[param] = param_options
        sampler = optuna.samplers.GridSampler(search_space)
    else:
        raise ValueError("Unknown Sampler")
    return sampler

def create_pruner(hyperparams_config):
    # Set the pruner
    pruner_type = hyperparams_config['sweep_info'].get('pruner')
    if pruner_type == 'hyperband':
        raise NotImplementedError
    elif pruner_type == 'asha':
        min_iters = hyperparams_config['sweep_info'].get('pruner_min_iter', None)
        if min_iters is None:
            # use 10% of max_iters as default
            min_iters = int(0.1*hyperparams_config['base_config']['num_train_steps'])
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=min_iters)
    else:
        print("No pruner specified, using None")
        return None

    return pruner

def get_hyperparams_config():
    # Load arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help="Path to hyperparams config file")
    argparser.add_argument('--dryrun', action='store_true', help="Dry run", default=False)
    args = argparser.parse_args()
    hyperparams_config = yaml.safe_load(open(args.config, 'r'))

    if args.dryrun:
        hyperparams_config = dry_run_args_update(hyperparams_config)

    optuna_storage_url = hyperparams_config.get('optuna_storage_url')
    if optuna_storage_url is None:
        # Use the default storage and update the hyperparams_config
        optuna_storage_url=DEFAULT_OPTUNA_STORAGE
        hyperparams_config['study_storage_url'] = optuna_storage_url

    return hyperparams_config

def setup_storage(optuna_storage_url):
    """
    The CustomRDBStorage class will create tables for sweeps if
    they don't exist
    """
    optuna_storage = CustomRDBStorage(url=optuna_storage_url)
    return optuna_storage

def setup_study(study_name, storage_url):
    """
    Sets up the study without a pruner or sampler
    """
    if study_exists(study_name, storage_url):
        # Retrieve study without pruner or sampler
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    else:
        # Create study without pruner or sampler
        study = optuna.create_study(direction='maximize', 
                                    study_name=study_name,
                                    storage=storage_url,
                                    load_if_exists=False)
    return study

if __name__ == "__main__":
    # Get hyperparams config
    hyperparams_config = get_hyperparams_config()
    study_name = hyperparams_config['project_name']
    storage_url = hyperparams_config['study_storage_url']
    
    # Set up the storage database
    optuna_storage = setup_storage(storage_url)

    # Set up the project/study
    study = setup_study(study_name, storage_url)
    
    # Define sweeper based on the study
    sweeper = Sweeper(study, hyperparams_config)
    hyperparams_config['sweep_info']['sweep_name'] = sweeper.sweep_name
    # Define pruner and sampler
    pruner = create_pruner(hyperparams_config)
    sampler = create_sampler(hyperparams_config)
    # Load the study with sampler and pruner
    study = optuna.load_study(study_name=study_name, storage=storage_url, pruner=pruner, sampler=sampler)

    partial_objective = partial(objective, hyperparams_config=hyperparams_config)
    study.optimize(partial_objective)