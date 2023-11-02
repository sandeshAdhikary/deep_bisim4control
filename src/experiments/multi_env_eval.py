import yaml
from trainer.rl.rl_evaluator import RLEvaluator
from trainer import Logger, Model, Sweeper
from trainer.rl import RLTrainer
import numpy as np
import json
import hashlib

# Project specific imports
from src.train.setup_env import make_env
from src.train.make_agent import make_agent

from typing import Dict
import os
from copy import copy, deepcopy
import mysql.connector
from trainer.storage import Storage
from warnings import warn
from tqdm import tqdm

class BisimModel(Model):

    def __init__(self, config: Dict):
        """
        Define self.model here
        """
        super().__init__(config)

        model_config = copy(config)
        self.model = make_agent(
            obs_shape=model_config.pop('obs_shape'),
            action_shape=model_config.pop('action_shape'),
            device=model_config.get('device'),
            args=model_config,
        )

    @property
    def module_path(self):
        return 'src.sweep'
        
    def parse_config(self, config):

        # Set encoder output dim to be feature dim if not set
        if not isinstance(config['encoder_output_dim'], int):
            config['encoder_output_dim'] = config['encoder_feature_dim']

        # If using cluster encoders, num_rews is the same as encoder's output dim
        if 'cluster' in config['encoder_type']:
            config['reward_decoder_num_rews'] = config['encoder_output_dim']

        return config

    def training_step(self, batch, batch_idx, step):
        return self.model.update(batch, step)
    
    def state_dict(self, **kwargs):
        return self.model.state_dict(**kwargs)
    
    def save_model(self, filename, save_optimizers=True):
        self.model.save(model_dir=os.path.dirname(filename),
                        filename=os.path.basename(filename),
                        save_optimizers=save_optimizers
                        )

    def load_model(self, state_dict=None, model_file=None, model_dir=None, chkpt_name=None):
        if state_dict is not None:
            self.model.load(state_dict=state_dict)
        elif model_file is not None:
            self.model.load(model_file=model_file)
        else:
            chkpt_name = chkpt_name or 'eval_checkpoint'
            self.model.load(model_dir=model_dir, step=chkpt_name)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def zero_grad(self):
        self.model.zero_grad()

    def sample_action(self, obs, batched=False):
        """
        Sample action from model. May be non-deterministic
        """
        return self.model.sample_action(obs, batched=batched)

    def select_action(self, obs, batched=False):
        """
        Select action from model. Should be deterministic
        """
        return self.model.select_action(obs, batched=batched)


class BisimRLEvaluator(RLEvaluator):
    def make_env(self, config):
        return make_env(config)
    
    @property
    def module_path(self):
        return 'src.experiments.cartpole.multi_env_eval'
    
    def after_eval(self, info):
       
        mysql_host = os.environ['SSH_HOST']
        mysql_user = os.environ['SSH_USERNAME']
        mysql_password = os.environ['MYSQL_PASSWORD']
        mysql_db = 'bisim'

        db_conn = mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_db
        )
        cursor = db_conn.cursor()
        

        # Create runs table
        cursor.execute(f"SHOW TABLES LIKE 'runs'")
        experiment_table_exists = cursor.fetchone()
        if experiment_table_exists is None:
            cursor.execute(f"""CREATE TABLE runs (
                           run_id VARCHAR(255),
                           sweep_id VARCHAR(255),
                           project_id VARCHAR(255),
                           folder VARCHAR(255),
                           PRIMARY KEY (run_id, sweep_id)
            )""")
            db_conn.commit()



        # Add run to runs table if not already present
        cursor.execute(f"SELECT * FROM runs WHERE run_id='{self.run}' AND sweep_id='{self.sweep}'")
        run_exists = cursor.fetchone()
        if run_exists is None:
            cursor.execute(f"""INSERT INTO runs \
                        (run_id, sweep_id, project_id, folder) \
                        VALUES ('{self.run}', '{self.sweep}', '{self.project}', '{self.output_storage.dir}') \
                        """)

        db_conn.commit()
        metrics = ['episode_rewards']

        for metric in metrics:

            metric = 'episode_rewards'
            # Check if metric table exist; if not, create metric table
            cursor.execute(f"SHOW TABLES LIKE '{metric}'")
            metric_table_exists = cursor.fetchone()
            if metric_table_exists is None:
                cursor.execute(f"""CREATE TABLE {metric} (
                               run_id VARCHAR(255),
                               sweep_id VARCHAR(255),
                               project_id VARCHAR(255),
                               eval_id VARCHAR(255),
                               eval_name VARCHAR(255),
                               value FLOAT,
                               value_std FLOAT,
                               step INT,
                               PRIMARY KEY (run_id, eval_id, step)
                )""")
                db_conn.commit()

            # Get env_id from info
            # If two envs have the same config except for their seed, they get the same env_id
            env_ids = []
            for env in info.keys():
                data = info[env][metric]
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                data_mean = np.mean(data, axis=1)
                data_std = np.std(data, axis=1)
                # Get env id: hash the env config
                env_config = deepcopy(self.config['envs'][env])
                env_config.pop('seed') # Remove seed before hasing
                env_config = json.dumps(env_config)
                env_id = hashlib.sha256(env_config.encode())
                env_id = env_id.hexdigest()
                env_ids.append(env_id)
                for idx in range(len(data)):
                    d_mean = data_mean[idx]
                    d_std = data_std[idx]
                    cursor.execute(f"""INSERT INTO {metric} \
                                (run_id, sweep_id, project_id, eval_id, eval_name, step, value, value_std) \
                                VALUES ('{self.run}', '{self.sweep}', '{self.project}', '{env_id}', '{env}', {idx}, {float(d_mean)}, {float(d_std)}) \
                                ON DUPLICATE KEY UPDATE value = {float(d_mean)}, value_std = {float(d_std)} \
                                """)
            db_conn.commit()


class MultiDistractorExperiment():
    """
    Experiment to test the generalization of RL models on multiple distrctor environments
    """


    def __init__(self, config, evaluator_cls, model_cls):
        self.config = config
        self.exp_config = config['experiment']
        self.project = self.exp_config['project']
        self.evaluator_cls = evaluator_cls
        self.model_cls = model_cls   
        # Connect to storage where runs are saved
        # self.storage = Storage(config['experiment']['storage'])     

    def train(self, config):
        pass

    def train_sweep(self, config):
        pass

    def evaluate(self, run, sweep=None):
        """
        Run evaluator on a single run
        output: results.pt
        """

        # Create evaluator config
        evaluator_config = deepcopy(self.config['evaluator'])
        evaluator_config['run'] = run
        evaluator_config['sweep'] = sweep
        # Set input storage to run: to download saved model
        evaluator_config['storage']['input']['run'] = run
        evaluator_config['storage']['input']['sub_dir'] = 'train'
        
        # Set output storage to run: to upload results
        evaluator_config['storage']['output']['overwrite'] = True
        evaluator_config['storage']['output']['run'] = run
        evaluator_config['storage']['output']['sub_dir'] = 'eval'
        if sweep is not None:
            evaluator_config['storage']['input']['sweep'] = sweep
            evaluator_config['storage']['output']['sweep'] = sweep
        


        evaluator = self.evaluator_cls(evaluator_config)

        # Load the run's model
        model_config = evaluator.input_storage.load('model_config.yaml', 
                                                    filetype='yaml')
        model_state_dict = evaluator.input_storage.load_from_archive('ckpt.zip', 
                                                        filenames=['model_ckpt.pt'],
                                                        filetypes=['torch'])
        
        # # Save the logger ckpt to the output storage
        # evaluator.output_storage.save('logger_ckpt.pt', model_state_dict['logger_ckpt.pt'], filetype='torch')


        model=model_cls(model_config)
        model.load_model(state_dict=model_state_dict['model_ckpt.pt'])

        

        evaluator.set_model(model)
        evaluator.run_eval()

    def analyze(self, run):
        """
        Analyze the output of evaluation run
        Convert raw data of eval run into vega-lite
        digestible format
        e.g. can be uploaded to wandb as artificat
        to make reports
        """
        pass

    def plot(self, run):
        """
        Plot the analysis of the evaluation run
        output: plot.json
        """
        pass

    #######

        


if __name__ == "__main__":
    import argparse
    from envyaml import EnvYAML
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--run', type=str, required=True)
    # parser.add_argument('--sweep', type=str)
    args = parser.parse_args()

    config = dict(EnvYAML(args.config))
    evaluator_cls = BisimRLEvaluator
    model_cls = BisimModel

    experiment = MultiDistractorExperiment(config, 
                                           evaluator_cls=evaluator_cls,
                                           model_cls=model_cls
                                           )
    
    # Get list of sweeps and runs
    # runs = {
    #     config['experiment']['project'] : {
    #         'dbc' : ['vu1p1ooc'],
    #         'dbc_mnist': ['hdkhw10u'],
    #         'spectral': ['5bt5v8zk'],
    #         'spectral_mnist': ['lhl4z9ek']
    #     }
    # }

    # runs = {
    #     'cheetah' : {
    #         'dbc' : ['vu1p1ooc', '6wnwgoqz'],
    #         'dbc_mnist': ['hdkhw10u','sa39esuy'],
    #         'spectral': ['5bt5v8zk','08ilqgne'],
    #         'spectral_mnist': ['lhl4z9ek','dlhtncgp']
    #     }
    # }

    runs = config['experiment']['runs']

    for sweep in tqdm(runs.keys()):
        for run in runs[sweep]:
            try:
                print(f"Running run {run} in sweep {sweep}")
                experiment.evaluate(run=run, sweep=sweep)
            except FileNotFoundError:
                print(f'Could not find run {run} in sweep {sweep}')
