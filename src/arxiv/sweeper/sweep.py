import argparse
import yaml
import numpy as np
from functools import partial
import wandb
import os
from src.defaults import DEFAULTS
import socket, platform
import pytz
from datetime import datetime
import mysql.connector
from src.train.trainer_old import RLTrainer
from copy import deepcopy, copy

WANDB_RUN_QUEUE_PATH = DEFAULTS['WANDB_RUN_QUEUE_PATH']

with open('src/studies/default_config.yaml', 'r') as f:
    DEFAULT_HYPERPARAMS = yaml.safe_load(f)


def dry_run_args_update(sweep_config, hyperparams):
    # Update sweep_config
    sweep_config['project_name'] = sweep_config['project_name'] + '_dryrun'
    # Update hyperparams
    hyperparams['episode_length'] = 5
    hyperparams['init_steps'] = 5
    hyperparams['eval_freq'] = 3
    hyperparams['num_train_steps'] = 10000
    hyperparams['batch_size'] = 10

    return sweep_config, hyperparams

def parse_config(config, goal='maximize', dryrun=False):
    # Set up sweep_config 
    sweep_config = config['sweep_config']
    # Add sweep goal to config
    sweep_config['metric'] = {'name': 'sweep_score', 'goal': goal}

    # Set up default parameters
    default_params = DEFAULT_HYPERPARAMS.copy()
    default_params['retain_logger'] = True
    # Update with base config provided
    default_params.update(config.get('base_config') or {})
    
    # Set defaults to minimal train config if dry_run
    if dryrun:
        sweep_config, default_params = dry_run_args_update(sweep_config, default_params)


    return sweep_config, default_params

def get_run_from_queue(project_name, sweep_id):
    conn = mysql.connector.connect(
        host='10.19.137.42',
        user='wandb_user',
        password='wandbLine1!',
        database='wandb_run_queue'
    )
    cur = conn.cursor()
    result = cur.execute(f"""
                         SELECT run_id, project_name, sweep_id, MAX(start_time) as last_time
                         FROM run_queue
                         WHERE project_name='{project_name}' AND sweep_id='{sweep_id}'
                         GROUP BY project_name, sweep_id, run_id
                         ORDER BY last_time DESC;
                         """)
    result = cur.fetchall()
    cur.close()
    conn.close()
    if (result is not None) and len(result) > 0:
        return result[0][0]
    else:
        return None

class SweepCallback():

    def __init__(self, run):
        self.run = run
    
    def before_run(self):
        self._add_run_to_queue()

    def after_run(self):
        self._delete_run_from_queue()


    def _add_run_to_queue(self):
        # Add run and machine info to database
        machine_str = f"{platform.system()}--{platform.version().split(' ')[0]}--{platform.machine()}--{socket.getfqdn()}"
        run_id = self.run.id
        project_name = self.run.project
        sweep_id = self.run._sweep_id
        current_time = datetime.now(pytz.timezone('US/Pacific'))
        current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        conn = mysql.connector.connect(
            host='10.19.137.42',
            user='wandb_user',
            password='wandbLine1!',
            database='wandb_run_queue'
        )
        cur = conn.cursor()
        cur.execute(f"INSERT INTO run_queue (machine, run_id, project_name, sweep_id, start_time) VALUES ('{machine_str}', '{run_id}', '{project_name}', '{sweep_id}', '{current_time}')")
        conn.commit()
        cur.close()
        conn.close()

    def _delete_run_from_queue(self):
        run_id = self.run.id
        project_name = self.run.project
        sweep_id = self.run._sweep_id
        # Remove the run from the queue
        conn = mysql.connector.connect(
            host='10.19.137.42',
            user='wandb_user',
            password='wandbLine1!',
            database='wandb_run_queue'
        )
        cur = conn.cursor()
        cur.execute(f"DELETE FROM run_queue WHERE project_name = '{project_name}' AND sweep_id = '{sweep_id}' AND run_id = '{run_id}'")
        conn.commit()
        cur.close()
        conn.close()


def objective(project_name=None, default_params=None, run_id=None, sweep_callback=None):
    resume_run = run_id is not None
    with wandb.init(project=project_name, id=run_id, resume=resume_run) as run:
        if default_params is not None:
            # Set default params
            run.config.setdefaults(default_params)

        sweep_callback = SweepCallback(run)


        sweep_callback.before_run()
        
        # Set up the trainer with the run    
        config = copy(dict(wandb.config))
        config['load_checkpoint'] = run.resumed
        # config['load_checkpoint'] = False
        trainer = RLTrainer(config, logger_sw=run)

        sweep_score = trainer.train()
        # Run the objective
        # avg_ep_reward = run_train(wandb.config)
        wandb.log({"sweep_score": sweep_score})
        

        sweep_callback.after_run()
        

        

if __name__ == "__main__":

    # Load arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help="Path to sweep config file")
    argparser.add_argument('--dryrun', action='store_true', help="Dry run", default=False)
    argparser.add_argument('--run_from_queue', action='store_true', 
                           help="Finish runs in queue, before starting new runs", default=False)
    args = argparser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    # Get hyperparams config
    sweep_config, default_params = parse_config(config, dryrun=args.dryrun)

    # Check if sweep exists
    api = wandb.Api()
    project_name = sweep_config['project_name']
    sweep_name = sweep_config['name']
    
    try:
        project_sweeps = {x.name:x.id for x in wandb.Api().project(project_name).sweeps()}
    except:
        # TODO: Handle this better
        project_sweeps = {}

    if sweep_name in project_sweeps.keys():
        print(f"Loading existing sweep: {project_name}/{sweep_name}")
        sweep_id = project_sweeps[sweep_name]
        if args.run_from_queue:
            # Check for runs in the queue
            running_from_queue = True
            while running_from_queue:
                run_id = get_run_from_queue(project_name, sweep_id)
                if run_id is not None:
                    # Process the run from the queue
                    objective(project_name=project_name, default_params=default_params, run_id=run_id)
                else:
                    running_from_queue = False
            # Once queue is done, run the agent for new runs
    else:
        print(f"Creating new sweep in project {project_name}")
        # Create a new sweep id
        sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id, function=partial(objective, 
                                           default_params=default_params,
                                           project_name=project_name), count=10, project=project_name)
    
