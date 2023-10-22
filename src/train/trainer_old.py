from src.utils import utils
from src.train.setup_env import setup_env
import gym
import os
import torch
from src.envs.vec_wrappers import VecEnvWrapper
from src.callbacks import TrainingCallback
from src.train.make_agent import make_agent
from src.logger import Logger
import numpy as np
from src.train.load_args import parse_args
import pytz
from datetime import datetime
from tqdm import trange
import shutil
import zipfile
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress,MofNCompleteColumn, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
import time
import multiprocessing as mp
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import print
# import io
# import sys
# from rich import print

class RLTrainer():

    def __init__(self, config, logger_sw=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        
        # The current values of the following are tracked
        self.reward = None
        self.obs = None
        self.done = None
        self.step = None

        self._setup_config(config)
        self._set_seeds()
        self._setup_env()
        self._setup_logger(summary_writer=logger_sw)
        self._setup_callbacks()
        self._setup_replay_buffer()
        self._setup_agent()
            
    def train(self):
        self.before_train()
        eval_process = None

        # stdout_buffer = io.StringIO()
        # sys.stdout = stdout_buffer
        # with self.progress as progress:
        # with Live(self.layout, screen=False, refresh_per_second=1) as live:
        with self.progress:
                while not self.train_end:
                    
                    # Pre-step callbacks
                    self.before_step()

                    # Collect rollout
                    action, next_obs, reward, terminated, truncated, info = self.collect_rollout(self.obs)

                    # Update the agent
                    num_agent_updates = self.update_agent()

                    # Update current values
                    self.reward = reward
                    self.obs = next_obs
                    self.done = terminated | truncated
                    self.step += self.env.num_envs

                    if self.step % 10 == 0:
                        if eval_process is not None:
                            eval_process.join()
                        eval_process = mp.Process(target=self.evaluate_agent)
                        eval_process.start()
                    
                    # self.footer_text = Text(f"Step: {self.step}")
                    # live.get_renderable()['footer']._renderable.renderable = f"Step: {self.step}"

                    # Post-step callback
                    self.after_step({
                        'num_agent_updates': num_agent_updates
                    })

                    self.progress.update(self.prog_train_steps, advance=self.env.num_envs)
                
                self._after_train()
        

    def evaluate_agent(self):
        for idx in range(2):
            print(f"\n \t Eval {idx}")
            time.sleep(10)


    def collect_rollout(self, obs):

        curr_reward = self.reward
        
        # Get Action
        if self.step < self.config.init_steps:
            action = self.env.action_space.sample()
        else:
            with utils.eval_mode(self.agent):
                action = self.agent.sample_action(obs, batched=True)
        
        # Env Step
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add experience to buffer
        self._add_to_replay_buffer(obs, action, curr_reward, reward, next_obs, terminated, info)
        for idx in range(self.env.num_envs):
            self.episode_reward[idx] += reward[idx].item()


        return action, next_obs, reward, terminated,truncated, info

    def update_agent(self):

        num_updates = 0
        if self.step == self.config.init_steps:
            num_updates = self.config.init_steps 
        elif self.step > self.config.init_steps:
            num_updates = self.env.num_envs

        for idu in range(num_updates):
            # self.prog_bar.set_description(f"Training: Update: {idu+1}/{num_updates}")
            self.agent.update(self.replay_buffer, self.logger, self.step)

        return num_updates

    def before_train(self, info=None):
        """"
        Initializes the trainer to being training
        Resets the env, counters, progress bar
        """
        # This will be set if training ended prematurely
        self.train_end = False
        self.train_end_exception = None

        # Initialize the environment
        self.obs, _ = self.env.reset()
        self.done = [False]*self.env.num_envs
        self.reward = [0]*self.env.num_envs

        # Set up counters
        self.start_time = datetime.now(pytz.timezone('US/Eastern'))
        self.num_episodes = np.zeros(self.env.num_envs)
        self.episode_reward = np.zeros(self.env.num_envs)
        self.episode_reward_list = [[] for _ in range(self.env.num_envs)]
        self.num_model_updates = 0
        self.step = 0

        # Optionally checkpoint
        # this will update trainer state, agent state, and replay buffer
        if hasattr(self.config, 'load_checkpoint') and self.config.load_checkpoint:
            self._load_checkpoint()

        # TODO: Prog bar num steps should not come from the agent
        self.console = Console()
        self.layout = Layout()

        # Run Info header panel
        header_panel = Panel(f""" Project: [orange4] {self.logger._sw.project} [/]\
                             Sweep: [orange4] {self.logger._sw.sweep_id} [/]\
                                Run: [orange4] {self.logger._sw.id} [/]""",
                                title='Run Info', border_style='orange4')

        # Progress Panel
        self.progress = Progress(
            TextColumn("Overall Progress"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            redirect_stdout=False
            )
        self.prog_train_steps = self.progress.add_task("[red] Training...", total=self.config.num_train_steps)

        progress_panel = Panel.fit(Columns([self.progress]), title="Progress", border_style="dark_sea_green4")

        # Misc info panel
        self.footer_text = Text("Start")
        footer_panel = Panel(self.footer_text,
                             title="Misc", border_style='red')

        # Add panels to layout
        self.layout.split(
            Layout(header_panel, size=3, name="header"),
            Layout(progress_panel, size=5, name="main"),
            Layout(footer_panel, size=3, name="footer"),
        )



        
    def after_train(self, info=None):
        pass

    def before_step(self, info=None):
        pass

    def after_step(self, info=None):

        # Add to the episode rewards
        for idx in range(self.env.num_envs):
            self.episode_reward[idx] += self.reward[idx].item()

        # Add to model_updates count
        self.num_model_updates += info['num_agent_updates']

        # If any env is done, update episode counters
        if any(self.done):
            # Update num episode counts
            self.num_episodes += self.done
            # Update episode rewards
            for idx in range(self.env.num_envs):
                # Record episode reward in list
                self.episode_reward_list[idx].append(self.episode_reward[idx])
                # Clear current episode reward tracker if done
                self.episode_reward[idx] = 0
        
        # Log training metrics
        self.logger.log('train/episode', sum(self.num_episodes), self.step)
        self.logger.log('train/num_model_updates', self.num_model_updates, self.step)


        # Log episode level training metrics
        if self.step > 0 and (self.step % self.config.eval_freq == 0):
            # Only log if all environments have completed at least one episode
            if all([x >= 1 for x in self.num_episodes]):
                # avg_ep_reward is the average of last episode rewards for all envs
                avg_ep_reward = np.mean([x[-1] for x in self.episode_reward_list])
                self.logger.log('train/episode_reward', avg_ep_reward, self.step)


        # Update the progress bar
        # self.prog_bar.set_description(f"Step: {self.step}/{self.config.num_train_steps}")


        # Run callbacks
        early_stop = self._run_train_callbacks()

        # Should training end after this step?
        self.train_end = early_stop or (self.step >= self.config.num_train_steps)


        # Checkpoint model
        self._save_checkpoint()

    def _save_checkpoint(self):
        
        # Save the agent state
        self.agent.save_checkpoint(self.chkpt_dir)

        # Save the trainer state
        trainer_state = {'obs': self.obs,
                         'done': self.done,
                         'reward': self.reward,
                         'train_end': self.train_end,
                         'train_end_exception': self.train_end_exception,
                         'start_time': self.start_time,
                         'num_episodes': self.num_episodes,
                         'episode_reward': self.episode_reward,
                         'episode_reward_list': self.episode_reward_list,
                         'num_model_updates': self.num_model_updates,
                         'step': self.step,
                        #  'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'), # Resume on resource available
                         }

        torch.save(trainer_state, f'{self.chkpt_dir}/trainer_chkpt.pt')

        # Save the replay buffer: 
        # This will only save the chunk of buffer after last save, 
        # and upto the current step
        buffer_save_folder = f'{self.chkpt_dir}/replay_buffer_chkpt'
        if not os.path.exists(buffer_save_folder):
            os.makedirs(buffer_save_folder)
        self.replay_buffer.save(buffer_save_folder)
        
        
        # Compress checkpoint folder to a single file. Then delete the chkpt directory
        chkpt_file_name = f'{self.logger._logdir}/checkpoint'
        shutil.make_archive(chkpt_file_name, 'zip', self.chkpt_dir)
        # shutil.rmtree(chkpt_dir)

        # Log the checkpoint files to the logger
        self.logger.log_checkpoint(f"{chkpt_file_name}.zip")

    def _load_checkpoint(self):
        chkpt_folder_name = f'{self.logger._logdir}/checkpoint'

        # Download checkpoint from logger
        self.logger.restore_checkpoint()


        with zipfile.ZipFile(f"{chkpt_folder_name}.zip", 'r') as zip_ref:
            zip_ref.extractall(chkpt_folder_name)
        
        # Load the agent
        self.agent.load_checkpoint(chkpt_folder_name)

        # Load the replay buffer
        self.replay_buffer.load(f'{chkpt_folder_name}/replay_buffer_chkpt')

        # Load the trainer state
        self._load_trainer_state(f'{chkpt_folder_name}/trainer_chkpt.pt')


    def _add_to_replay_buffer(self, obs, action, curr_reward, reward, next_obs, terminated, info):
        # Allow infinite bootstrap: In the buffer, don't store truncated as done
        buffer_sample = obs, action, curr_reward, reward, next_obs, terminated
        if hasattr(self.replay_buffer, 'infos') and (self.replay_buffer.infos is not None):
            buffer_sample = (*buffer_sample, info)
        self.replay_buffer.add(*buffer_sample, batched=True) # batched because multi_env

        self._buffer_snapshot()

    def _buffer_snapshot(self):
        pass



    def _log_train_step(self):
        pass
        # self.logger.log('train/episode', total_eps, self.step)
        # self.logger.log('train/num_model_updates', num_model_updates, self.step)
        # pass


    def _load_trainer_state(self, load_dir):

        trainer_state = torch.load(load_dir)
        for key,val in trainer_state.items():
            setattr(self, key, val)

    #####

    def _after_train(self):
        avg_ep_reward = self.training_callback.after_train(self.agent, self.logger, self.step)
        self.logger.finish()
        return avg_ep_reward

    def _run_train_callbacks(self):                        
        early_stop = False
        return early_stop


    def _update_counters(self, done):
        self.episode += done 

        # Update episode rewards
        for idx in range(self.env.num_envs):
            if done[idx]:
                # Record episode reward in list
                self.episode_reward_list[idx].append(self.episode_reward[idx])
                # Clear current episode reward tracker if done
                self.episode_reward[idx] = 0


    def _setup_config(self, config):
        self.config = parse_args(config)
        

    def _setup_agent(self):
        # Create the agent
        self.agent = make_agent(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            args=self.config,
            device=self.device,
        )

    def _set_seeds(self):
        """
        Set seeds for random, numpy, cuda, torch
        """
        utils.set_seed_everywhere(self.config.seed)

    def _setup_env(self):
        self.env, self.eval_env, self.env_callback = setup_env(self.config)
    
        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        # If VecEnvs, set obs and action shapes to be that of a single env
        if isinstance(self.env, VecEnvWrapper):
            self.obs_shape = self.env.base_observation_space
            self.action_shape = self.env.base_action_space
        elif isinstance(self.env, (gym.vector.AsyncVectorEnv, gym.vector.SyncVectorEnv, gym.vector.VectorEnv)):
            self.obs_shape = self.obs_shape[1:] # skip the num_envs dimension
            self.action_shape = self.action_shape[1:] # skip the num_envs dimension
        else:
            ValueError("The env must be wrapped in a VecEnvWrapper")

    def _setup_logger(self, summary_writer=None):

        logger_config = {
            'dir': self.config.dir,
            'img_downscale_factor': self.config.logger_img_downscale_factor,
            'tracked_params': self.config.__dict__,
            'logger_tags': self.config.logger_tags if hasattr(self.config, 'logger_tags') else None
        }

        if self.config.logger == 'tensorboard':
            logger_config.update({
                'sw': 'tensorboard', 
                'format_config': 'rl'
                })
        elif self.config.logger == 'wandb':
            logger_config.update({
                'sw': 'wandb',
                'project': self.config.logger_project,
                })
        self.logger = Logger(logger_config, summary_writer=summary_writer)

        self.chkpt_dir = f"{self.logger._logdir}/checkpoint"
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)


    def _setup_callbacks(self):

        if not all([hasattr(self, x) for x in ['env', 'eval_env', 'env_callback']]):
            raise ValueError("Call _setup_env before setting up callbacks")

        minimal_logging = self.config.logger_minimal
        sub_callbacks = [] if minimal_logging else [self.env_callback]
        training_callback = TrainingCallback({
            'project_name': self.config.logger_project,
            'save_video': self.config.save_video,
            'train_args': self.config,
            'save_model_mode': self.config.save_model_mode,
            'num_eval_episodes': self.config.num_eval_episodes,
            'callbacks': sub_callbacks,
            'log_video_freq': self.config.logger_video_log_freq,
            'sweep_config': self.config.sweep_config,
            'logdir': self.logger._logdir
        })
        training_callback.set_env(self.eval_env) # Will also set env for domain_callback

    def _setup_replay_buffer(self):
        # Set up the replay buffer
        self.replay_buffer = utils.ReplayBuffer(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            capacity=self.config.replay_buffer_capacity,
            batch_size=self.config.batch_size,
            device=self.device,
            store_infos=self.config.agent.endswith('decomp')
        )

    


