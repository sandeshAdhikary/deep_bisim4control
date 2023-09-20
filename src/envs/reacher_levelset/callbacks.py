import torch 
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from encoder import _CLUSTER_ENCODERS
import pickle
import os
from envs.dmc2gym.callbacks import DMCCallback

class ReacherLevelSetEvalCallback(DMCCallback):

    def __init__(self, config=None):
        super().__init__(config)
        

    def log_artifacts(self, agent, logger, step):
        super().log_artifacts(agent, logger, step)
        