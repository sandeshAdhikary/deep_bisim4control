import os
import glob
import numpy as np
from einops import rearrange
import torch
from torch.utils.data import DataLoader

class ClusteringEvalCallback():
    def __init__(self, config) -> None:
        self.config = config
        self.batch_size = self.config.get('batch_size', 128)
        # Load dataset
        data_path = self.config['data_path']
        self.data = torch.load(data_path)[self.config['env_name']]

    def __call__(self, model, env, tracked_data=None):
        
        model.eval()
        data = DataLoader(self.data, batch_size=self.batch_size, shuffle=False)

        features = []
        values = []
        with torch.no_grad():
            for batch in data:
                batch = batch.to(model.device)
                # Get features from encoder
                batch_features = model.model.critic.encoder(batch) 
                # Get actions for features from actor
                action, _, _, _= model.model.actor(batch, compute_pi=False, compute_log_pi=False)
                # Get Q values for action and features
                batch_q1 = model.model.critic.Q1(batch_features, action)
                batch_q2 = model.model.critic.Q2(batch_features, action)
                batch_values = torch.min(batch_q1, batch_q2)
                features.append(batch_features)
                values.append(batch_values)
        features = torch.cat(features)
        values = torch.cat(values)

        return {
            'features': features.detach().cpu().numpy(),
            'values': values.detach().cpu().numpy()
            }
        