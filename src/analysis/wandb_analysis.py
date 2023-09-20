import wandb
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import sem
from matplotlib.patches import Rectangle

api = wandb.Api()
encoder_modes = ['dbc', 'spectral']
img_shrink_factor = 2.0
distract_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
domain = 'gridworld'

for encoder_mode in encoder_modes:


    entity, project = "adhikary-sandesh", f"{domain}-invariance-exp-{encoder_mode}-v2"
    runs = api.runs(entity + "/" + project)


    summary_keys = ['eval/features', 'eval/obs_grad', 'train/batch_reward', 'eval/episode_reward', 'train/episode_reward']
    config_keys = ['decoder_lr', 'encoder_lr', 'distraction_level', 'seed']


    data = []
    for run in tqdm(runs):
        if len(run.summary._json_dict) > 0 and run.summary._json_dict['train/episode'] > 10:
            # The summary._json_dict stores the latest recorded value for metrics
            summary = run.summary._json_dict
            summary = {k:v for (k,v) in summary.items() if k in summary_keys}

            # Get configs
            config = {k:v for (k,v) in run.config.items() if k in config_keys}

            # Download eval/features as artifact
            artifact_download_path = api.artifact(f"{entity}/{project}/run-{run._attrs['name']}-evalfeatures:latest", type='run_table')
            artifact_download_path = artifact_download_path.download()
            features = json.load(open(f"{artifact_download_path}/eval/features.table.json", 'r'))
            features = np.array(features['data'])
            config['eval/features'] = features
            # Delete downloaded folder
            os.system(f"rm -rf {artifact_download_path}")

            # Download gradients as artifacts
            artifact_download_path = api.artifact(f"{entity}/{project}/run-{run._attrs['name']}-evalobs_grads:latest", type='run_table')
            artifact_download_path = artifact_download_path.download()
            obs_grad = json.load(open(f"{artifact_download_path}/eval/obs_grads.table.json", 'r'))
            obs_grad = np.array(obs_grad['data'])
            config['eval/obs_grad'] = obs_grad
            # Delete downloaded folder
            os.system(f"rm -rf {artifact_download_path}")
            data.append({
                'name': run.name,
                'config': config,
                'summary': summary,
            })


    # Get best performing models for all distraction levels
    best_scores = {k:-np.inf for k in distract_levels}
    features = {k:None for k in distract_levels}
    obs_grads = features.copy()
    for d in data:
        for distract_level in distract_levels:
            current_distract_level = d['config'].get('distraction_level')
            if current_distract_level == distract_level:
                score = d['summary'].get('eval/episode_reward')
                if score is not None:
                    if score > best_scores[distract_level]:
                        best_scores[distract_level] = score
                        features[distract_level] = d['config']['eval/features']
                        obs_grads[distract_level] = d['config']['eval/obs_grad']
                        
            
    # Plot variation in learned features
    fig, axes = plt.subplots(1,2)
    diffs = []
    for level,feature in features.items():
        features_level = features.get(level)
        if features_level is not None:
            features_err = np.std(features_level - features[0.0], axis=1)
            axes[0].plot(range(len(features_err)), [0]*len(features_err))
            axes[0].fill_between(range(len(features_err)), -features_err, features_err, label=f"{level}", alpha=0.2)
            
            axes[1].plot(np.linalg.norm(features_level - features[0.0], axis=1), label=f"{level}")
    
    axes[0].legend()
    # Move legend for axes[0] outside the plot
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_ylim(-3.0, 3.0)
    axes[0].set_ylabel('Elementwise feature differences w.r.t 0 distraction')
    axes[0].set_xlabel('Trajectory Steps')
    # Move legend for axes[1] outside the plot
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_ylabel('Norm of feature differences w.r.t 0 distraction')
    axes[1].set_xlabel('Trajectory Steps')
    axes[1].set_ylim(0.0, 25.0)
    ## change horizontal spacing on axes[1]
    plt.subplots_adjust(wspace=1.0)
    fig.set_tight_layout(True)
    fig.savefig(f'features_{encoder_mode}.png')
    plt.close()

    import ot
    from scipy.spatial.distance import pdist, squareform, cdist
    from einops import rearrange

    def get_wass_dist(grads1, grads2):
        a = abs(grads1)
        a = a/a.sum()
        a = a.reshape(-1)
        b = abs(grads2)
        b = b/b.sum()
        b = b.reshape(-1)
        # distance matrix
        elems = np.array([(x,y) for x in range(84) for y in range(84)])
        M = cdist(elems, elems, metric='euclidean')
        wass_dist = ot.emd2(a, b, M)
        return wass_dist

    


    # Plot gradients
    fig, axes = plt.subplots(1, len(distract_levels)+2 ,figsize=(25,5))
    wass_dists = []
    grad_padding_props = []
    for idx, (level, grad) in enumerate(obs_grads.items()):
        if grad is not None:
            wass_dist = get_wass_dist(obs_grads[0.0], grad)
            wass_dists.append(wass_dist)
            axes[idx].imshow(abs(grad))
            # Create an empty square using Rectangle and set fill=False
            side_len = int(grad.shape[0] / img_shrink_factor)
            margin = int((grad.shape[0] - side_len)/2)
            square = Rectangle((margin, margin), side_len, side_len, fill=False)
            axes[idx].add_patch(square)
            axes[idx].set_title(f"{level}")

            # Get the proportion of gradients outside the square
            total_grad_in_square = abs(grad[margin:-margin, margin:-margin]).sum()
            total_grad_padding = abs(grad).sum() - total_grad_in_square
            grad_padding_prop = total_grad_padding / abs(grad.sum())
            grad_padding_props.append(grad_padding_prop)

    # Plot proportion of grad in the padding
    axes[-2].plot(distract_levels, grad_padding_props, '-o')
    axes[-2].set_title('Proportion of grad in padding')
    axes[-2].set_xlabel('Distraction level')
    axes[-2].set_ylabel('Proportion of grad in padding')
    axes[-2].set_ylim(0.0, 1.0)

    axes[-1].plot(distract_levels, wass_dists, '-o')
    axes[-1].set_title('Wass. distances from 0.0')
    axes[-1].set_xlabel('Distraction level')
    axes[-1].set_ylabel('Wass. distances')
    axes[-1].set_ylim(0.0, 20.0)
    
    plt.subplots_adjust(wspace=1.0)

    fig.savefig(f'grads_{encoder_mode}.png')
    plt.close()
