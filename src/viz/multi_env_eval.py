import wandb
import pandas as pd
from tqdm import tqdm
import numpy as np
import yaml
import os
import glob
import json
import math
import altair as alt

def chart_grid(charts):
    # Calculate the number of rows and columns based on the number of charts
    num_charts = len(charts)
    num_cols = math.ceil(math.sqrt(num_charts))
    num_rows = math.ceil(num_charts / num_cols)

    # Create an empty grid to hold the charts
    grid = [[] for _ in range(num_rows)]

    # Fill the grid with charts
    for i, chart in enumerate(charts):
        row = i // num_cols
        grid[row].append(chart)

    # Create the grid layout
    grid_layout = alt.vconcat(*[alt.hconcat(*row) for row in grid])

    return grid_layout



def plot_lines(data, chart_defs):

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X(f"{chart_defs['x']['col_name']}:{chart_defs['x']['type']}", title=chart_defs['x']['render_name']),
        y=alt.Y(f"{chart_defs['y']['col_name']}:{chart_defs['y']['type']}", title=chart_defs['y']['render_name']),
        color=alt.Color(f"{chart_defs['color']['col_name']}:N", title=chart_defs['color']['render_name']),
    ).properties(
        title=chart_defs.get('title')
    )
    return chart

if __name__ == "__main__":

    api = wandb.Api()
    entity = api.default_entity

    project_names = ['cartpole_spectral_ortho', 'cartpole_dbc_seeds']
    # project_names = ['walker-exp-spectral', 'walker-exp-dbc']
    for img_source in ['mnist', None]:
        metric = 'eval/episode_reward'

        # Get a list of projects
        projects = [x for x in api.projects(entity) if x.name in project_names]

        # Get the best runs
        def run_filter_fn(run):
            keep_run = run.config['img_source'] == img_source
            # if run.config.get('logger_tags'):
            #     keep_run = keep_run and ('best' in run.config['logger_tags'])
            # else:
            #     keep_run = False
            # keep_run = keep_run and ('current' in run.tags)
            return keep_run
        
        # Get all runs
        table_name = 'eval_rewards_lineplot_table'
        full_df = pd.DataFrame()
        for project in tqdm(projects):

            path = f'{entity}/{project.name}'
            runs = [r for r in api.runs(path) if run_filter_fn(r)]
            for idx, run in enumerate(runs):
                print(f'Project: {project.name}, Run: {run.name}')

                table_id =  f"{entity}/{project.name}/run-{run.id}-{table_name}:latest"
                artifact = api.artifact(table_id, type='run_table')
                artifact_download_path = artifact.download()
                file = glob.glob(artifact_download_path + '/*.json')[0]
                table = json.load(open(file, 'r'))
                df = pd.DataFrame(table['data'], columns=table['columns'])
                df['project'] = project.name
                df['run_idx'] = idx
                full_df = pd.concat([full_df, df])
    

        # Get mean and std over multiple runs for each env
        full_df = full_df[['step', 'env_0', 'env_1', 'env_2', 'env_3', 'project', 'run_idx']]
        df_mean = full_df.groupby(['project', 'step'])[['env_0', 'env_1', 'env_2','env_3']].mean()
        df_mean.columns = [f'{col}_mean' for col in df_mean.columns]
        df_mean = df_mean.reset_index()
        df_std = full_df.groupby(['project', 'step'])[['env_0', 'env_1', 'env_2','env_3']].std()
        df_std.columns = [f'{col}_std' for col in df_std.columns]
        df_std = df_std.reset_index()

        full_df = pd.merge(df_std, df_mean, on=['project', 'step'])
        charts = []
        # Create the Altair chart
        envs = {'env_0':'No Distraction', 
                'env_1':'Color', 
                'env_2':'Moving MNIST', 
                'env_3':'Driving Videos'}

        for y_col, env_name in envs.items():
            line = alt.Chart(full_df).mark_line().encode(
                x=alt.X('step:O', title='Number of Env Steps'),
                y=alt.Y(f'{y_col}_mean:Q', title='Episode Reward (Eval)', scale=alt.Scale(domain=[-20,250])),
                color=alt.Color('project:N', title='Project')
            ).properties(
                title=f'Distraction: {env_name}'
            )
            band = alt.Chart(full_df).transform_calculate(
                y_low=f'datum.{y_col}_mean - datum.{y_col}_std',
            ).transform_calculate(
                y_high=f'datum.{y_col}_mean + datum.{y_col}_std',
            ).mark_errorband().encode(
                alt.Y('y_high:Q', scale=alt.Scale(domain=[0,250]), title='Episode Reward (Eval)'),
                alt.Y2('y_low:Q'),
                alt.X('step:O'),
                alt.Color('project:N', title='Project')
            )            

            chart = band+line
            charts.append(chart)
        chart = charts[0] & charts[1] & charts[2] & charts[3]

        chart.save(f'test_{img_source}.html')