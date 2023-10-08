import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
import altair as alt
import os
import math
import json
import glob
import yaml
import wandb.apis.reports as wr


def get_metric(run, metric, avg_window=3):

    try:
        # Get the value of the metric by taking an average of the last avg_window steps
        value = run.history(keys=[metric]).tail(avg_window)[metric].mean()
    except KeyError:
        # The run does not have the metric
        value = -np.inf

    return {'name': run.name, 'value': value, 'project': run.project, 'id': run.id, 'config': run.config}

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

def get_best_runs(projects, metric, run_filter_fn=None):
    best_runs = {}
    for project in tqdm(projects):

        path = f'{entity}/{project.name}'
        runs = api.runs(path)
        if run_filter_fn:
            runs = [r for r in runs if run_filter_fn(r)]
        metrics = [get_metric(r, metric) for r in runs]
        best_run = max(metrics, key=lambda x: x['value'] or -np.inf)
        best_runs[project.name] = best_run
    return best_runs

if __name__ == "__main__":

    api = wandb.Api()
    entity = api.default_entity

    # project_names = ['cartpole-exp-spectral', 'cartpole-exp-dbc']
    project_names = ['walker-exp-spectral', 'walker-exp-dbc']
    for img_source in ['mnist', None]:
        metric = 'eval/episode_reward'

        # Get a list of projects
        projects = [x for x in api.projects(entity) if x.name in project_names]

        # Get the best runs
        run_filter_fn = lambda run: run.config['img_source'] == img_source
        best_runs = get_best_runs(projects, metric, run_filter_fn)


        # Save the best configs
        for run_name, run_details in best_runs.items():
            with open(os.path.join(os.path.dirname(__file__), f'best_configs_{run_details["project"]}_{img_source}.yaml'), 'w') as f:
                yaml.dump(run_details['config'], f)

        # Get histories
        cols_to_plot = ['train/episode_reward', 'eval/episode_reward']
        histories = [{project:api.run(f'{entity}/{project}/{run_details["id"]}').history(keys=cols_to_plot)} for (project, run_details) in best_runs.items()]

        full_df = pd.DataFrame()    
        for hist in histories:
            project_name, df = list(hist.items())[0]
            df['project'] = project_name
            df['img_source'] = best_runs[project_name]['config']['img_source'] or 'None'
            full_df = pd.concat([full_df, df])


        # Plot comparisons between best models across the projects
        chart_defs = {
            'x': {'col_name': '_step', 'type': 'Q', 'render_name': 'Steps'},
            'y': {'col_name': 'train/episode_reward', 'type': 'Q', 'render_name': 'Episode Reward (Train)'},
            'color': {'col_name': 'project', 'render_name': 'Encoder Modes'},
            'title': 'Best Models'
        }

        for img_mode in ['None', 'mnist']:
            chart_df = full_df[full_df['img_source'] == img_mode]
            chart = plot_lines(full_df, chart_defs)

        chart.save(os.path.join(os.path.dirname(__file__), f'best_models_chart_{img_source}.html'))
        with open(os.path.join(os.path.dirname(__file__), f'best_models_chart_{img_source}.json'), 'w') as f:
            f.write(chart.to_json())
        

        # Plot evaluations on different distractors
        table_name = 'eval_rewards_lineplot_table'
        full_df = pd.DataFrame()
        for project_name, run in best_runs.items():
            table_id =  f"{entity}/{project_name}/run-{run['id']}-{table_name}:latest"
            artifact = api.artifact(table_id, type='run_table')
            artifact_download_path = artifact.download()
            file = glob.glob(artifact_download_path + '/*.json')[0]
            table = json.load(open(file, 'r'))
            df = pd.DataFrame(table['data'], columns=table['columns'])
            df['project'] = project_name
            full_df = pd.concat([full_df, df])


        charts = []
        envs = full_df.columns[full_df.columns.str.startswith('env_')]
        env_names = ['No Distraction', 'Colors', 'MNIST', 'Driving Video']
        for idx, env in enumerate(envs):
            chart_defs = {
                'x': {'col_name': 'step', 'type': 'Q', 'render_name': 'Steps'},
                'y': {'col_name': f'{env}', 'type': 'Q', 'render_name': 'Episode Reward (Eval)'},
                'color': {'col_name': 'project', 'render_name': 'Encoder Modes'},
                'title': f'{env_names[idx]}'
            }
            charts.append(plot_lines(full_df, chart_defs))

        combined_chart = chart_grid(charts)
        combined_chart.save(os.path.join(os.path.dirname(__file__), f'eval_distractions_{img_source}.html'))
        with open(os.path.join(os.path.dirname(__file__), f'eval_distractions_{img_source}.json'), 'w') as f:
            f.write(combined_chart.to_json())

