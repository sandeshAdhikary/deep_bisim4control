import streamlit as st
from streamlit.connections import SQLConnection
import altair as alt
import altair as alt
import pandas as pd
import numpy as np
from trainer.storage import SSHFileSystemStorage
import os
import torch
from einops import rearrange
import time
import imageio
from io import BytesIO

def flatten_multiindex(df):
    df.columns = ['_'.join(col).strip() for col in df.columns.values if len(str(col).strip()) > 1]
    df.columns = [col.rstrip('_') for col in df.columns]
    return df

def pretty_title(x):
    return x.replace('_', ' ').title()

def exclude_key(x):
    exclude = False
    exclude = exclude or x.startswith('storage')
    exclude = exclude or ('freq' in x)
    exclude = exclude or (x == 'env')
    return exclude

def get_runs(conn, project):
    df = conn.query(f"select * from runs where project_id='{project}'")
    # # Load configs for each run
    all_dicts = []
    all_keys = []
    for i, row in df.iterrows():
        run_id = row['run_id']
        sweep_id = row['sweep_id']

        # Connect to the ssh storage to retrieve files
        ssh_storage_config = {
            'type': 'ssh',
            'project': project,
            'sweep': sweep_id,
            'run': run_id,
            'root_dir': os.environ['SSH_DIR'],
            'sub_dir': 'train',
            'host': os.environ['SSH_HOST'],
            'username': os.environ['SSH_USERNAME'],
            'password': os.environ['SSH_PASSWORD']
        }
        storage = SSHFileSystemStorage(ssh_storage_config)

        # Get configs with flattened keys
        try:
            model_config = storage.load(filename='model_config.yaml', filetype='env_yaml')
            trainer_config = storage.load(filename='trainer_config.yaml', filetype='env_yaml')
            model_dict, trainer_dict = {}, {}
            for k in model_config.keys():
                model_dict[k] = str(model_config[k])
            for k in trainer_config.keys():
                trainer_dict[k] = str(trainer_config[k])
            model_dict = set(model_dict.items())
            trainer_dict = set(trainer_dict.items())
            info_dict = set([('run_id', run_id), ('sweep', sweep_id)])
            run_dict = set.union(model_dict, trainer_dict, info_dict)
            all_dicts.append(run_dict)
            all_keys.append(set([x[0] for x in run_dict]))
        except FileNotFoundError:
                pass
    

                
    all_keys = set.union(*all_keys)
    matching_keys =  set([x[0] for x in set.intersection(*all_dicts)])
    diff_keys = all_keys - matching_keys

    diff_keys = [x for x in diff_keys if not exclude_key(x)]
    for idx in range(len(all_dicts)):
         all_dicts[idx] = {k:v for k,v in all_dicts[idx] if k in diff_keys}
    df = pd.DataFrame(all_dicts)    
    df['tag'] = [None]*len(df)

    return df

def get_metrics(conn, runs, metric):
    run_names = str(tuple(runs))
    df = conn.query(f"SELECT * FROM {metric} where run_id in {run_names}")
    return df

def plot(avg_values, error_mode, metric):
    for eval_name in avg_values['eval_name'].unique():
        data = avg_values[avg_values['eval_name']==eval_name]            
        
        if error_mode == 'Group':
            chart_error = alt.Chart(data).mark_errorband(opacity=0.2).encode(
                alt.Y('value_mean_low', title=''),
                alt.Y2('value_mean_high', title=''),
                alt.X('step:Q'),
                color=alt.Color('group:N', ),
            )
            subtitle = ["Error bands correspond to standard error in the mean across runs within a group"]
        elif error_mode == 'Evaluation Seeds':
            chart_error = alt.Chart(data).mark_errorband(opacity=0.2).encode(
                alt.Y('value_eval_mean_low', title=''),
                alt.Y2('value_eval_mean_high', title=''),
                alt.X('step:Q'),
                color=alt.Color('group:N', ),
            )
            subtitle = ["Error bands correspond to average standard deviation across evaluation seeds"]
            
        chart_title = alt.TitleParams(
            f"{pretty_title(metric)} for {pretty_title(eval_name)}",
            subtitle=subtitle,
            subtitleColor='white'
        )
        chart = alt.Chart(data, title=chart_title).mark_line().encode(
            x=alt.X('step:Q'),
            y=alt.Y('value_mean:Q', title=pretty_title(metric)),
            color='group:N',
        )

        chart = chart + chart_error
        st.altair_chart(chart, use_container_width=True)


def plot_agg(input_data, error_mode, metric):
    if error_mode == 'Evaluation Seeds':
        all_data = input_data.groupby(['group', 'run_id', 'sweep_id', 'eval_name'], as_index=False).agg({'value': ['sum'], 'value_std': [('sum', lambda x: np.sqrt(np.sum(x**2)))]})
        all_data = flatten_multiindex(all_data)
        all_data = all_data.groupby(['group', 'eval_name'], as_index=False).agg({'value_sum': ['mean'], 'value_std_sum': ['mean']})
        all_data = flatten_multiindex(all_data)
        for eval_name in all_data['eval_name'].unique():
            data = all_data[all_data['eval_name'] == eval_name].copy()
            data['value_high'] = pd.Series(data['value_sum_mean'] + data['value_std_sum_mean'])
            data['value_low'] = pd.Series(data['value_sum_mean'] - data['value_std_sum_mean'])
            chart_title = alt.TitleParams(
               f"{pretty_title(metric)} for {pretty_title(eval_name)}",
                )
            bar_chart = alt.Chart(data, title=chart_title).mark_bar().encode(
                x=alt.X('group:N'),
                y=alt.Y('value_sum_mean:Q', title=pretty_title(metric)),
                color='group:N',
            )
            line_high = alt.Chart(data).mark_errorbar(color='white').encode(
                alt.X('group:N'),
                alt.Y('value_high:Q', title=""),
                alt.Y2('value_low:Q', title=""),
            )
            chart = bar_chart + line_high
            st.altair_chart(chart, use_container_width=True)
    elif error_mode == 'Group':
        all_data = input_data.groupby(['group', 'run_id', 'sweep_id', 'eval_name'], as_index=False).agg({'value': ['sum']})
        all_data = flatten_multiindex(all_data)
        all_data.drop(['run_id', 'sweep_id'], inplace=True, axis='columns')
        all_data = all_data.groupby(['group', 'eval_name'], as_index=False).agg({'value_sum': ['mean', 'sem']})
        all_data = flatten_multiindex(all_data)
        for eval_name in all_data['eval_name'].unique():
            data = all_data[all_data['eval_name'] == eval_name].copy()
            data['value_high'] = pd.Series(data['value_sum_mean'] + data['value_sum_sem'])
            data['value_low'] = pd.Series(data['value_sum_mean'] - data['value_sum_sem'])
            chart_title = alt.TitleParams(
               f"{pretty_title(metric)} for {pretty_title(eval_name)}",
                )
            bar_chart = alt.Chart(data, title=chart_title).mark_bar().encode(
                x=alt.X('group:N'),
                y=alt.Y('value_sum_mean:Q', title=pretty_title(metric)),
                color='group:N',
            )
            line_high = alt.Chart(data).mark_errorbar(color='white').encode(
                alt.X('group:N'),
                alt.Y('value_high:Q', title=""),
                alt.Y2('value_low:Q', title=""),
            )
            chart = bar_chart + line_high
            st.altair_chart(chart, use_container_width=True)


def plot_metric(conn, runs, metric, group_cols):

    metrics = get_metrics(conn, runs['run_id'], metric)

    if len(group_cols) == 0:
        avg_values = metrics
        avg_values.drop('eval_id', inplace=True, axis='columns')
        avg_values['value_high'] = avg_values['value'] + avg_values['value_std']
        avg_values['value_low'] = avg_values['value'] - avg_values['value_std']
        # No grouping: Error bands over evaluation seeds
        avg_values['group'] = avg_values.apply(lambda x: x['run_id'], axis=1)
        for eval_name in avg_values['eval_name'].unique():
            data = avg_values[avg_values['eval_name']==eval_name]            
            
            chart_title = alt.TitleParams(
                f"{pretty_title(metric)} for {pretty_title(eval_name)}",
                subtitle=["Error bands correspond to standard deviation across evaluation seeds"],
                subtitleColor='white'
            )
            chart = alt.Chart(data, title=chart_title).mark_line().encode(
                x=alt.X('step:Q'),
                y=alt.Y('value:Q', title=pretty_title(metric)),
                color='group:N',
            )
            error_band = alt.Chart(data).mark_errorband(opacity=0.1).encode(
                alt.Y('value_high', title=''),
                alt.Y2('value_low', title=''),
                alt.X('step:Q'),
                color=alt.Color('group:N', )
            )
            chart = chart + error_band
            st.altair_chart(chart, use_container_width=True)

    else:
        run_groups = runs.copy()
        run_groups['group'] = runs.groupby(group_cols).ngroup()
        run_groups['group'] = run_groups.apply(lambda x: ' | '.join([f'{g} = {x[g]}' for g in group_cols]), axis=1)
        run_groups = run_groups[['run_id', 'group']]
        metrics['group'] = metrics.apply(lambda x: run_groups[run_groups['run_id']==x['run_id']]['group'].values[0], axis=1)
        aggregate_over_steps = st.radio('Aggregate over steps:', ['No', 'Yes'], horizontal=True)
        error_mode = st.radio('Show error bands for mean over:', ['Group', 'Evaluation Seeds'], horizontal=True)
        
        if aggregate_over_steps == 'Yes':
            metrics = metrics.drop('eval_id', axis='columns')
            plot_agg(metrics, error_mode, metric)
        else:
            metrics.drop(['run_id', 'sweep_id'], inplace=True, axis='columns')
            avg_values = metrics.groupby(['group', 'step', 'eval_name'], as_index=False).agg({
                'value': ['mean', 'sem'],
                'value_std': ['mean']
                })
            avg_values = flatten_multiindex(avg_values)
            avg_values['value_mean_high'] = avg_values['value_mean'] + avg_values['value_sem']
            avg_values['value_mean_low'] = avg_values['value_mean'] - avg_values['value_sem']
            avg_values['value_eval_mean_high'] = avg_values['value_mean'] + avg_values['value_std_mean']
            avg_values['value_eval_mean_low'] = avg_values['value_mean'] - avg_values['value_std_mean']
        

            plot(avg_values, error_mode, metric)


def get_eval_imgs(project, runs):
    eval_imgs = {}
    for idx,row in runs.iterrows():
        run = row['run_id']
        sweep = row['sweep']
        # Connect to the ssh storage to retrieve files
        ssh_storage_config = {
            'type': 'ssh',
            'project': project,
            'sweep': sweep,
            'run': run,
            'root_dir': os.environ['SSH_DIR'],
            'sub_dir': 'eval',
            'host': os.environ['SSH_HOST'],
            'username': os.environ['SSH_USERNAME'],
            'password': os.environ['SSH_PASSWORD']
        }
        storage = SSHFileSystemStorage(ssh_storage_config)
        filenames = storage.get_filenames()
        eval_files = [os.path.basename(filename).rstrip('\n') for filename in filenames if filename.startswith("eval") and filename.endswith(".pt\n")]
        eval_imgs[f"{sweep}--{run}"] = {}
        for eval_file in eval_files:
            obses = storage.load(filename=eval_file, filetype='torch')['episode_obs']
            eval_imgs[f"{sweep}--{run}"][eval_file.rstrip('.pt').lstrip('eval_output_')] = obses
    return eval_imgs

def play_video(img_container, imgs):
    for step in range(imgs.shape[0]):
        update_video(img_container, imgs[step,:])
        # img_container.image(imgs[step,:])
        

def update_video(video, img):
    video.image(img)
    time.sleep(0.1)

# def play_videos(img_containers, img_lists):
    # for img_container in img_containers:



def array_to_video(image_frames):
    # Create a BytesIO object to store the video
    video_bytesio = BytesIO()

    # Define some video parameters
    fps = 24  # Frames per second

    # Create a video writer
    with imageio.get_writer(video_bytesio, format='mp4', mode='I', fps=fps) as writer:
        for frame in image_frames:
            writer.append_data(frame)
    
    return video_bytesio

def plot_eval_imgs(project, runs, group_cols):
    eval_imgs = get_eval_imgs(project, runs) # keys are {sweep_run}

    if len(group_cols) == 0:
        group_cols = ['run_id']


    for sweep_run in eval_imgs.keys():
        sweep, run = sweep_run.split('--')
        env_names = eval_imgs[sweep_run].keys()
        all_envs = []
        for env_name in env_names:
            imgs = eval_imgs[sweep_run][env_name] # (T,3,H,W)
            imgs = rearrange(imgs, 't c h w -> t h w c')
            all_envs.append(imgs)
        all_envs = rearrange(np.stack(all_envs), 'n t h w c -> t h (n w) c') # stack horizontally
        vid_object = array_to_video(all_envs)
        run_title = []
        for col in group_cols:
            col_value = runs[runs['run_id']==run][col].values[0]
            run_title.append(f"{col}={str(col_value)}")
        run_title = pretty_title(' | '.join([x for x in run_title]))
        st.subheader(run_title)
        vid_object = st.video(vid_object, format='video/webm')


if __name__ == '__main__':
    
    from envyaml import EnvYAML
    
    config_path = st.text_input("Enter config path", value="src/experiments/cartpole/experiment_config.yaml")
    if len(config_path) > 0:
        config = dict(EnvYAML(config_path))
        project = config['experiment']['project']

        # Connect to the experiment database
        mysql_host = os.environ['SSH_HOST']
        mysql_port = 3306
        mysql_user = os.environ['SSH_USERNAME']
        mysql_password = os.environ['MYSQL_PASSWORD']
        mysql_db = 'bisim'
        conn = st.connection(
            name='bisim',
            type='sql',
            url=f"mysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
        )

        runs = get_runs(conn, project=config['experiment']['project'])
        runs = st.data_editor(runs)

        # Select columns to group by
        default_group_cols = []
        group_cols = st.multiselect(label='Group by', options=runs.columns, default=default_group_cols)

        metrics_tab, imgs_tab = st.tabs(["Metrics", "Videos"])

        with imgs_tab:
            # pass
            plot_eval_imgs(project, runs, group_cols)
        with metrics_tab:
            metric = 'episode_rewards'
            plot_metric(conn, runs, metric, group_cols)
