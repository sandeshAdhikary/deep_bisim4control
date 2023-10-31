import streamlit as st
from streamlit.connections import SQLConnection
import altair as alt
import altair as alt
import pandas as pd
import numpy as np
from trainer.storage import SSHFileSystemStorage
import os
from trainer.utils import flatten_dict


def exclude_key(x):
    exclude = False
    exclude = exclude or x.startswith('storage')
    exclude = exclude or ('freq' in x)
    return exclude

def get_runs(conn):
    df = conn.query("select * from runs")
    # # Load configs for each run
    all_dicts = []
    all_keys = []
    for i, row in df.iterrows():
        run_id = row['run_id']
        sweep_id = row['sweep_id']

        # Connect to the ssh storage to retrieve files
        ssh_storage_config = {
            'type': 'ssh',
            'project': 'test',
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
            info_dict = set([('run', run_id), ('sweep', sweep_id)])
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


if __name__ == '__main__':

    st.write("Welcome")

    # Connect to the experiment database
    mysql_host = os.environ['SSH_HOST']
    mysql_port = 3306
    mysql_user = os.environ['SSH_USERNAME']
    mysql_password = os.environ['SSH_PASSWORD']
    mysql_db = 'bisim'
    conn = st.connection(
        name='bisim',
        type='sql',
        url=f"mysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
    )

    runs = get_runs(conn)
    runs_select = st.data_editor(runs)
    tag_map = {run_id:tag for run_id, tag in zip(runs_select['run'], runs_select['tag'])}
    
    active_tags = set(tag_map.values())
    runs = runs[runs['tag'].isin(active_tags)]
    selected_runs = runs['run'].unique()

    metric = 'episode_rewards'
    metrics = get_metrics(conn, selected_runs, metric)
    metrics['tag'] = metrics.apply(lambda x: tag_map[x['run_id']], axis=1)
    # st.write(metrics)


    value_mean = metrics.groupby(['run_id', 'eval_id', 'step', 'tag'], as_index=False)['value'].agg({'value_mean':'mean'})
    value_std = metrics.groupby(['run_id', 'eval_id', 'step', 'tag'], as_index=False)['value'].agg({'value_std':'std'})
    eval_names = metrics.groupby(['run_id', 'eval_id', 'tag'], as_index=False)['eval_name'].agg({'eval_name':'first'})
    values = pd.merge(value_mean, value_std, on=['run_id', 'eval_id', 'step', 'tag'])
    values['eval_name'] = values.apply(lambda x: eval_names[(eval_names['run_id']==x['run_id']) & (eval_names['eval_id']==x['eval_id'])]['eval_name'].values[0], axis=1)
    values['value_high'] = values['value_mean'] + values['value_std']
    values['value_low'] = values['value_mean'] - values['value_std']

    if st.button('Plot'):
        for eval_name in values['eval_name'].unique():
            data = values[values['eval_name']==eval_name]
            
            chart = alt.Chart(data).mark_line().encode(
                x=alt.X('step:Q'),
                y=alt.Y('value_mean:Q', title=metric),
                color='tag:N',
            )
            error_band = alt.Chart(values).mark_errorband().encode(
                alt.Y('value_high'),
                alt.Y2('value_low'),
                alt.X('step:Q'),
            )
            chart = chart + error_band
            chart = chart.properties(title=f'{eval_name}')
            st.altair_chart(chart, use_container_width=True)
    else:
        pass



