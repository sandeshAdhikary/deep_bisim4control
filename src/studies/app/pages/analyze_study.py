import streamlit as st
import yaml
from streamlit_ace import st_ace
from streamlit_ace import THEMES as ace_themes
from streamlit_monaco import st_monaco
import os
from trainer.study import Study
from omegaconf import OmegaConf, DictConfig
import src.studies.app.app_utils as app_utils
import pandas as pd
import json
import altair as alt
from streamlit_js_eval import streamlit_js_eval

def block_content(block):
    study.update_blocks()
    
    with st.form(f"update_{block}_config", border=False, clear_on_submit=True):
        save_config = st.form_submit_button(f'Save {block} config')

    cols = st.columns(2)
    with cols[0]:
        # Make edits to config
        st.markdown("##### Edit config below")
        config = yaml.dump(getattr(study, block.lower()))
        new_config = st_ace(value=config, language='yaml', theme=ACE_THEME)
        
    if new_config is not None:
        with cols[1]:
            new_config = yaml.load(new_config, Loader=yaml.SafeLoader)
            new_config_errs = study.block_errors(block.lower(), new_config)
            
            if len(new_config_errs) == 0:
                st.markdown("##### Valid ✅")
            else:
                st.markdown(f"##### Invalid ❌")
                st.error('\n'.join(new_config_errs))
                save_config = False
                
            st.write(new_config)

            if save_config:
                with open(os.path.join(study.folder, f'{block.lower()}.yaml'), 'w') as f:
                    yaml.dump(new_config, f)
            study.update_blocks()



# st.set_page_config(layout="wide")
ACE_THEME = ace_themes[13]

with open('./assets/css/style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# Set up study from the url
url_params = st.experimental_get_query_params()
study_name = url_params['study_name'][0]
# Load studies yaml
with open('./assets/studies/studies.yaml') as f:
    studies = yaml.load(f, Loader=yaml.SafeLoader)['studies']
study_config = [s['path'] for s in studies if s['name'] == study_name][0]
study = Study(DictConfig(OmegaConf.load(study_config)))
runs = study.show_runs()

# Title
st.title(f"{app_utils.pretty_title(study.name)}")
st.markdown(study.config['study']['desc'])

# # Study Information
with st.expander("View the study's default config", expanded=False):
    app_utils.study_defaults(study)


# Select the runs to plot
st.header("Select Runs")
with st.expander("Select Runs", expanded=True):
    st.write("""Check runs in the `Select` column to include runs. 
                Use the `Tag` column to create new groups.
                Use the Group Sselector in the sidebar to define groups.
                """)
    runs = app_utils.run_selector(study)

if runs is  None:
    st.write("Pick some runs to view plots")
else:

    # Define columns to group-by
    with st.sidebar:
        runs, group_cols = app_utils.group_runs(runs)
        runs = app_utils.avg_over_group(runs)


    # View selected data
    st.header("View Selected Data")
    with st.expander("View Data", expanded=False):
        app_utils.view_selected_data(runs)

    ## 1. Learning Curves
    st.header("Learning Curves")
    with st.expander("Learning Curves", expanded=False):
        st.write("Here are the learning curves for the selected runs. Note that these curves are not grouped.")
        # Plot Training metrics
        all_train_data = []
        all_train_eval_data = []
        not_found_train_data = []
        not_found_train_eval_data = []
        for idr, row in runs.iterrows():
            project = row['project']
            sweep = row['sweep']
            run_id = row['run_id']
            run_group =  f"Project: {project} | Sweep: {sweep} | Run: {run_id}"
            if sweep is not None:
                train_file = f'{project}/sweep_{sweep}/{run_id}/eval/train_history.json'
                eval_file = f'{project}/sweep_{sweep}/{run_id}/eval/train_eval_history.json'
            else:
                train_file = f'{project}/{run_id}/eval/train_history.json'
                eval_file = f'{project}/{run_id}/eval/train_eval_history.json'


            try:
                train_data = study.storage.load(train_file, filetype='json')
                if isinstance(train_data, str):
                    train_data = json.loads(train_data)
                train_data = pd.DataFrame(train_data)
                train_data['group'] = run_group
                all_train_data.append(train_data)
            except FileNotFoundError as e:
                not_found_train_data.append(run_group)
            try:
                train_eval_data = study.storage.load(eval_file, filetype='json')
                if isinstance(train_eval_data, str):
                    train_eval_data = json.loads(train_eval_data)
                train_eval_data = pd.DataFrame(train_eval_data)
                train_eval_data['group'] = run_group
                all_train_eval_data.append(train_eval_data)
            except FileNotFoundError as e:
                not_found_train_eval_data.append(run_group)

        if len(all_train_data) > 0:
            st.subheader("Training Learning Curves")
            all_train_data = pd.concat(all_train_data)
            train_smoothing_value = -st.slider("Smoothing", min_value=0, max_value=50, step=1, value=1, key='train_learning_curve_slider')
            app_utils.plot_learning_curve(all_train_data, 
                                        x_value = 'trainer_step',
                                        y_value='train/episode_reward',
                                        smoothing_value=train_smoothing_value, 
                                        title='')

        if len(not_found_train_data) > 0:
            st.warning(f"""Could not find training learning curves for the following runs:""")
            st.write(not_found_train_data)


        if len(all_train_eval_data) > 0:
            st.subheader("Evaluation Learning Curves")
            all_train_eval_data = pd.concat(all_train_eval_data)
            eval_smoothing_value = -st.slider("Smoothing", min_value=0, max_value=50, step=1, value=1, key='eval_learning_curve_slider')
            app_utils.plot_learning_curve(all_train_eval_data,
                                        x_value='eval_step',
                                        y_value='eval/episode_reward_avg', 
                                        y_errs='eval/episode_reward_std',
                                        smoothing_value=eval_smoothing_value, 
                                        title='')
        if len(not_found_train_eval_data) > 0:
            st.warning(f"""Could not find evaluation history for the following runs:""")
            st.write(not_found_train_eval_data)


    ## 3. Metrics
    screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key = 'SCR_WIDTH')
    container_width = screen_width

    # # Evaluation metrics
    st.header("Evaluation Metrics")
    for metric_name, metric in study.metrics.items():
            with st.expander(app_utils.pretty_title(metric_name), expanded=False) as container:
                try:
                    # Get metric data
                    metric_data = study.metric_table(metric_name, limit=None)
                    metric_data = metric_data.merge(runs, on=['run_id', 'sweep', 'project'],how='right')
                    metric_data = metric_data.dropna(subset=['eval_name']) 


                    # Slider to select chart size
                    chart_size = st.slider("Chart Size", min_value=0.1, max_value=1.0, value=0.3,
                                            key=f"chart_size_{metric_name}", format="")
                    chart_size = container_width*chart_size

                    # Average out data over the groups                        
                    # metric_data = app_utils.avg_over_group(metric_data)
                    # Plots
                    if metric_data.shape[0] > 0:
                        st.subheader(app_utils.pretty_title(metric_name))
                        if metric_name == 'observation_videos':
                            app_utils.plot_videos(metric_data, storage=study.storage, key_prefix=metric_name)
                        elif metric_name in ['kmeans', 'tsne']:
                            app_utils.plot_chart(metric_data, storage=study.storage, key_prefix=metric_name)
                        elif metric_name in ['rewards_dataframe']:
                            # Don't plot here since these are inter-run metrics
                            pass
                        else:
                            facet = {'name': 'eval_name', 'columns': int(container_width//chart_size)}
                            app_utils.plot_scalars(metric, metric_data, group_cols, chart_size, facet)
                except Exception as e:
                    st.error(f"Could not plot {metric_name}")
                    st.error(e)
    
    # ## Rliable Metrics
    st.header("RLiable Metrics")
    inter_run_metrics = {}
    for metric_name, metric in study.metrics.items():
        if metric_name == 'rewards_dataframe':
            metric_data = study.metric_table(metric_name, limit=None)
            metric_data = metric_data.merge(runs, on=['run_id', 'sweep', 'project'],how='right')
            metric_data = metric_data.dropna(subset=['eval_name'])
            app_utils.plot_rliable_metrics(metric_data, study.storage)