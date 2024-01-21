import streamlit as st
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode, DataReturnMode
import altair as alt
import seaborn as sns
from rliable import library as rly
from rliable import metrics as rliable_metrics
from rliable import plot_utils
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from copy import copy, deepcopy

COLORS = {
   'tableau10': ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", 
              "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AB"],
    'tableau20': ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", 
              "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
              "#AEC7E8", "#FFBC79", "#98DF8A", "#FF9896", "#C5B0D5", 
              "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"]
}

def merge_with_runs(data, all_runs, group_cols=None):
    data_cols = [c for c in data.columns if c not in ['project', 'sweep', 'run_id']]
    df = pd.DataFrame.merge(data, 
                            all_runs[np.unique(['project', 'sweep', 'run_id', 'group'] + group_cols)], 
                            on=['project', 'sweep', 'run_id']
                            )
    df = df[np.unique(np.concatenate([data_cols, group_cols]))]
    return df

def pretty_title(x):
    x = x.replace('_', ' ').title()
    return x.lstrip(' ').rstrip(' ')

def study_defaults(study):
    for key, value in study.config.items():
        # if key != 'desc':
        st.write(pretty_title(key))
        # st.write(type(value))
        if isinstance(value, (OmegaConf, DictConfig)):
            st.json(OmegaConf.to_container(value, resolve=False), expanded=False)
        elif isinstance(value, dict):
            st.json(value, expanded=False)

    # # Combine all sets, only keep keys that are different
    # all_keys = set.union(*all_keys)
    # matching_keys =  set([x[0] for x in set.intersection(*all_dicts)])
    # diff_keys = all_keys - matching_keys
    # diff_keys = [x for x in diff_keys if not exclude_key(x)]

    # if 'project' not in diff_keys:
    #     diff_keys.append('project')
    # if 'sweep' not in diff_keys:
    #     diff_keys.append('sweep')

    # for idx in range(len(all_dicts)):
    #      all_dicts[idx] = {k:v for k,v in all_dicts[idx] if k in diff_keys}
    # df = pd.DataFrame(all_dicts)    
    # df['tag'] = [None]*len(df)


    # new_cols = [x for x in df.columns if x not in runs.columns]
    # df = df[['project', 'sweep', 'run_id', 'steps', *new_cols]]


    # builder = GridOptionsBuilder.from_dataframe(df)
    # builder.configure_column(field='sweep', width=100, header_name="Sweep")
    # builder.configure_column(field='project', width=200, header_name="Project")
    # builder.configure_column(field='run_id', width=100, header_name="Run")
    # builder.configure_column(field='steps', width=100, header_name="Steps")
    # builder.configure_column(field='tag', width=100, header_name="Tag", editable=True)
    # builder.configure_column(field='select', checkboxSelection=True, header_name="Select")

    # builder.configure_grid_options(groupDefaultExpanded=-1, rowSelection='multiple',)
    # gridOptions = builder.build()
    # # gridOptions['getRowStyle'] = jscode
    # data_selector = AgGrid(df, gridOptions=gridOptions,
    #             columns_auto_size_mode=1,theme="streamlit", allow_unsafe_jscode=True,
    #             data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    #             header_checkbox_selection_filtered_only=True
    # )
    #             # update_mode=GridUpdateMode.MODEL_CHANGED,)
    # # Get filtered data
    # final_data = pd.DataFrame(data_selector.data).fillna("None")
    # if len(data_selector.selected_rows) > 0:
    #     selected_data = pd.DataFrame(data_selector.selected_rows).drop('_selectedRowNodeInfo', axis='columns')
    #     selected_data = selected_data.fillna("None")
    #     final_data = final_data.merge(selected_data, how='inner', on=final_data.columns.tolist())
    #     final_data['sweep'] = final_data['sweep'].apply(lambda x: str(x).lower())
    #     return final_data
    # return None        else:
            # st.write(f"`{value}`")


def exclude_key(x):
    exclude = False
    exclude = exclude or x.startswith('storage')
    exclude = exclude or ('freq' in x)
    exclude = exclude or (x == 'env')
    return exclude


def run_selector(study):

    # Get list of runs
    runs = study.show_runs()
    all_dicts = []
    all_keys = []
    for idr, row in runs.iterrows():
        # Get run info from study
        project, sweep, run_id, steps = row['project'], row['sweep'], row['run_id'], row['steps']
        if sweep.lower() == 'none':
            sweep = None

        try:
            run_info = study.run_info(project=project, sweep=sweep, run_id=run_id)
            
            # Reformat model and trainer dicts as sets
            model_dict, trainer_dict = {}, {}
            for k in run_info['model'].keys():
                model_dict[k] = str(run_info['model'][k])
            for k in run_info['trainer'].keys():
                trainer_dict[k] = str(run_info['trainer'][k])
            model_dict = set(model_dict.items())
            trainer_dict = set(trainer_dict.items())
            info_dict = set([('run_id', run_id), ('sweep', sweep), ('steps', steps)])
            run_dict = set.union(model_dict, trainer_dict, info_dict)
            all_dicts.append(run_dict)
            all_keys.append(set([x[0] for x in run_dict]))
        except FileNotFoundError as e:
            # print(f"Run {run_id} not found")
            continue


    # Combine all sets, only keep keys that are different
    all_keys = set.union(*all_keys)
    matching_keys =  set([x[0] for x in set.intersection(*all_dicts)])
    diff_keys = all_keys - matching_keys
    diff_keys = [x for x in diff_keys if not exclude_key(x)]

    if 'project' not in diff_keys:
        diff_keys.append('project')
    if 'sweep' not in diff_keys:
        diff_keys.append('sweep')

    for idx in range(len(all_dicts)):
         all_dicts[idx] = {k:v for k,v in all_dicts[idx] if k in diff_keys}
    df = pd.DataFrame(all_dicts)    
    df['tag'] = [None]*len(df)

    new_cols = [x for x in df.columns if x not in runs.columns]
    df = df[['project', 'sweep', 'run_id', 'steps', *new_cols]]

    builder = GridOptionsBuilder.from_dataframe(df)
    builder.configure_column(field='sweep', width=100, header_name="Sweep")
    builder.configure_column(field='project', width=200, header_name="Project")
    builder.configure_column(field='run_id', width=100, header_name="Run")
    builder.configure_column(field='steps', width=100, header_name="Steps")
    builder.configure_column(field='tag', width=100, header_name="Tag", editable=True)
    builder.configure_column(field='select', checkboxSelection=True, header_name="Select")

    builder.configure_grid_options(groupDefaultExpanded=-1, rowSelection='multiple',)
    gridOptions = builder.build()
    
    data_selector = AgGrid(df, gridOptions=gridOptions,
                columns_auto_size_mode=1,theme="streamlit", allow_unsafe_jscode=True,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                header_checkbox_selection_filtered_only=True
    )
    # Get filtered data
    final_data = pd.DataFrame(data_selector.data).fillna("None")
    if len(data_selector.selected_rows) > 0:
        selected_data = pd.DataFrame(data_selector.selected_rows).drop('_selectedRowNodeInfo', axis='columns')
        selected_data = selected_data.fillna("None")
        final_data = final_data.merge(selected_data, how='inner', on=final_data.columns.tolist())
        # final_data['sweep'] = final_data['sweep'].apply(lambda x: str(x).lower())
        return final_data
    return None

    
def plot_learning_curve(data, x_value, y_value, 
                        y_errs=None,
                        smoothing_value=1.0, 
                        x_title='Steps', 
                        y_title='Episode Reward', 
                        title='Learning Curve',
                        group_cols=None,
                        y_norm_lims=None,
                        x_norm_lims=None,
                        y_plot_lims=None,
                        x_plot_lims=None,
                        colors=None,
                        key_suffix='learning_curve',
                        titlePadding=30):
    

    data['group'] = data.groupby(group_cols).ngroup()
    data['group'] = data.apply(lambda x: ' | '.join([f'{g} = {x[g]}' for g in group_cols]), axis=1)

    # Pick font sizes
    label_font_size = st.number_input(label='Label Font Size', 
                                      value=30, min_value=0, 
                                      max_value=50, key=f'label_font_size_{key_suffix}')
    title_font_size = st.number_input(label='Title Font Size',
                                    value=20, min_value=0, 
                                    max_value=50, key=f'title_font_size_{key_suffix}')
    
    # Set height and width
    chart_height = st.number_input(label='Chart Height',
                                    value=500, min_value=0, 
                                    max_value=1000, key=f'chart_height_{key_suffix}')
    chart_width = st.number_input(label='Chart Width',
                                    value=500, min_value=0, 
                                    max_value=1000, key=f'chart_width_{key_suffix}')


    group_names = list(data['group'].unique())
    if colors is None:
        colors = COLORS['tableau20'][:len(group_names)]

    # Legend with color picker
    for idg, group in enumerate(group_names):
        cols = st.columns([0.05, 0.95])
        colors[idg] = cols[0].color_picker(label='', value=colors[idg], 
                                           key=f'color_picker_{group}_{key_suffix}')
        cols[1].text_input("", value=group, key=f'group_name_{group}_{key_suffix}')
    
    
    if y_norm_lims is not None:
        data[y_value] = (data[y_value] - y_norm_lims[0])/(y_norm_lims[1] - y_norm_lims[0])
        if y_errs is not None:
            data[y_errs] = data[y_errs]/(y_norm_lims[1] - y_norm_lims[0])

    if x_norm_lims is not None:
        data[x_value] = (data[x_value] - x_norm_lims[0])/(x_norm_lims[1] - x_norm_lims[0])
    y_plot_lims = y_plot_lims or [data[y_value].min(), data[y_value].max()]
    # st.write(y_plot_lims)
    selection = alt.selection_point(fields=['group'], bind='legend')
    
    # The base chart
    chart = alt.Chart(data, title=title).mark_line()

    if x_plot_lims is None:
        chart = chart.encode(x=alt.X(f'{x_value}:Q', title=x_title))
    else:
        chart = chart.encode(x=alt.X(f'{x_value}:Q', title=x_title, 
                                      scale=alt.Scale(domain=x_plot_lims)))
    
    if y_plot_lims is None:
        chart = chart.encode(y=alt.Y(f'{y_value}:Q', title=y_title, 
                                      axis=alt.Axis(titlePadding=titlePadding)))
    else:
        chart = chart.encode(y=alt.Y(f'{y_value}:Q', title=y_title, 
                                     scale=alt.Scale(domain=y_plot_lims),
                                     axis=alt.Axis(titlePadding=titlePadding)))


    
    chart = chart.encode(
        color=alt.Color('group:N', scale = alt.Scale(domain=group_names, range=colors), legend=None),
        tooltip='group:N',
    ).add_params(
        selection
    )

    if y_errs is not None:
            data['y_max'] = data[y_value] + data[y_errs]
            data['y_min'] = data[y_value] - data[y_errs]
            error_chart = alt.Chart(data).mark_area(
                opacity=0.2
            ).encode(
                x=alt.X(f'{x_value}:Q', title=x_title),
                y=alt.Y('y_min:Q'),
                y2=alt.Y2('y_max:Q'),
                color=alt.Color('group:N'),
                tooltip='group:N',
            )
            chart = error_chart + chart

    chart = chart.configure_axis(
    labelFontSize=label_font_size,
    titleFontSize=title_font_size
    )

    chart = chart.properties(
        width=chart_width,
        height=chart_height
    )


    st.altair_chart(chart, use_container_width=False)


def group_runs(runs, default_group_cols=None):
    if default_group_cols is None:
         default_group_cols = ['project', 'sweep']
    group_cols = st.multiselect(label='Select Group Columns', 
                                options=runs.columns, default=default_group_cols, 
                                placeholder="Select Group Columns")
    if len(group_cols) == 0:
        # Use run_id as group
        runs['group'] = runs.apply(lambda x: f"run_{x['run_id']}", axis=1)
        group_cols = ['run_id']
    else:
        # Group by groups provided
        runs['group'] = runs.groupby(group_cols).ngroup()
        runs['group'] = runs.apply(lambda x: ' | '.join([f'{g} = {x[g]}' for g in group_cols]), axis=1)
    return runs, group_cols

def avg_over_group(data, group_by_cols=None):
    """"
    data should have a column named 'group'
    will return a dataframe grouped by the 'group' col
    all numerical columns will be averaged
    For non-numeric columns, we'll take the first value
    """
    if group_by_cols is None:
        group_by_cols  = ['group']
    # Define custom aggregation functions
    aggregation_funcs = {}
    # Get the list of numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    # Get the list of non-numeric columns
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    # Apply 'median' aggregation to numeric columns
    for col in numeric_columns:
        aggregation_funcs[col] = 'mean'

    # Apply 'first' aggregation to non-numeric columns
    for col in non_numeric_columns:
        aggregation_funcs[col] = 'first'
    result = data.groupby(group_by_cols).agg(aggregation_funcs).reset_index(drop=True)
    
    return result

def view_selected_data(runs):
        st.write("""
                Here's the data based on your selection and grouping criteria.
                The `group` column shows each run's assigned group.
                Numeric values have been averaged over the groups. For non-numeric 
                columns, we display the first value for the group.
                """)
        projects = ' '.join([f"`{x}`" for x in runs['project'].unique()])
        st.write(f"Projects: {projects}")
        sweeps = ' '.join([f"`{x}`" for x in runs['sweep'].unique()])
        st.write(f"Sweeps: {sweeps}")
        runs = runs[['group', *runs.columns.difference(['group'])]]
        st.write(runs)


def df_apply_colors(val, color_dict):
     color = color_dict[val]
     return f'background-color: {color}; color: {color}; font-size:0.01em'

def make_legend_table(data, group_cols, legend_color_map, legend_col):
    cols = [*group_cols, 'group']
    data = data.loc[:, cols].drop_duplicates()
    # legend_table = data.loc[:,cols].drop_duplicates()
    data.rename({f'{legend_col}': "legend"}, axis='columns', inplace=True)
    # move legend to first column
    col = data.pop("legend")
    data.insert(0, col.name, col)
    data.reset_index(drop=True, inplace=True)
    data.columns = [pretty_title(x) for x in data.columns]
    data = data.style.map(lambda x: df_apply_colors(x, legend_color_map), subset=['Legend'])
    return data

def plot_scalars(metric, data, group_cols, chart_size, facet=None):
    if facet is None:
        facet = {}
    chart, chart_info = metric.plot(data, facet=facet, 
                                    show_legend=False,
                                    chart_properties={'width': chart_size, 
                                                    'height': chart_size,}
                                                    )
    legend_table = make_legend_table(data, group_cols, 
                                     legend_color_map=chart_info['legend_colors'], 
                                     legend_col='group')
    st.dataframe(legend_table, hide_index=True)
    st.altair_chart(chart)

def plot_videos(data, storage, key_prefix='video'):
    project_select = st.selectbox("Select Project", data['project'].unique(), key=f"{key_prefix}_project_select")
    sweep_select = st.selectbox("Select Sweep", data.query(f"project=='{project_select}'")['sweep'].unique(), key=f"{key_prefix}_sweep_select")
    run_select = st.selectbox("Select Run", data.query(f"project=='{project_select}' and sweep=='{sweep_select}'")['run_id'].unique(), key=f"{key_prefix}_run_select")
    evals = data.query(f"project=='{project_select}' and sweep=='{sweep_select}' and run_id=='{run_select}'")
    num_vids = evals.shape[0]
    vid_cols = st.columns(num_vids)
    for idv in range(num_vids):
        data_row = evals.iloc[idv]
        filepath = data_row['filepath']
        filepath = filepath.split(storage.dir)[1].lstrip('/')
        video = storage.load(filepath, filetype='bytesio')
        vid_cols[idv].video(data=video)
        vid_cols[idv].write(f"Eval Name: `{pretty_title(data_row['eval_name'])}`")


def plot_chart(data, storage, key_prefix='chart'):
    project_select = st.selectbox("Select Project", data['project'].unique(), key=f"{key_prefix}_project_select")
    sweep_select = st.selectbox("Select Sweep", data.query(f"project=='{project_select}'")['sweep'].unique(), key=f"{key_prefix}_sweep_select")
    run_select = st.selectbox("Select Run", data.query(f"project=='{project_select}' and sweep=='{sweep_select}'")['run_id'].unique(), key=f"{key_prefix}_run_select")
    evals = data.query(f"project=='{project_select}' and sweep=='{sweep_select}' and run_id=='{run_select}'")
    num_charts = evals.shape[0]
    chart_cols = st.columns(num_charts)
    for idc in range(num_charts):
        data_row = evals.iloc[idc]
        filepath = data_row['filepath']
        filepath = filepath.split(storage.dir)[1].lstrip('/')
        chart = storage.load(filepath, filetype='json')
        st.vega_lite_chart(chart)

def plot_rliable_metrics(data, storage, key_prefix='rliable'):
    for idr in range(len(data)):
        data_row = data.iloc[idr]
        filepath = data_row['filepath']
        df = pd.DataFrame.from_records(storage.load(filepath, filetype='json'))
        df['name'] = data_row['group']
        df['project'] = data_row['project']
        df['sweep'] = data_row['sweep']
        if idr == 0:
            full_df = df
        else:
            full_df = pd.concat([full_df,df])
    
    sweeps = full_df['sweep'].unique()
    projects = full_df['project'].unique()
    n_trials = 30
    n_reps = 10_000
    min_score = 0.0
    max_score = 1000.0

    all_data = {}
    sweep_data = np.zeros((len(projects), n_trials))
    for sweep in sweeps:
        for idp, project in enumerate(projects):
            scores = full_df.query(f"sweep=='{sweep}' & project=='{project}'")['reward'].values
            scores = (scores - min_score)/(max_score - min_score)
            sweep_data[idp] = scores
        all_data[sweep] = np.array(sweep_data)
        

    # st.markdown('## Aggregate Scores')    
    # agg_func_names = ['Median', 'IQM', 'Mean', 'OptGap']
    # aggregate_func = lambda x: np.array([
    # rliable_metrics.aggregate_median(x),
    # rliable_metrics.aggregate_iqm(x),
    # rliable_metrics.aggregate_mean(x),
    # rliable_metrics.aggregate_optimality_gap(x)])
    
    # aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(all_data, aggregate_func, 
    #                                                                    reps=n_reps)

    # data = [[key, *aggregate_scores[key], *aggregate_score_cis[key][0], *aggregate_score_cis[key][1]] for key in aggregate_scores.keys()]
    # data = pd.DataFrame(data,
    #                       columns=['algorithm', 
    #                              *[f'{x}_agg' for x in agg_func_names],
    #                              *[f'{x}_lower' for x in agg_func_names],
    #                              *[f'{x}_upper' for x in agg_func_names]])

    # algorithm_names = list(aggregate_scores.keys())
    # colors = COLORS['tableau20'][:len(algorithm_names)]

    # # Set height and width
    # chart_height = st.number_input(label='Chart Height',
    #                                 value=500, min_value=0, 
    #                                 max_value=1000, key=f'chart_height_{key_prefix}')
    # chart_width = st.number_input(label='Chart Width',
    #                                 value=100, min_value=0, 
    #                                 max_value=1000, key=f'chart_width_{key_prefix}')


    # # Legend with color picker
    # for idg, group in enumerate(algorithm_names):
    #     cols = st.columns([0.05, 0.95])
    #     colors[idg] = cols[0].color_picker(label='', value=colors[idg], 
    #                                        key=f'color_picker_{group}_{key_prefix}')
    #     cols[1].text_input("", value=group, key=f'group_name_{group}_{key_prefix}')

    # charts = []
    # for ida, agg_func_name in enumerate(agg_func_names):
    #     # Bar chart
    #     chart = alt.Chart(data, height=chart_height, width=chart_width).mark_bar().encode(
    #         y = alt.Y(f'{agg_func_name}_lower:Q', title=None, 
    #                   scale=alt.Scale(domain=[0.0, 1.0]), 
    #                   axis=alt.Axis(labels=True, tickCount=10)),
    #         x=alt.Y('algorithm:N', title=agg_func_name, axis=alt.Axis(labels=False, ticks=False)),
    #         y2= f'{agg_func_name}_upper:Q',
    #         color=alt.Color('algorithm:N', legend=None, scale=alt.Scale(domain=algorithm_names, range=colors)),
    #     )
    #     # Line for aggregate score
    #     agg_chart = alt.Chart(data).mark_point(color='black', shape='stroke').encode(
    #         x=alt.Y('algorithm:N', title=agg_func_name, axis=alt.Axis(labels=False, ticks=False)),
    #         y=alt.Y(f'{agg_func_name}_agg:Q', title=None),
    #     )
    #     chart = chart + agg_chart

    #     charts.append(chart)
    # chart = alt.hconcat(*charts, spacing=50
    #                     ).resolve_scale(y='shared'
    #                     )

    # st.altair_chart(chart, use_container_width=False)

    # st.markdown('## Performance Profiles') 
    # # # Performance Profiles
    # perf_thresholds = np.linspace(0.0, 1.0, 50)
    # perf_profiles, perf_profiles_cis = rly.create_performance_profile(all_data, perf_thresholds)

    # algorithm_names = list(perf_profiles.keys())
    # df = pd.DataFrame()
    # for algo in algorithm_names:
    #     algo_df = pd.DataFrame(
    #         {   
    #             'algorithm': algo,
    #             'threshold': perf_thresholds,
    #             'fraction': perf_profiles[algo],
    #             'lower': perf_profiles_cis[algo][0],
    #             'upper': perf_profiles_cis[algo][1],
    #         }
    #     )
    #     df = pd.concat([df, algo_df])
    


    # # Set height and width
    # chart_height = st.number_input(label='Chart Height',
    #                                 value=500, min_value=0, 
    #                                 max_value=1000, key=f'chart_height_{key_prefix}_perf_profile')
    # chart_width = st.number_input(label='Chart Width',
    #                                 value=100, min_value=0, 
    #                                 max_value=1000, key=f'chart_width_{key_prefix}_perf_profile')

    # label_font_size = st.number_input(label='Label Font Size', 
    #                                   value=30, min_value=0, 
    #                                   max_value=50, key=f'label_font_size_{key_prefix}_perf_profile')
    # title_font_size = st.number_input(label='Title Font Size',
    #                                 value=20, min_value=0, 
    #                                 max_value=50, key=f'title_font_size_{key_prefix}_perf_profile')


    # # Legend with color picker
    # colors = COLORS['tableau20'][:len(algorithm_names)]
    # for idg, group in enumerate(algorithm_names):
    #     cols = st.columns([0.05, 0.95])
    #     colors[idg] = cols[0].color_picker(label='', value=colors[idg], 
    #                                        key=f'color_picker_{group}_{key_prefix}_perf_profile')
    #     cols[1].text_input("", value=group, key=f'group_name_{group}_{key_prefix}_perf_profile')

    # chart = alt.Chart(df, height=chart_height, width=chart_width).mark_line().encode(
    #     x=alt.X('threshold:Q', title='Normalized Score (\u03c4)'),
    #     y=alt.Y('fraction:Q', title='Fraction of runs with score > \u03c4'),
    #     color=alt.Color('algorithm:N', legend=None, scale=alt.Scale(domain=algorithm_names, range=colors)),
    # )

    # chart_err = alt.Chart(df).mark_area(opacity=0.2).encode(
    #     x=alt.X('threshold:Q'),
    #     y=alt.Y('lower:Q'),
    #     y2=alt.Y2('upper:Q'),
    #     color=alt.Color('algorithm:N', legend=None, scale=alt.Scale(domain=algorithm_names, range=colors)),
    # )
    # chart = (chart + chart_err).configure_axis(
    # labelFontSize=label_font_size,
    # titleFontSize=title_font_size
    # )

    # st.altair_chart(chart)

    # # Probability of improvement
    sweep_pairs = list(permutations(sweeps, 2))
    repeats = 10
    pi_data = {}
    for pair in sweep_pairs:
        pi_data[f'{pair[0]}, {pair[1]}'] = (all_data[pair[0]], all_data[pair[1]])
    average_probabilities, average_prob_cis = rly.get_interval_estimates(pi_data, rliable_metrics.probability_of_improvement, reps=repeats)


    for idk, key in enumerate(average_probabilities.keys()):
        algo_x, algo_y = key.split(', ')
        algo_df = pd.DataFrame({'algo_x': algo_x, 
                    'algo_y': algo_y, 
                    'algo_pair': f'{algo_x}, {algo_y}',
                    'prob': average_probabilities[key], 
                    'prob_lower': average_prob_cis[key][0], 
                    'prob_upper': average_prob_cis[key][1]}
                    )
        if idk > 0:
            df = pd.concat([df, algo_df])
        else:
            df = algo_df

    chart_height = st.number_input(label='Chart Height',
                                    value=500, min_value=0, 
                                    max_value=1000, key=f'chart_height_{key_prefix}')
    chart_width = st.number_input(label='Chart Width',
                                    value=100, min_value=0, 
                                    max_value=1000, key=f'chart_width_{key_prefix}')

    algorithm_names = list(df['algo_x'].unique())
    colors = {x:y for x,y in zip(algorithm_names, COLORS['tableau20'][:len(algorithm_names)])}
    # Legend with color picker
    for idg, group in enumerate(algorithm_names):
        cols = st.columns([0.05, 0.95])
        colors[group] = cols[0].color_picker(label='', value=colors[group], 
                                           key=f'color_picker_{group}_{key_prefix}')
        cols[1].text_input("", value=group, key=f'group_name_{group}_{key_prefix}')


    df = df.sort_values(by=['algo_x'])
    # df['significant'] = df.apply(lambda x: x['prob'] > 0.5, axis=1)
    charts = []

    algo_x_select = st.multiselect("Select Algorithm X", df['algo_x'].unique(), key=f"{key_prefix}_algo_x_select",
                                    default=df['algo_x'].unique())
    
    prob_perf_cols = st.columns(len(algo_x_select))
    for idx, algo_x in enumerate(algo_x_select):
        algo_df = df.query(f"algo_x=='{algo_x}'")

        other_algos = algo_df['algo_y'].unique()

        other_colors = [colors[x] for x in other_algos]

        chart = alt.Chart(algo_df, title=algo_x, height=chart_height, width=chart_width).mark_bar(color = colors[algo_x]).encode(
            x = alt.X('algo_y:N', title=None,),
            y = alt.Y(f'prob_lower:Q', title=f'Probability of Improvement', scale=alt.Scale(domain=[0.0, 1.0])),
            y2 = alt.Y2(f'prob_upper:Q')
            # color=alt.Color('algo_y:N', legend=None, scale=alt.Scale(domain=other_algos, range=other_colors)),
        )
        # Add stroke for probability
        chart += alt.Chart(algo_df, height=chart_height, width=chart_width
                           ).mark_point(color='black', shape='stroke'
                           ).encode(x=alt.X('algo_y:N'),
                            y=alt.Y('prob:Q'),
                            )

        # Add line at 0.5
        chart += alt.Chart(pd.DataFrame({'y': [0.5]}), height=chart_height, width=chart_width).mark_rule(
            color='black',
            strokeDash=[5,5]
        ).encode(y='y')
        prob_perf_cols[idx].altair_chart(chart)

        

        # # st.write(df)
        # chart = alt.Chart(algo_df, height=chart_height, width=chart_width).mark_bar().encode(
        #     y = alt.Y(f'prob_lower:Q', title=None, 
        #                 scale=alt.Scale(domain=[0.0, 1.0]), 
        #                 axis=alt.Axis(labels=True, tickCount=10)),
        #     x=alt.X('algo_pair:N', axis=alt.Axis(labels=True, orient='bottom', labelExpr='split(datum.label, ",")[0]', title='Algorithm X')),
        #     y2= 'prob_upper:Q',
        #     color=alt.Color('algorithm:N', legend=None),
        # )
        # # Line for aggregate score
        # agg_chart = alt.Chart(algo_df).mark_point(color='black', shape='stroke').encode(
        #     x=alt.X('algo_pair:N', axis=alt.Axis(labels=True, orient='top', labelExpr='split(datum.label, ",")[1]', title='Algorithm Y')),
        #     y=alt.Y(f'prob:Q', title='P(Algorithm X > Algorithm Y)'),
        # )
        # chart = chart + agg_chart

    # layered_chart = alt.layer(
    #     chart.encode(x=alt.X('algo_x:Q', 
    #                          axis=alt.Axis(title='Top X-Axis', orient='top', labelExpr="datum.label", labels=True),
    #                          )),
    #     chart.encode(x=alt.X('algo_y:Q', 
    #                          axis= alt.Axis(title='Bottom X-Axis', orient='bottom', labelExpr="datum.label", labels=True)
    #                          ))
    # )
    # chart = layered_chart

        # charts.append(chart)
    # chart = alt.hconcat(*charts, spacing=50)
    # st.altair_chart(chart)

    # pi_fig, ax = plt.subplots(figsize=(4,3))
    # axes = plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis, ax=ax)
    # st.pyplot(pi_fig)
