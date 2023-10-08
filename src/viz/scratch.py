import altair as alt
import numpy as np
import pandas as pd

data_gridworld = {
    'distractor': [0.0, 0.2, 0.4, 0.8, 1.0, 0.0, 0.2, 0.4, 0.8, 1.0],
    'project': ['dbc', 'dbc', 'dbc', 'dbc', 'dbc', 'spectral', 'spectral', 'spectral', 'spectral', 'spectral'],
    'value': [8.3, 5.6, 7.2, 7.7, 5.4, 8.1, 8.0, 7.7, 7.9, 7.4]
}

data_cheetah = {
    'distractor': [0.0, 0.2, 0.4, 0.8, 1.0, 0.0, 0.2, 0.4, 0.8, 1.0],
    'project': ['dbc', 'dbc', 'dbc', 'dbc', 'dbc', 'spectral', 'spectral', 'spectral', 'spectral', 'spectral'],
    'value': [75.4, 52.9, 63.1, 41.8, 37.1, 45.7, 51.3, 66.3, 38.2, 29.6]
}
data = data_gridworld
# data = data_cheetah

data = pd.DataFrame(data)
# Create the Altair bar chart
chart = alt.Chart(data).mark_bar().encode(
    x=alt.X('distractor:O', title='Distraction Level'),
    y=alt.Y('value:Q', title='Episode Reward (Eval)'),
    xOffset='project:N',
    color=alt.Color('project:N', title='Model'),
).properties(
    width=800,
    height=500
).configure_axis(
    labelFontSize=20,
    titleFontSize=20,
).configure_legend(
    labelFontSize=20,
    titleFontSize=20,
)

chart.save('scratch_gridworld.html')
# chart.save('scratch_cheetah.html')