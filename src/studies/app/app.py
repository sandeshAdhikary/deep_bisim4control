import streamlit as st
from src.studies.app.containers import StudyCard, NewStudyCard
from trainer.study import Study
# from src.studies.app.study import Study
import yaml


# st.set_page_config(layout="wide")
with open('./assets/css/style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.header("Dashboard")

# Load studies yaml
with open('./assets/studies/studies.yaml') as f:
    studies = yaml.load(f, Loader=yaml.SafeLoader)['studies']

num_cols = min(3, len(studies))
num_rows = (len(studies) // num_cols) + 1



# Build a grid for cards for studies
study_num = 0
for i in range(num_rows):
    cols = st.columns(num_cols)
    for idc, col in enumerate(cols):
        if study_num < len(studies):
            with col:
                card = StudyCard(studies[study_num])
                card.display()
            study_num += 1
        elif study_num == len(studies):
            with col:
                card = NewStudyCard()
                card.display()
                study_num += 1


# Add a 'New Study' card
            