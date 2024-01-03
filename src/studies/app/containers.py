import streamlit as st
from src.studies.app.defaults import defauts
import os
from omegaconf import DictConfig, OmegaConf
import random

with open('./assets/css/style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

class StudyCard():

    def __init__(self, config) -> None:
        if config is None:
            config = {}
        self.config = config['path']
        assert os.path.exists(self.config), f"{self.config} does not exist"
        self.config = OmegaConf.load(self.config)['study']
        self.study_name = self.config.get('name', 'Test Study')
        self.desc = self.config.get('desc', 'Description for the study')
        # Get icon
        self.icon = self.config.get('icon')
        if self.icon is None or not os.path.exists(self.icon):
            self.icon = defauts['study_icon']
        self.study_id = config.get('study_id', str(hash(self.study_name))
                                   + str(random.randint(0, 1000)))
        self.width = config.get('icon_width', 200)
        self.height = config.get('icon_width', 200)

    def display(self):
        container = st.container(border=True)
        with container:
            st.markdown(f'<a href="/analyze_study?study_id={self.study_id}&study_name={self.study_name}" target="_self">{self.study_name}</a>', unsafe_allow_html=True)
            st.image(self.icon, width=self.width)
            st.markdown(self.desc)
        return container

class NewStudyCard():
    def __init__(self) -> None:
        self.icon = defauts['new_study_icon']
        self.width = 200
        self.height = 200

    def display(self):
        container = st.container(border=True)
        with container:
            st.markdown(f'<a href="/new_study" target="_self">Add New Study</a>', unsafe_allow_html=True)
            st.image(self.icon, width=self.width)

        return container