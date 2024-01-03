# import os
# import yaml
# import random
# from src.studies.app.defaults import defauts
# from omegaconf import DictConfig, OmegaConf
# import streamlit as st

# class Study():
#     def __init__(self, config=None):
#         if config is None:
#             config = {}
#         self.config = config['path']
#         assert os.path.exists(self.config), f"{self.config} does not exist"
#         self.config = OmegaConf.load(self.config)['study']
#         self.study_name = self.config.get('name', 'Test Study')
#         self.desc = self.config.get('desc', 'Description for the study')
#         # Get icon
#         self.icon = self.config.get('icon')
#         if self.icon is None or not os.path.exists(self.icon):
#             self.icon = defauts['study_icon']
#         self.study_id = config.get('study_id', str(hash(self.study_name))
#                                    + str(random.randint(0, 1000)))
