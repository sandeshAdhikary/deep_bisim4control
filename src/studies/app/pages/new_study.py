import streamlit as st
import os
import yaml

st.header("Create a new study")

form = st.form("new_study")
with form:
    study_name = st.text_input('Study Name')
    path = st.text_input('Path')
    st.divider()
    study_desc = st.text_area('Study Description')
    study_image = st.file_uploader('Study Image')

    submit_button = st.form_submit_button(label='Create Study')


    if submit_button:
        print("Submitted!")
        studies_entry = {
            'name': study_name,
            'path': path,
        }

        # Add study config to studies.yaml
        with open('./assets/studies/studies.yaml', 'r') as f:
            studies = yaml.load(f, Loader=yaml.SafeLoader)
            # st.write(studies['studies'])
            existing_studies = [s['name'] for s in studies['studies']]
            st.write(study_name)
            if study_name in existing_studies:
                st.error(f"Study with name {study_name} already exists")
                st.stop()
            else:
                studies['studies'].append(studies_entry)
        with open('./assets/studies/studies.yaml', 'w') as f:
            yaml.dump(studies, f)
                
        # Create a new folder for the study
        if not os.path.exists(path):
            os.makedirs(path)
            st.info(f"Created new folder at {path}")
        else:
            st.info(f"Folder {path} already exists")

        # Create study.yaml
        study_config = {
            'name': study_name,
            'desc': study_desc,
        }
        with open(os.path.join(path, 'study.yaml'), 'w') as f:
            yaml.dump(study_config, f)
        st.info(f"Saved study.yaml at {path}/study.yaml")

        # Save icon if provided
        if study_image is not None:
            with open(os.path.join(path, 'icon.png'), 'wb') as f:
                f.write(study_image.read())
            st.info(f"Saved icon.png at {path}/icon.png")

        
