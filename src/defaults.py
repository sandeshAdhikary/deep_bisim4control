import yaml
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(f'{PROJECT_DIR}/src/defaults.yaml') as f:
    DEFAULTS = yaml.load(f, Loader=yaml.FullLoader)
DEFAULTS['project_dir'] = PROJECT_DIR