# Import required data from the remote
cd /project
defaults_yaml="src/defaults.yaml"
yaml() {
    python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

data_path=$(yaml "$defaults_yaml" "['DATA_FOLDER']")

scp -r "$data_path/*" "src/data"
