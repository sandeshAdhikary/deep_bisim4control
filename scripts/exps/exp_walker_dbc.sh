#!/bin/bash

experiment_name="walker_dbc"
configs_folder="src/experiments/walker/dbc" 
cd $ROOT_DIR

# Use or create experiment_log_folder to save std out logs
experiment_log_folder="src/_exp_script_logs"
if [ ! -d "$experiment_log_folder" ]; then
    # If it doesn't exist, create it
    mkdir -p "$experiment_log_folder"
    echo "Folder created: $experiment_log_folder"
else
    echo "Folder already exists: $experiment_log_folder"
fi

# # Get list of config files in the exp folder
configs=($(find "$configs_folder" -type f -name "*.yaml" -exec basename {} \;))
num_config=${#configs[@]}
for ((idx=0; idx<num_config; idx++)); do
    sub_exp_name="${configs[idx]%%.*}"
    config="$configs_folder/${configs[idx]}"
    cmd="python src/sweep.py --config $config"
    current_datetime=$(date "+%m-%d-%H-%M-%S")
    tmux new-session -d -s "$experiment_name-$sub_exp_name-$current_datetime" "$cmd > $experiment_log_folder/$experiment_name-$sub_exp_name-$current_datetime.log"
done
