#!/bin/bash

experiment_name="walker_spectral"
configs_folder="src/experiments/cartpole/walker" 
cd /project

# # Get list of config files in the exp folder
configs=($(find "$configs_folder" -type f -name "*.yaml" -exec basename {} \;))
num_config=${#configs[@]}
for ((idx=0; idx<num_config; idx++)); do
    sub_exp_name="${configs[idx]%%.*}"
    config="$configs_folder/${configs[idx]}"
    cmd="python src/sweep.py --config $config"
    # echo $cmd
    tmux new-session -d -s "$experiment_name-$sub_exp_name" "$cmd"
done
