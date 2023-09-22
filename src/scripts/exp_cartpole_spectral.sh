#!/bin/bash

experiment_name="cartpole_spectral"
configs_folder="experiments/cartpole/spectral" 
cd /project/src

# # Get list of config files in the exp folder
configs=($(find "$configs_folder" -type f -name "*.yaml" -exec basename {} \;))
num_config=${#configs[@]}
for ((idx=0; idx<num_config; idx++)); do
    sub_exp_name="${configs[idx]%%.*}"
    config="$configs_folder/${configs[idx]}"
    cmd="python sweep.py --config $config"
    # echo $cmd
    tmux new-session -d -s "$experiment_name-$idx" "$cmd"
done
