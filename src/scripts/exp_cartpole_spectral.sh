#!/bin/bash

experiment_name=cartpole_spectral
parent_dir=$(dirname "$PWD")
configs_folder="$parent_dir/experiments/cartpole/spectral"

# # Get list of config files in the exp folder
configs=($(find "$configs_folder" -type f -name "*.yaml" -exec basename {} \;))

num_config=${#configs[@]}
for ((idx=0; idx<num_config; idx++)); do
    sub_exp_name="${configs[idx]%%.*}"
    config="/experiments/cartpole/dbc/${configs[idx]}"
    cmd="python $parent_dir/sweep.py --config $config"
    echo $cmd
done

