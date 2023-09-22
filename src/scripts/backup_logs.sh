#!/bin/bash

# Source and destination directories
logdir='/project/src/logdir/'
backup_host='sandesh@10.19.137.42'
backup_dir='~/shared/wandb_backup/log_dir'

# Backup logs
# r=recursize
# z=compress and decompress at destination
# a=preserve permission and symbolic links
rsync -raz "$logdir" "$backup_host:$backup_dir"

# Delete local logs of runs that have already been synced
# Check of files synced more than an hour ago, and delete them
wandb sync --clean --clean-old-hours 1