#!/bin/bash

cd /project

# Set permissions for the scripts
chown -R root src/scripts

## Install the project with pip
pip install -e .

## Log in to wandb
wandb login

# Import required data from the remote
scp -r sandesh@10.19.137.42:/home/sandesh/Repositories/deep_bisim4control/src/distractors/images src/distractors/images

#### Set up auto log backup every hour ####
# Set up cron job to run periodic backups
CRON_COMMAND="cd /project & /src/scripts/backup_logs.sh"
CRON_SCHEDULE="0 * * * *" # Run every 
# Add the cron job to the user's crontab
(crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -
# Check if the cron job was added successfully
if [ $? -eq 0 ]; then
  echo "Cron job added. The log backup script will run every hour"
else
  echo "Failed to add the cron job."
fi

# add to ssh agent to avoid passphrase
ssh-keygen -t rsa -b 4096 -C "adhikary.sandesh@gmail.com"
ssh-copy-id -i /root/.ssh/id_rsa.pub sandesh@10.19.137.42
exec ssh-agent $SHELL
ssh-add

