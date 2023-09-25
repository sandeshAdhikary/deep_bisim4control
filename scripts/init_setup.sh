root_dir="/project"

# Save the root dir as env variable
export ROOT_DIR=$root_dir

defaults_yaml="src/defaults.yaml"
num_steps=5
# # Set permissions for the scripts
echo "Setting execute permissions to run bash scripts."
sleep 2
chown -R root scripts
echo "Done [1/$num_steps]"

## Install the project with pip
echo "Installing the project with pip"
sleep 2
pip install -e .
echo "Done [2/$num_steps]"

# Login to wandb
echo "Setting up wandb"
sleep 2
wandb login
echo "Done [3/$num_steps]"

# ## Download data
echo "Downloading data"
sleep 2
bash scripts/download_data.sh
echo "Done [4/$num_steps]"

# #### Set up auto log backup every hour ####
# Set up cron job to run periodic backups
echo "Setting up cron job to backup logs"
sleep 2
CRON_COMMAND="cd $root_dir & scripts/backup_logs.sh"
CRON_SCHEDULE="5 * * * *" # Run every 5 hours
# Add the cron job to the user's crontab
(crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -
# Check if the cron job was added successfully
if [ $? -eq 0 ]; then
  echo "Cron job added. The log backup script will run every 5 hours"
else
  echo "Failed to add the cron job."
fi
echo "Done [4/$num_steps]"

# add to ssh agent to avoid passphrase
echo "Setting up ssh agent"
sleep 2
ssh_email=$(yaml "$defaults_yaml" "['SSH_EMAIL']")
remote_host==$(yaml "$defaults_yaml" "['REMOTE_HOST']")
ssh-keygen -t rsa -b 4096 -C "$ssh_email"
ssh-copy-id -i /root/.ssh/id_rsa.pub $remote_host
exec ssh-agent $SHELL
ssh-add
echo "Done [5/$num_steps]"

