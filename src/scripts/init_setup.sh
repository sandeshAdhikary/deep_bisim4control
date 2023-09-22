#!/bin/bash


#### Set up auto log backup every hour ####
# Set up cron job to run periodic backups
CRON_COMMAND="/src/scripts/backup_logs.sh"
CRON_SCHEDULE="0 * * * *" # Run every 
# Add the cron job to the user's crontab
(crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -
# Check if the cron job was added successfully
if [ $? -eq 0 ]; then
  echo "Cron job added. The log backup script will run every hour"
else
  echo "Failed to add the cron job."
fi