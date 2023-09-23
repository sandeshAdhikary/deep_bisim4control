#!/bin/bash
cd /project
defaults_yaml="src/defaults.yaml"
yaml() {
    python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}
backup_folder=$(yaml "$defaults_yaml" "['BACKUP_FOLDER']")
logdir=$(yaml "$defaults_yaml" "['LOG_DIR']")

# Backup logs
# r=recursize
# z=compress and decompress at destination
# a=preserve permission and symbolic links
logfolders=$(find "$logdir" -maxdepth 1 -type d)
for folder in $logfolders; do
  if [ "$folder" != "$logdir" ]; then
    echo "Backing up subfolder: $folder"
    rsync -raz  "$folder" "$backup_folder"
  fi
done

# Delete local logs of runs that have already been synced
# Check of files synced more than an hour ago, and delete them
wandb sync --clean --clean-old-hours 1