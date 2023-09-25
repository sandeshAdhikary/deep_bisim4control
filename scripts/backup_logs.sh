#!/bin/bash
cd $ROOT_DIR
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
    sync_success=$?
    # Check the exit status
    if [ "$sync_success" -eq 0 ]; then
        echo "Synced $folder successfully."
        wandb sync $folder --clean --clean-old-hours 1
    else
        echo "Backup sync failed with exit status $sync_success."
    fi

  fi
done

