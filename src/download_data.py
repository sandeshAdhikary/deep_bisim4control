from trainer.storage import Storage
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gridworld_trajectories', action='store_true', default=False)
    parser.add_argument('--distracted_dmc_images',action='store_true', default=False)
    parser.add_argument('--driving_stereo_images', action='store_true', default=False)
    parser.add_argument('--moving_mnist_images',action='store_true', default=False)
    parser.add_argument('--kinetics_videos', action='store_true', default=False)
    parser.add_argument('--dmc_trajectory', action='store_true', default=False)
    args = parser.parse_args()


    # Set the data folders to download based on arguments
    data_folders = {}
    if args.distracted_dmc_images:
        data_folders['distracted_dmc_images'] = {'remote': os.environ['REMOTE_DISTRACTED_DMC_DATASET']}
    if args.driving_stereo_images:
        data_folders['driving_stereo_images'] = {'remote': os.environ['REMOTE_DRIVING_STEREO_DATASET']}
    if args.moving_mnist_images:
        data_folders['moving_mnist_images'] = {'remote': os.environ['REMOTE_MOVING_MNIST_DATASET']}
    if args.kinetics_videos:
        data_folders['kinetics_videos'] = {'remote': os.environ['REMOTE_KINETICS_VIDEOS_DATASET']}
    if args.dmc_trajectory:
        data_folders['dmc_observations'] = {'remote': os.environ['REMOTE_DMC_OBSERVATIONS_DATASET']}
    if args.gridworld_trajectories:
        data_folders['gridworld_trajectories'] = {'remote': os.environ['REMOTE_GRIDWORLD_DATA_DATASET']}
        
    for k,v in data_folders.items():
        data_folders[k]['local'] = os.environ['LOCAL_DATA_DIR']

    if len(data_folders) == 0:
        raise ValueError("No datasets selected. Please select at least one dataset to download.")

    data_download_config = {
        'data_folders': data_folders,
        'storage': {
            'type': 'ssh',
            'host': os.environ['REMOTE_DATA_HOST'],
            'username': os.environ['REMOTE_DATA_USERNAME'],
            'password': os.environ['REMOTE_DATA_PASSWORD'],
            'root_dir': os.environ['REMOTE_DATA_DIR'],
            'overwrite': False,
        }
    }

    storage = Storage(data_download_config['storage'])
    for data_name, data_dict in data_download_config['data_folders'].items():
        print(f"Downloading {data_name}")
        storage.download(data_dict['remote'], data_dict['local'], extract_archives=True, delete_archive_after_extract=True)