from trainer.storage import Storage
import os

if __name__ == "__main__":

    data_download_config = {
        'data_folders' : {
            'distracted_dmc_images': {'remote': os.environ['REMOTE_DISTRACTED_DMC_DATASET'], 
                               'local': os.environ['LOCAL_DATA_DIR']
                               },
            'driving_stereo_images': {'remote': os.environ['REMOTE_DRIVING_STEREO_DATASET'], 
                    'local': os.environ['LOCAL_DATA_DIR']
                    },
            'moving_mnist_images': {'remote': os.environ['REMOTE_MOVING_MNIST_DATASET'], 
                             'local': os.environ['LOCAL_DATA_DIR']
                             },
            'kinetics_videos': {'remote': os.environ['REMOTE_KINETICS_VIDEOS_DATASET'], 
                                'local': os.environ['LOCAL_DATA_DIR']
                                }
        },
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