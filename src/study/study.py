from trainer.study import Study
from omegaconf import OmegaConf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_config', type=str, default=None)
    parser.add_argument('--run_config', type=str, default=None)
    args = parser.parse_args()

    # study_config_path = '/project/src/study/configs/study.yaml'
    # run_config_path = '/project/src/study/configs/train_sample.yaml'
    # study_config_path = 'src/study/configs/cartpole_study_default.yaml'
    # run_config_path = 'src/study/configs/cartpole_study_dbc_kinetics.yaml'

    study_cfg = OmegaConf.load(args.study_config)
    run_cfg = OmegaConf.load(args.run_config)
    
    study = Study(study_cfg)

    # study.sweep(run_cfg)
    study.train(run_cfg)
    # study.evaluate(run_cfg)