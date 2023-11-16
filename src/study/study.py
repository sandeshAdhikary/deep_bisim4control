from trainer.study import Study
from omegaconf import OmegaConf

if __name__ == '__main__':

    # study_config_path = '/project/src/study/configs/study.yaml'
    # run_config_path = '/project/src/study/configs/train_sample.yaml'
    study_config_path = 'src/study/configs/cartpole_study_default.yaml'
    run_config_path = 'src/study/configs/cartpole_study_spectral_kinetics.yaml'

    study_cfg = OmegaConf.load(study_config_path)
    run_cfg = OmegaConf.load(run_config_path)
    
    study = Study(study_cfg)

    study.sweep(run_cfg)
    # study.train(run_cfg)
    # study.evaluate(run_cfg)