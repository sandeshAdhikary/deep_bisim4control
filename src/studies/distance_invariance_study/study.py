from trainer.study import Study
from omegaconf import DictConfig, OmegaConf
import hydra
from trainer.utils import import_module_attr
from copy import deepcopy

class NeuralEFStudy(Study):
    def _make_model(self, config, trainer):
        """
        Overwrite default _make_model to pass trainer.kernel to the model
        """
        model_config = self._merge_configs(self.config['model'], config.get('model', {}))
        model_cls = import_module_attr(model_config['module_path'])
        return model_cls(dict(model_config), trainer=trainer)

@hydra.main(version_base=None, config_path="configs", config_name='default_config')
def main(cfg: DictConfig) -> (DictConfig, DictConfig):

    # Resolve the config
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=False))

    # Get the composed config
    study_config = deepcopy(cfg)
    project_overrides = deepcopy(cfg['project']['overrides'])
    exp_config = deepcopy(cfg['exp'])
    # study_config.__delattr__('project')
    study_config.__delattr__('exp')

    # Project overrides
    study_config = OmegaConf.merge(study_config, project_overrides)

    # Define study
    study = NeuralEFStudy(study_config)

    # Get the experiment mode
    exp_mode = exp_config['exp_mode']
    exp_config.__delattr__('exp_mode')

    # Run experiment
    if exp_mode == 'train': 
        study.train(exp_config)
    elif exp_mode == 'sweep':
        study.sweep(exp_config, num_runs=exp_config['sweeper']['num_runs'])
    elif exp_mode == 'evaluate':
        study.evaluate(exp_config)
    else:
        raise ValueError(f'exp_mode {exp_mode} not recognized')


if __name__ == '__main__':
    main()
