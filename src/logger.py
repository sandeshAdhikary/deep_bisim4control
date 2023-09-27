# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored
import wandb
from einops import rearrange
import shutil

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), ('step', 'S', 'int'), 
            ('duration', 'D', 'time'), ('episode_reward', 'R', 'float'),
            ('batch_reward', 'BR', 'float'), ('actor_loss', 'ALOSS', 'float'),
            ('critic_loss', 'CLOSS', 'float'), ('ae_loss', 'RLOSS', 'float'),
            ('max_rat', 'MR', 'float')
        ],
        'eval': [('step', 'S', 'int'), ('episode_reward', 'ER', 'float')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()

class Logger(object):
    def __init__(self, config=None): 
        self._project_name = config.get('project', 'misc')
        self._log_dir = os.path.join(config['log_dir'], self._project_name)
        # Set image downscaling factor to prevent huge file sizes. 1 means no downscaling
        self.img_downscale_factor = config.get('img_downscale_factor', 3)
        self.img_downscale_factor = max(int(self.img_downscale_factor), 1)

        format_config = config.get('format_config', 'rl')

        self.sw_type = config.get('sw', None) # Summary writer
        if self.sw_type == 'tensorboard':
            tb_dir = os.path.join(self._log_dir, 'tb')
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        elif self.sw_type == 'wandb':
            project = config.get('project', self._project_name)
            tracked_params = config.get('tracked_params', {})
            tags = config.get('logger_tags', None)
            if not isinstance(tags, list):
                tags = list(tags)
            self._sw = wandb.init(project=project, dir=self._log_dir, config=tracked_params, tags=tags)
        else:
            self._sw = None

        self._train_mg = MetersGroup(
            os.path.join(self._log_dir, 'train.log'),
            formating=FORMAT_CONFIG[format_config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(self._log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[format_config]['eval']
        )

    def _try_sw_log(self, key, value, step):
        if self.sw_type == 'tensorboard':
            self._sw.add_scalar(key, value, step)
        elif self.sw_type == 'wandb':
            self._sw.log({key: value}, step=step)

    def _try_sw_log_image(self, key, image, step, image_mode='hwc'):
        if self.sw_type == 'tensorboard':
            if not torch.is_tensor(image):
                image = torch.from_numpy(image)
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(0))
            self._sw.add_image(key, grid, step)
        elif self.sw_type == 'wandb':
            if image_mode == 'chw':
                image = rearrange(image, 'c h w -> h w c')
            if torch.is_tensor(image):
                image = image.detach().cpu().numpy()
            image = image[:,::self.img_downscale_factor,::self.img_downscale_factor]
            self._sw.log({key: [wandb.Image(image)]}, step=step)

    def _try_sw_log_video(self, key, frames, step, image_mode='hwc'):
        if self.sw_type == 'tensorboard':
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step)
        elif self.sw_type == 'wandb':
            frames = np.array(frames)
            if image_mode == 'hwc':
                frames = rearrange(frames, 't h w c -> t c h w')
            frames = frames[:,:,::self.img_downscale_factor,::self.img_downscale_factor]
            self._sw.log({key: wandb.Video(frames, fps=1)}, step=step)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self.sw_type == 'tensorboard':
            self._sw.add_histogram(key, histogram, step)
        elif self.sw_type == 'wandb':
            histogram_np = histogram
            if isinstance(histogram, torch.Tensor):
                histogram_np = histogram.detach().cpu().numpy()
            self._sw.log({key: wandb.Histogram(histogram_np)}, step=step)

    def _try_sw_log_table(self, key, data, step):
        if self.sw_type == 'wandb':
            data = data.reshape(data.shape[0], -1)
            table = wandb.Table(data=list(data), columns=list(range(data.shape[1])))
            self._sw.log({key: table}, step=step)

    def _try_sw_log_agent(self, key, agent, step):
        from datetime import datetime
        if self.sw_type == 'wandb':
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            temp_dir = os.path.join(self._log_dir, 'temp', f'{self._sw.entity}-{self._sw.name}-models-{timestamp}')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            torch.save(
                agent.actor.state_dict(), os.path.join(temp_dir,'actor.pt')
            )
            torch.save(
                agent.critic.state_dict(), os.path.join(temp_dir,'critic.pt')
            )
            torch.save(
                agent.reward_decoder.state_dict(), os.path.join(temp_dir,'reward_decoder.pt')
            )

            artifact = wandb.Artifact(key, type="model")
            artifact.add_dir(temp_dir)
            self._sw.log_artifact(artifact)

            # Delete the temp folder
            shutil.rmtree(temp_dir)

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_image(self, key, image, step, image_mode='hwc'):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_image(key, image, step, image_mode)

    def log_video(self, key, frames, step, image_mode='hwc'):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step, image_mode)

    def log_histogram(self, key, histogram, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def log_table(self, key, data, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_table(key, data, step)
    
    def log_agent(self, key, model, step):
        self._try_sw_log_agent(key, model, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')

    def finish(self):
        if self.sw_type == 'wandb':
            self._sw.finish()