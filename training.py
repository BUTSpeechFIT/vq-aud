import os
import json
import time
import torch
from torch import optim
import models
from utils import reduce_lr, stop_early, since

_optimizer_kinds = {'Adam': optim.Adam,
                    'SGD': optim.SGD}


class SimpleLoader:
    def initialize_args(self, **kwargs):
        for key, val in kwargs.items():
            self.params_dict[key] = val
        for key, val in self.params_dict.items():
            setattr(self, key, val)


class SimpleTrainingLoop(SimpleLoader):
    def __init__(self, model, device, **kwargs):
        self.params_dict = {
            'optimizer': 'Adam',
            'learning_rate': 1e-3,
            'num_epochs': 40,
            'verbose': True,
            'use_early_stopping': True,
            'early_stopping_loss': 0,
            'cooldown': 0,
            'num_epochs_early_stopping': 10,
            'delta_early_stopping': 1e-4,
            'learning_rate_lower_bound': 1e-6,
            'learning_rate_scale': 0.5,
            'num_epochs_reduce_lr': 4,
            'num_epochs_cooldown': 8,
            'use_model_checkpoint': True,
            'model_checkpoint_period': 1,
            'start_epoch': 0,
            'gradient_clip': None,
            'warmup_initial_scale': 0.2,
            'warmup_num_epochs': 2,
            'loss_weights': [1, 1, 0.2],
        }
        super(SimpleTrainingLoop, self).initialize_args(**kwargs)
        self.device = device
        if isinstance(model, str):
            # In this case it's just a path to a previously stored model
            modelsubdirs = sorted([int(_) for _ in os.listdir(model) if _.isnumeric()], reverse=True)
            if modelsubdirs:
                for modelsubdir in modelsubdirs:
                    modelsubdir_str = str(modelsubdir)
                    modeldir = os.path.join(model, modelsubdir_str)
                    if os.path.isfile(os.path.join(modeldir, 'loss_history.json')):
                        self.load_from_dir(modeldir)
                        print("Resuming from epoch {}".format(modelsubdir))
                        break
                else:
                    raise FileNotFoundError(os.path.join(model, '{{nnetdir}}', 'loss_history.json'))
            else:
                raise FileNotFoundError(os.path.join(model, '{{nnetdir}}'))
        else:
            self.model = model
            _opt = _optimizer_kinds[self.params_dict['optimizer']]
            self.optimizer = _opt(self.model.parameters(),
                                  lr=self.params_dict['learning_rate'])
            self.loss_history = {}

    def train_one_epoch(self, criterion, data_loaders, phases=('train', 'test')):
        epoch_beg = time.time()
        self.model.to(self.device)
        dataset_sizes = {x: len(data_loaders[x].dataset)
                         for x in phases}
        batch_sizes = {x: data_loaders[x].batch_size for x in phases}
        try:
            m = (1 - self.params_dict['warmup_initial_scale'])/(self.params_dict['warmup_num_epochs'] - 1)
            c = self.params_dict['warmup_initial_scale']
        except ZeroDivisionError:
            m = 1
            c = 0
        lr_scale = m * self.elapsed_epochs() + c
        if self.elapsed_epochs() < self.params_dict['warmup_num_epochs']:
            self.optimizer.param_groups[0]['lr'] *= lr_scale
        self.loss_history[str(self.elapsed_epochs())] = {}
        print('Epoch {}/{} - lr={}'.format(self.elapsed_epochs(), self.params_dict['num_epochs'],
                                           self.optimizer.param_groups[0]['lr'])
              )
        for phase in phases:
            print('\t{} '.format(phase.title()), end='')
            phase_beg = time.time()
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            running_loss = 0.
            running_dist_sum = 0.
            running_count = 0.
            running_count_per_class = torch.zeros(self.model.num_centroids)

            for batch_no, batch_data in enumerate(data_loaders[phase]):
                self.optimizer.zero_grad()
                data_batch, label_batch = batch_data
                data_batch = data_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                with torch.set_grad_enabled(phase == 'train'):
                    encoded = self.model.encoder(data_batch).contiguous()
                    quantized, alignment = self.model.quantize(encoded, return_alignment=True)
                    count_per_class = torch.tensor([(alignment == i).sum()
                                                    for i in range(self.model.num_centroids)])
                    running_count_per_class += count_per_class
                    output = self.model.decoder(quantized)
                    predicted_centroids = self.model.centroids[alignment]
                    encoded = encoded.view(-1, encoded.size(-1))
                    # In case of subsampling
                    new_length = output.size(1)
                    # old_length = label_batch.size(1)
                    # if old_length % new_length:
                    #     subs_factor = int(old_length / new_length) + 1
                    # else:
                    #     subs_factor = int(old_length / new_length)
                    # label_batch = label_batch[:, ::subs_factor]
                    label_batch = label_batch[:, :new_length]
                    distance_loss = criterion['dis_loss'](encoded, predicted_centroids.detach())
                    commitment_loss = criterion['com_loss'](predicted_centroids, encoded.detach())
                    reconstruction_loss = criterion['rec_loss'](output, label_batch)
                    total_loss = reconstruction_loss + distance_loss + commitment_loss
                    # Not part of the graph, just metrics to be monitored
                    mask = (label_batch != data_loaders[phase].dataset.pad_value).float()
                    numel = torch.sum(mask).item()
                    running_loss += reconstruction_loss.item()
                    running_dist_sum += distance_loss.item()
                    running_count += numel

                    class0_sum = running_dist_sum / running_count
                    px = running_count_per_class / running_count_per_class.sum()
                    px.clamp_min_(1e-20)
                    entropy = -(px * torch.log2(px)).sum().item()
                if phase == 'train':
                    total_loss.backward()
                    if self.params_dict['gradient_clip']:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.params_dict['gradient_clip'])
                    self.optimizer.step()
                phase_elapse = since(phase_beg)
                eta = int(phase_elapse
                          * (dataset_sizes[phase] // batch_sizes[phase]
                             - batch_no - 1)
                          / (batch_no + 1))
                if self.params_dict['verbose']:
                    print('\r\t{} batch: {}/{} batches - ETA: {}s - loss: {:.4f} - dist: {:.4f} - '
                          'entropy: {:.4f}'.format(phase.title(),
                                                   batch_no + 1,
                                                   dataset_sizes[phase] // batch_sizes[phase] + 1,
                                                   eta, running_loss/running_count,
                                                   class0_sum, entropy
                                                   ), end='')
            epoch_loss = running_loss/running_count
            print(" - loss: {:.4f} - dist: {:.4f} - entropy: {:.4f}".format(epoch_loss, class0_sum,
                                                                            entropy))
            self.loss_history[str(self.elapsed_epochs() - 1)][phase] = (epoch_loss,
                                                                        class0_sum, entropy,
                                                                        running_count)
        print('\tTime: {}s'.format(int(since(epoch_beg))))
        if self.elapsed_epochs() <= self.params_dict['warmup_num_epochs']:
            self.optimizer.param_groups[0]['lr'] /= lr_scale

    def train(self, outdir, criterion, data_loaders,
              phases=('train', 'test'),
              job_num_epochs=None,
              **kwargs):
        if not job_num_epochs:
            job_num_epochs = self.params_dict['num_epochs']
        for i in range(self.params_dict['num_epochs']):
            self.train_one_epoch(criterion, data_loaders, phases, **kwargs)
            if self.params_dict['use_model_checkpoint']\
                    and (self.elapsed_epochs() % self.params_dict['model_checkpoint_period'] == 0):
                self.save_to_dir(os.path.join(outdir,
                                              str(self.elapsed_epochs())))
            history_sum = [self.loss_history[str(_)][phases[-1]][self.params_dict['early_stopping_loss']]
                           for _ in range(self.elapsed_epochs())]
            if history_sum[-1] == min(history_sum):
                self.save_to_dir(os.path.join(outdir, 'best'))
            rl = reduce_lr(history=history_sum,
                           lr=self.optimizer.param_groups[0]['lr'],
                           cooldown=self.params_dict['cooldown'],
                           patience=self.params_dict['num_epochs_reduce_lr'],
                           mode='min',
                           difference=self.params_dict['delta_early_stopping'],
                           lr_scale=self.params_dict['learning_rate_scale'],
                           lr_min=self.params_dict['learning_rate_lower_bound'],
                           cool_down_patience=self.params_dict['num_epochs_cooldown'])
            self.optimizer.param_groups[0]['lr'], self.params_dict['cooldown'] = rl
            if self.params_dict['use_early_stopping']:
                if stop_early(history_sum,
                              patience=self.params_dict['num_epochs_early_stopping'],
                              mode='min',
                              difference=self.params_dict['delta_early_stopping']):
                    print('Stopping Early.')
                    break
            if self.elapsed_epochs() >= self.params_dict['num_epochs']:
                break
            if i >= job_num_epochs:
                return
        with open(os.path.join(outdir, '.done.train'), 'w') as _w:
            pass

    def elapsed_epochs(self):
        return len(self.loss_history)

    def load_from_dir(self, trainer_dir, model_kind=models.VQVAE):
        if os.path.isfile(os.path.join(trainer_dir, 'nnet_kind.txt')):
            model_classname = open(os.path.join(trainer_dir, 'nnet_kind.txt')).read().strip()
            model_kind = models.get_model(model_classname)
        self.model = model_kind.load_from_dir(trainer_dir)
        self.model.to(self.device)
        with open(os.path.join(trainer_dir, 'optimizer.txt')) as _opt:
            opt_name = _opt.read()
        _opt = _optimizer_kinds[opt_name]
        self.optimizer = _opt(self.model.parameters(), lr=1e-3)
        self.optimizer.load_state_dict(torch.load(
            os.path.join(trainer_dir, 'optimizer.state'))
        )
        jsonfile = os.path.join(trainer_dir, 'trainer.json')
        with open(jsonfile) as _json:
            self.params_dict = json.load(_json)
        jsonfile = os.path.join(trainer_dir, 'loss_history.json')
        with open(jsonfile) as _json:
            self.loss_history = json.load(_json)

    def save_to_dir(self, trainer_dir):
        if not os.path.isdir(trainer_dir):
            os.makedirs(trainer_dir)
        self.model.save(trainer_dir)
        opt_name = str(self.optimizer).split()[0]
        with open(os.path.join(trainer_dir, 'optimizer.txt'), 'w') as _opt:
            _opt.write(opt_name)
        torch.save(self.optimizer.state_dict(),
                   os.path.join(trainer_dir, 'optimizer.state')
                   )
        with open(os.path.join(trainer_dir, 'trainer.json'), 'w') as _json:
            json.dump(self.params_dict, _json)
        with open(os.path.join(trainer_dir, 'loss_history.json'), 'w') as _json:
            json.dump(self.loss_history, _json)


_trainers = {'Simple': SimpleTrainingLoop}


def get_trainer(trainer_name):
    return _trainers[trainer_name]
