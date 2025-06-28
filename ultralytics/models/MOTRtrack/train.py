# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy
import time
import torch
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import TrackingModel
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from .val import TrackDataset, TrackValidator
from ultralytics.data import build_dataloader, build_track_dataloader
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks, clean_url,
                               colorstr, emojis, yaml_save)
from tqdm import tqdm
from torch import distributed as dist
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
import subprocess
import os
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, print_args
from torch.cuda import amp
from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle, select_device,
                                           strip_optimizer)
import math
from ultralytics.utils.autobatch import check_train_batch_size
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim


class TrackTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg=DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)
        self.track_batch_datasets = {}
        world_size = torch.cuda.device_count()
        ##   记得去掉注释
        if self.batch_size != 1:
            import warnings
            warnings.warn("batch只能为1")
            self.batch_size = 1

        batch_size = self.batch_size // max(world_size, 1)

        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        # self._build_track_dataset()

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = TrackingModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build RTDETR Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """

        return TrackDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        # boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.dataset.labels], 0)
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.dataset.labels], 0)
        # cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        # boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        from ultralytics.utils.plotting import plot_labels
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)

    def get_dataloader(self, dataset_path, batch_size=1, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        if mode == 'train':
            shuffle = True
        else:
            shuffle = False
        return build_track_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def get_validator(self):
        """Returns a DetectionValidator for RTDETR model validation."""
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return TrackValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""

        batch = super().preprocess_batch(batch)
        # print('preprocess_batch batch:',batch)
        # print(type(batch))
        bs = len(batch['img'])
        # print('len:',bs)
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class,gt_id = [], [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
            # gt_id.append(batch['track_id'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, int) or self.args.device:  # i.e. device=0 or device=[0,1,2,3]
            world_size = torch.cuda.device_count()
        elif torch.cuda.is_available():  # i.e. device=None or device=''
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting rect=False")
                self.args.rect = False
            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'DDP command: {cmd}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches

        nw = max(round(self.args.warmup_epochs *
                       nb), 100) if self.args.warmup_epochs > 0 else -1  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases

        # 主要是为了增加视频流输入的随机性，不然全按一个顺序输入训练了,当然主要是不想重新写一篇数据加载了，下面哪个_build_track_dataset会爆内存，算了就随机数吧
        import random

        self.update_model_num = 2  # 我也想一个视频更新一次，奈何显存不允许 注释，要改

        for epoch in range(self.start_epoch, self.epochs):
            skip_probability = 0
            if epoch > 1:
                # 暂定最高0.7，最低0.3
                skip_probability = max(0.7 * (200 - 100 + epoch) / 200, 0.3)
                if skip_probability > 0.7:
                    skip_probability = 0.7

            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            # print(self.train_loader)
            label_dict = {label['im_file'][-15:-4]: label['track_id'] for label in
                          self.train_loader.dataset.dataset.labels}
            # label_dict = {label['im_file'][-15:-4]: label['track_id'] for label in
            #               self.train_loader.dataset.labels}
            video_name = ""

            is_skip = False
            list_video_names = []

            for i, batch in pbar:
                is_first = True
                frame_count = 0

                for item_batch in batch:
                    img_name = item_batch['im_file'][0][-15:-4]

                    # 从item_batch中获取图像路径和边界框坐标

                    # item_batch['track_id'] = label_dict.get(img_name, None)

                    # Forward
                    with torch.cuda.amp.autocast(self.amp):
                        item_batch = self.preprocess_batch(item_batch)
                        # print(item_batch)
                        # loss，大概是
                        self.loss_one_frame, self.loss_items_one_frame, num_object = self.model(item_batch,
                                                                                                is_first=is_first)
                    is_first = False
                    if frame_count == 0:
                        self.loss = self.loss_one_frame
                        self.all_object_num = num_object + 1
                        frame_count += 1
                    else:
                        self.loss += self.loss_one_frame
                        self.all_object_num += num_object

                # Warmup 这一段可能要改？
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        # x['lr'] = 0.0002
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                self.loss /= self.all_object_num
                # self.loss_one_frame /= num_object
                self.scaler.scale(self.loss).backward()
                # if ni - last_opt_step >= self.accumulate:
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                if RANK != -1:
                    self.loss_one_frame *= world_size
                self.tloss = (self.tloss * i + (self.loss_items_one_frame / (num_object + 1))) / (
                        i + 1) if self.tloss is not None \
                    else self.loss_items_one_frame / (num_object + 1)
                # # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                # # emmm，啥时候更新权重这是一个好问题
                # if ni - last_opt_step >= self.accumulate:
                #     self.optimizer_step()
                #     last_opt_step = ni
                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch[0]['cls'].shape[0],
                         batch[0]['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch[0], ni)

            # 一轮结束
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):
                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()

                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors
            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()  # 节约时间，不要了
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def _build_track_dataset(self):
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        nb = len(self.train_loader)  # number of batches
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        if RANK in (-1, 0):
            LOGGER.info(self.progress_string())
            pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
        label_dict = {label['im_file'][-15:-4]: label['track_id'] for label in self.train_loader.dataset.labels}
        for i, batch in pbar:
            img_name = batch['im_file'][0][-15:-4]
            video_name = img_name[0:4]
            batch['track_id'] = label_dict.get(img_name, None)
            batch['frames_num'] = int(img_name[5:11])
            try:
                self.track_batch_datasets[video_name].append(batch)
            except KeyError:
                self.track_batch_datasets[video_name] = [batch]

        # for video_name, batches in self.track_batch_datasets.items():
        #     sorted_batches = sorted(batches, key=lambda x: x['frames_num'])
        #     self.track_batch_datasets[video_name] = sorted_batches
        # print(self.track_batch_datasets)

    def _setup_train(self, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
        if self.batch_size == -1:
            if RANK == -1:  # single-GPU only, estimate best batch size
                self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        # print(self.trainset)

        # print(self.train_loader)
        # print(self.train_loader.dataset)

        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size, rank=-1,
                                                   mode='val')  # 坑爹啊，这里的batch——size原本会乘2,注释
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            # print(self.model)
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        # for group in self.optimizer.param_groups:
        #     for param in group["params"]:
        #         print(param.grad)
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize RTDETR model given training data and device."""
    model = 'rtdetr-l.yaml'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    # NOTE: F.grid_sample which is in rt-detr does not support deterministic=True
    # NOTE: amp training causes nan outputs and end with error while doing bipartite graph matching
    args = dict(model=model,
                data=data,
                device=device,
                imgsz=640,
                exist_ok=True,
                batch=4,
                deterministic=False,
                amp=False)
    trainer = TrackTrainer(overrides=args)
    trainer.train()


if __name__ == '__main__':
    train()
