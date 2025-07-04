# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch
from yaml import warnings

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import TrackingModel
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr

from .val import TrackDataset, TrackValidator


class TrackTrainer(DetectionTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = TrackingModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode='val', batch=1):
        """Build RTDETR Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.跟踪的只能是1，因为最好是用整个视频来训练
        """
        if batch != 1:
            warnings("batch_size should be 1,when you train traking mode")
            batch = 1
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

    def get_validator(self):
        """Returns a DetectionValidator for RTDETR model validation."""
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return TrackValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess_batch(batch)
        bs = len(batch['img'])
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch


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
