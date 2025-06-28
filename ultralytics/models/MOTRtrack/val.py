# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, callbacks, colorstr, emojis
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
import json
from tqdm import tqdm
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data import YOLOTrackDataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import colorstr, ops, callbacks
from ultralytics.utils.hota import HOTA

__all__ = 'TrackValidator',  # tuple or list

from ultralytics.utils.checks import check_imgsz

from ultralytics.utils.torch_utils import smart_inference_mode


def filter_dt_by_score(dt_instances, prob_threshold):
    keep = dt_instances.scores > prob_threshold
    keep &= dt_instances.obj_idxes >= 0
    return dt_instances[keep]


def filter_dt_by_area(dt_instances, area_threshold):
    wh = dt_instances.boxes[..., 2:4] - dt_instances.boxes[..., 0:2]
    areas = wh[..., 0] * wh[..., 1]
    keep = areas > area_threshold
    return dt_instances[keep]


# TODO: Temporarily, RT-DETR does not need padding.
class TrackDataset(YOLOTrackDataset):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)

    # NOTE: add stretch version load_image for rtdetr mosaic
    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            # im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('imshow',im)
            # cv2.waitKey(0)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]



class TrackValidator(DetectionValidator):

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return TrackDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def postprocess(self, preds, track_instances):
        """Apply Non-maximum suppression to prediction outputs."""
        bs, _, nd = preds[0].shape

        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        scores = track_instances.pred_logits.unsqueeze(0)
        if len(scores.shape) != 3:
            scores = scores.unsqueeze(-1)
        bboxes = track_instances.pred_boxes.unsqueeze(0)

        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs

        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here.
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # sort by confidence to correctly get internal metrics.
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        return outputs

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):

            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]

            # id = torch.tensor(batch['track_id'],device=idx.device)
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            predn[..., [0, 2]] *= shape[1] / self.args.imgsz  # native-space pred
            predn[..., [1, 3]] *= shape[0] / self.args.imgsz  # native-space pred

            # Evaluate
            if nl:
                tbox = ops.xywh2xyxy(bbox)  # target boxes
                tbox[..., [0, 2]] *= shape[1]  # native-space pred
                tbox[..., [1, 3]] *= shape[0]  # native-space pred
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                # NOTE: To get correct metrics, the inputs of `_process_batch` should always be float32 type.
                correct_bboxes = self._process_batch(predn.float(), labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def update_track_metrics(self, preds, batch):
        """Metrics."""
        prob_threshold = 0.5
        area_threshold = 60

    #  没错这个八成也要重写
    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            self.args.half = self.device.type != 'cpu'  # force FP16 val during training

            model = model.half() if self.args.half else model.float()
            if self.device.type != 'cpu':
                model = model.to(self.device)
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items_one_frame, device=trainer.device)
            self.args.plots = trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks('on_val_start')
            assert model is not None, 'Either trainer or model is needed for validation'
            model = AutoBackend(model,
                                device=select_device(self.args.device, self.args.batch),
                                dnn=self.args.dnn,
                                data=self.args.data,
                                fp16=self.args.half)
            self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

            if isinstance(self.args.data, str) and self.args.data.endswith('.yaml'):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(
                    emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type == 'cpu':
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split),
                                                                     self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each vals
        label_dict = {label['im_file'][-15:-4]: label['track_id'] for label in self.dataloader.dataset.labels}
        frame_count = 0
        is_first = True
        track_truth_datas = {}
        track_pred_datas = {}
        video_name = ""
        self.data_hota = {}
        hota = HOTA()

        num_tracker_dets = 0
        num_gt_dets = 0
        unique_gt_ids = []
        unique_tracker_ids = []
        self.data_hota['num_timesteps'] = 0

        is_skip = False
        for batch_i, batch in enumerate(bar):
            # print(batch)
            self.run_callbacks('on_val_batch_start')
            frame_count += 1
            img_name = batch['im_file'][0][-15:-4]
            if batch_i <= 1:
                video_name = img_name[0:4]
            video_name_now = img_name[0:4]
            # 这段是为了减小计算量
            if is_first:
                import random
                is_skip_num = random.random() * 2
                if is_skip_num < 0.6:
                    is_skip = True
                    # Backward,没错是

            is_first = False

            self.batch_i = batch_i
            batch['track_id'] = label_dict.get(img_name, None)
            try:
                if video_name_now != video_name and batch_i >= 2:
                    video_name = video_name_now
                    is_first = True
                    frame_count = 0
                    self.data_hota['num_tracker_dets'] = num_tracker_dets
                    self.data_hota['num_gt_dets'] = num_gt_dets

                    self.data_hota['num_tracker_ids'] = len(unique_tracker_ids)
                    self.data_hota['num_gt_ids'] = len(unique_gt_ids)

                    # Calculate similarities for each timestep.
                    similarity_scores = []
                    for t, (gt_dets_t, tracker_dets_t) in enumerate(
                            zip(self.data_hota['gt_dets'], self.data_hota['tracker_dets'])):
                        try:
                            ious = self._calculate_hota_similarities(gt_dets_t, tracker_dets_t)
                        except:
                            tracker_dets_t = np.array([[0, 0, 1, 1]])
                            ious = self._calculate_hota_similarities(gt_dets_t, tracker_dets_t)
                        similarity_scores.append(ious)
                    self.data_hota['similarity_scores'] = similarity_scores

                    tracking_hota = hota.eval_sequence(data=self.data_hota)  # emmmm先不要
                    # 记得取消注释
                    # print('')
                    # print(video_name + ':')
                    # print('HOTA:', tracking_hota)
                    # print('')
                    # print('其他一些完整数据', tracking_hota)
                    self.data_hota = {}
                    track_truth_datas[frame_count] = batch
                    track_pred_datas[frame_count] = track_instances
                    num_tracker_dets = 0
                    num_gt_dets = 0
                    unique_gt_ids = []
                    unique_tracker_ids = []
                    self.data_hota['num_timesteps'] = 0
            except:
                # print('HOTA计算还是有问题')
                video_name = video_name_now
                is_first = True

            # if video_name_now != video_name and batch_i >= 2:
            #
            #     is_first = True
            #     frame_count = 0
            #     self.data_hota['num_tracker_dets'] = num_tracker_dets
            #     self.data_hota['num_gt_dets'] = num_gt_dets
            #
            #     self.data_hota['num_tracker_ids'] = len(unique_tracker_ids)
            #     self.data_hota['num_gt_ids'] = len(unique_gt_ids)
            #
            #     # Calculate similarities for each timestep.
            #     similarity_scores = []
            #     for t, (gt_dets_t, tracker_dets_t) in enumerate(
            #             zip(self.data_hota['gt_dets'], self.data_hota['tracker_dets'])):
            #         try:
            #             ious = self._calculate_hota_similarities(gt_dets_t, tracker_dets_t)
            #         except:
            #             tracker_dets_t = np.array([[0,0,1,1]])
            #             ious = self._calculate_hota_similarities(gt_dets_t, tracker_dets_t)
            #         similarity_scores.append(ious)
            #     self.data_hota['similarity_scores'] = similarity_scores
            #
            #     tracking_hota = hota.eval_sequence(data=self.data_hota)  # emmmm先不要
            #     print('')
            #     print(video_name + ':')
            #     print('HOTA:', tracking_hota)
            #     print('')
            #     print('其他一些完整数据', tracking_hota)
            #
            #     video_name = video_name_now
            #     self.data_hota = {}
            #     track_truth_datas[frame_count] = batch
            #     track_pred_datas[frame_count] = track_instances
            #     num_tracker_dets = 0
            #     num_gt_dets = 0
            #     unique_gt_ids = []
            #     unique_tracker_ids = []
            #     self.data_hota['num_timesteps'] = 0

            # if is_skip:
            #     continue

            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                # preds = model(batch['img'], augment=self.args.augment) preds, track_instances = model(batch['img'],
                # is_first=is_first, batch=batch) if preds is None else preds

                preds, track_instances = model(batch['img'], is_first=is_first)
                try:
                    track_instances = track_instances[track_instances.obj_idxes >= 0]
                    obj_idxes = track_instances.obj_idxes
                    scores = track_instances.scores
                    pred_boxes = track_instances.pred_boxes
                except:
                    obj_idxes = torch.tensor([])
                    scores = torch.empty((1, 1))
                    pred_boxes = torch.tensor([])

            # Loss
            with dt[2]:
                if self.training:
                    loss_one_frame, loss_items_one_frame, num_object = model.loss(batch, preds)
                    self.loss += (loss_items_one_frame / (num_object + 1))

            # Postprocessor
            with dt[3]:
                preds = self.postprocess(preds, track_instances)

            # 处理跟踪数据
            bbox = batch['bboxes']
            shape = batch['ori_shape'][0]
            tbox = ops.xywh2xyxy(bbox)  # target boxes
            tbox[..., [0, 2]] *= shape[1]  # native-space pred
            tbox[..., [1, 3]] *= shape[0]  # native-space pred
            self.data_hota['num_timesteps'] += 1

            for si, pred in enumerate(preds):
                shape = batch['ori_shape'][si]
                predn = pred.clone()
                predn[..., [0, 2]] *= shape[1] / self.args.imgsz  # native-space pred
                predn[..., [1, 3]] *= shape[0] / self.args.imgsz  # native-space pred

                pbox = predn[..., [0, 1, 2, 3]]

            try:
                self.data_hota['gt_ids'].append(batch['track_id'].astype(int))
                self.data_hota['tracker_ids'].append(obj_idxes.cpu().numpy().astype(int))
                self.data_hota['tracker_dets'].append(pred_boxes.cpu().numpy())
                self.data_hota['gt_dets'].append(tbox.cpu().numpy())
            except:
                self.data_hota['gt_ids'] = []
                self.data_hota['tracker_ids'] = []
                self.data_hota['gt_dets'] = []
                self.data_hota['tracker_dets'] = []

                self.data_hota['tracker_ids'].append(obj_idxes.cpu().numpy().astype(int))
                self.data_hota['gt_ids'].append(batch['track_id'].astype(int))
                self.data_hota['tracker_dets'].append(pred_boxes.cpu().numpy())
                self.data_hota['gt_dets'].append(tbox.cpu().numpy())

            num_tracker_dets += len(self.data_hota['tracker_ids'][-1])

            num_gt_dets += len(self.data_hota['gt_ids'][-1])
            unique_gt_ids = list(unique_gt_ids)
            try:
                unique_gt_ids += (self.data_hota['gt_ids'][-1].T.tolist()[0])
            except:
                unique_gt_ids = unique_gt_ids

            unique_gt_ids = np.array(unique_gt_ids)
            unique_gt_ids = list(np.unique(unique_gt_ids))

            unique_tracker_ids = list(unique_tracker_ids)

            unique_tracker_ids += list(np.unique(self.data_hota['tracker_ids'][-1]))

            if len(unique_gt_ids) > 0:
                # 将Python列表转换为NumPy数组
                unique_gt_ids = np.array(unique_gt_ids)
                unique_gt_ids = unique_gt_ids.astype(int)
                unique_gt_ids = list(unique_gt_ids)
                unique_gt_ids = np.unique(unique_gt_ids)
                gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
                gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
                # for t in range(self.data_hota['num_timesteps']):
                #     if len(self.data_hota['gt_ids'][t]) > 0:
                #         self.data_hota['gt_ids'][t] = gt_id_map[self.data_hota['gt_ids'][t]].astype(np.int)

            if len(unique_tracker_ids) > 0 and (-1 not in unique_tracker_ids):
                unique_tracker_ids = np.array(unique_tracker_ids)
                unique_tracker_ids = unique_tracker_ids.astype(int)
                unique_tracker_ids = list(unique_tracker_ids)
                unique_tracker_ids = np.unique(unique_tracker_ids)

                unique_tracker_ids = np.unique(unique_tracker_ids)
                tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
                # print(unique_tracker_ids)
                tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
                # for t in range(self.data_hota['num_timesteps']):
                #     if len(self.data_hota['tracker_ids'][t]) > 0:
                #         self.data_hota['tracker_ids'][t] = tracker_id_map[self.data_hota['tracker_ids'][t]].astype(np.int)

            track_pred_datas[frame_count] = track_instances

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            results = {**stats,
                       **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            return {k: round(float(v), 5) for k, v in
                    results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                'Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def _calculate_hota_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
        return similarity_scores

    @staticmethod
    def _calculate_box_ious(bboxes1, bboxes2, box_format='xywh', do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of boxes.
        Allows variable box formats ('xywh' and 'x0y0x1y1').
        If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
        used to determine if detections are within crowd ignore region.
        """
        from copy import deepcopy
        if box_format in 'xywh':
            # layout: (x0, y0, w, h)
            bboxes1 = deepcopy(bboxes1)
            bboxes2 = deepcopy(bboxes2)

            bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
            bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
            bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
            bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
        elif box_format not in 'x0y0x1y1':
            raise (NotImplementedError('box_format %s is not implemented' % box_format))

        # layout: (x0, y0, x1, y1)
        min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

        if do_ioa:
            ioas = np.zeros_like(intersection)
            valid_mask = area1 > 0 + np.finfo('float').eps
            ioas[valid_mask, :] = intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]

            return ioas
        else:
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
            intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
            intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
            intersection[union <= 0 + np.finfo('float').eps] = 0
            union[union <= 0 + np.finfo('float').eps] = 1
            ious = intersection / union
            return ious
