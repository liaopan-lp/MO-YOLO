# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math
from ultralytics.utils import ops
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, MOTRTransformerDecoder, \
    MOTRDecoderLayer, pos2posemb
from .utils import bias_init_with_prob, linear_init_
from ultralytics.utils.ops import HungarianMatcherGroup, HungarianMatcher
from MOTR.models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

# from MOTR.models.deformable_transformer_plus import *

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder', 'MOTRTrack'


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class MOTRTrack(nn.Module):
    """YOLOv8 Track head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), d_model=256, aux_loss=False, nq=300):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.aux_loss = aux_loss
        self.is_first = True
        from MOTR.models.qim import build as build_query_interaction_layer
        from MOTR.main import get_args_parser
        import argparse
        parser = argparse.ArgumentParser('MOTR training and evaluation script', parents=[get_args_parser()])
        args = parser.parse_args()
        self.nq = nq
        self.decoder = MYDecoder(nc=nc, ch=ch, nq=self.nq)

        self.is_first = True

        self.track_embed = build_query_interaction_layer(args, args.query_interaction_layer, d_model,
                                                         self.decoder.hidden_dim, d_model * 2)
        self.mem_bank_len = 0  # if memory_bank is None else memory_bank.max_his_length

        self.track_base = RuntimeTrackerBase(training=self.training)
        if self.training:
            self.matcher = HungarianMatcherGroup()  #
        else:
            self.matcher = None
        # parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[
        # motr_args_parser()]) args = parser.parse_args() transfomers=build_deforamble_transformer(args)

        # decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation,
        # num_feature_levels, nhead, dec_n_points, decoder_self_cross, sigmoid_attn=sigmoid_attn,
        # extra_track_attn=extra_track_attn)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # self.query_embed = nn.Embedding(self.decoder.num_queries, self.decoder.hidden_dim * 2, device=device)
        # self.query_embed = nn.Embedding(self.decoder.num_queries, self.decoder.hidden_dim)

        # 这个是跟踪信息
        self.track_instances = None
        self.track_instances_pre = None

        # if self.track_instances is None:
        #     self.track_instances = self._generate_empty_tracks()

        # 没啥用,之后可能有用，先none
        self.memory_bank = None

    def _generate_empty_tracks(self, len_before=0):
        from MOTR.models.structures import Instances
        track_instances = Instances((1, 1))
        num_queries, dim = self.decoder.num_queries, self.decoder.hidden_dim * 2
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # device = self.query_embed.weight.device
        # self.decoder.reference_points.to(device)

        # track_instances.ref_pts = self.decoder.reference_points(self.query_embed.weight[:, :dim // 2])
        track_instances.ref_pts = torch.rand(num_queries, 4, device=device).to(device)

        # self.query_embed.weight.to(device)
        # track_instances.query_pos = self.query_embed.weight
        track_instances.query_pos = torch.zeros((num_queries, 256), dtype=torch.float, device=device)

        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)

        # track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device) #这个可能是id？
        track_instances.obj_idxes = torch.full((len(track_instances), 1), -1, dtype=torch.long,
                                               device=device)  # ID
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), 1), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.nc), dtype=torch.float,
                                                  device=device)

        # mem_bank_len = self.mem_bank_len
        # track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32,
        #                                        device=device)
        # track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool,
        #                                               device=device)
        # track_instances.save_period = torch.zeros((len(track_instances),), dtype=torch.float32, device=device)

        return track_instances.to(device)

    def forward(self, x, batch=None, is_first=True):
        """Concatenates and returns predicted bounding boxes and class probabilities."""

        shape = x[0].shape  # BCHW
        # transformer_decoder = self.decoder  # 替换为你的 Transformer Decoder 类
        # for i in range(self.nl):
        #     x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.is_first or self.track_instances is None:
            self.track_instances = self._generate_empty_tracks()
            self.track_base = RuntimeTrackerBase(training=self.training)
            self.track_base.clear()
            ref_pts = None
            pre_class = None
            track_query_pos = None
        else:
            # if self.training:
            #     tmp = {'init_track_instances': self._generate_empty_tracks(),
            #            'track_instances': self.track_instances}
            #
            #     self.track_instances = self.track_embed(tmp)

            # active_track_instances = self.track_instances[:len(self.track_instances) - self.nq]
            active_track_instances = self.track_instances
            ref_pts = active_track_instances.ref_pts
            pre_class = active_track_instances.pred_logits
            track_query_pos = active_track_instances.query_pos
            if len(active_track_instances) <= 0:
                ref_pts = None
                pre_class = None
                track_query_pos = None

        [dec_bboxes, dec_scores, enc_bbox, enc_outputs_class, dn_meta, init_reference,
         dec_output_embeding] = self.decoder(x,
                                             track_query_pos=track_query_pos,
                                             track_ref_pts=ref_pts,
                                             batch=batch,
                                             is_first=self.is_first,
                                             pre_class=pre_class)

        x = dec_bboxes, dec_scores, enc_bbox, enc_outputs_class, dn_meta, init_reference, dec_output_embeding
        match_indices = self._update_track_instances(x, is_first=self.is_first, batch=batch)
        if self.training:
            return x, self.track_instances, self.nq, match_indices
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        if self.export:
            return y, self.track_instances
        else:
            return (y, x), self.track_instances

    def _update_track_instances(self, out, is_first=False, batch=None):

        dec_bboxes, dec_scores, enc_bbox, enc_outputs_class, enc_outputs_coord_unact, init_reference, dec_output_embeding = out
        if enc_outputs_coord_unact is not None:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, enc_outputs_coord_unact['dn_num_split'], dim=2)
            _, init_reference = torch.split(init_reference, enc_outputs_coord_unact['dn_num_split'], dim=1)
            dn_scores, dec_scores = torch.split(dec_scores, enc_outputs_coord_unact['dn_num_split'], dim=2)
        else:
            dec_bboxes = dec_bboxes
            init_reference = init_reference
            dec_scores = dec_scores

        # dec_scores
        # torch.Size([6, 1, 288, 5]) KITTI
        # torch.Size([6, 1, 91, 1]) MOT

        outputs_classes = []
        outputs_coords = []
        for lvl in range(dec_bboxes.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = dec_bboxes[lvl - 1]
            from MOTR.util.misc import inverse_sigmoid
            reference = inverse_sigmoid(reference)
            # outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = dec_bboxes[lvl]

            # if reference.shape[-1] == 4:
            #     tmp = tmp + reference
            # else:
            #     assert reference.shape[-1] == 2
            #     new_tmp = tmp[..., :2] + reference
            #     tmp[..., :2] = new_tmp

            outputs_coord = tmp
            # outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = dec_scores
        outputs_coord = torch.stack(outputs_coords)
        #        [[[0.6640, 0.5385, 0.5343, 0.5781],

        if not self.training:
            init_reference = init_reference.clone().to(dec_bboxes.device)

        ref_pts_all = torch.cat([init_reference[None], dec_bboxes[:, :, :, :4]], dim=0).to(outputs_class.device)

        # outputs_class[-1]
        # torch.Size([1, 114, 1]) MOT17
        # torch.Size([1, 101, 5]) KITTI

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'ref_pts': ref_pts_all[-1],
               'hs': dec_output_embeding[-1]}
        frame_res, indices, unmatched_track_idxes = self._post_process_single_image(out, self.track_instances,
                                                                                    batch=batch)

        self.track_instances = frame_res['track_instances']
        return [indices, unmatched_track_idxes]

    def _post_process_single_image(self, frame_res, track_instances, batch=None):
        indices = None
        with torch.no_grad():
            # torch.Size([60, 5]) KITTI
            # torch.Size([116, 1]) MOT
            if self.training:
                # print('训练时：', frame_res['pred_logits'].shape)
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                # print('推理时：', frame_res['pred_logits'].shape)
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values

        # 这个可以先不用管
        track_instances.scores = track_scores

        track_instances.pred_logits = frame_res['pred_logits'][0]

        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        # 我不要它的loss（因为不太好改），我这里只要id
        track_instances.output_embedding = frame_res['hs']
        matched_indices = None
        unmatched_track_idxes = None
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        if batch is not None and track_instances is not None:
            indices = None
            frame_res['track_instances'] = track_instances

            pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
            pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.
            track_instances.matched_gt_idxes[...] = -1
            if len(batch['track_id']) == 0:
                # tmp = {'init_track_instances': self._generate_empty_tracks(),
                #        'track_instances': track_instances}

                # out_track_instances = self.track_embed(tmp)
                frame_res['track_instances'] = track_instances
                return frame_res, None, None

            if not (track_instances.obj_idxes != -1).any():  # 没有跟踪
                indices = self.matcher(frame_res["pred_boxes"], frame_res["pred_logits"], batch['bboxes'], batch['cls'],
                                       batch['gt_groups'])

                indices = [(ind[0].to(pred_logits_i.device), ind[1].to(pred_logits_i.device)) for ind in indices]

                for i, ind in enumerate(indices):
                    track_instances.matched_gt_idxes[ind[0]] = ind[1]

                    track_instances.obj_idxes[ind[0]] = batch['track_id'][ind[1]].long()
                    active_idxes = torch.logical_and(track_instances.obj_idxes[:, 0] >= 0,
                                                     track_instances.matched_gt_idxes >= 0)
                    active_track_instances = track_instances[active_idxes]
                    active_track_boxes = active_track_instances.pred_boxes
                    active_idxes = track_instances.matched_gt_idxes >= 0

                    if len(active_track_boxes) > 0:
                        gt_boxes = batch['bboxes'][
                            track_instances.matched_gt_idxes[track_instances.matched_gt_idxes >= 0]]

                        # track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes),
                        #                                                         Boxes(gt_boxes))
                        gt_bboxes = batch['bboxes']
                        true_indices = torch.nonzero(active_idxes, as_tuple=False)
                        for i in true_indices:
                            active_track_box = track_instances.pred_boxes[i]
                            # 转换为xyxy类型
                            active_track_box_xyxy = ops.xywh2xyxy(active_track_box)
                            gt_bboxes_xyxy = ops.xywh2xyxy(gt_bboxes)
                            # 计算交集和并集
                            try:
                                intersection = torch.min(active_track_box_xyxy, gt_bboxes_xyxy)
                            except:
                                intersection = 0

                            # 找到最大的 IoU
                            union = torch.max(active_track_box_xyxy, gt_bboxes_xyxy)
                            # 计算 IoU
                            iou = intersection.sum(dim=1) / union.sum(dim=1)
                            max_iou = iou.max()
                            track_instances.iou[i] = max_iou
                            # try:
                            #     union = torch.max(active_track_box_xyxy, gt_bboxes_xyxy)
                            #     # 计算 IoU
                            #     iou = intersection.sum(dim=1) / union.sum(dim=1)
                            #     max_iou = iou.max()
                            #     track_instances.iou[i] = max_iou
                            # except:
                            #     track_instances.iou[i] = 0
                        else:
                            active_track_boxes = None
            else:
                active_idxes = (track_instances.obj_idxes >= 0).squeeze()
                gt_bboxes = batch['bboxes']
                gt_obj_idxes = batch['track_id']
                # 将两个张量展平，以便进行比较
                track_indices_flat = track_instances.obj_idxes.view(-1)
                gt_indices_flat = gt_obj_idxes.view(-1)

                # 找到两个张量中相同元素的索引
                matching_indices = torch.nonzero(track_indices_flat[:, None] == gt_indices_flat)
                # 如果需要分别得到索引，可以使用以下代码
                i, j = matching_indices[:, 0], matching_indices[:, 1]
                track_instances.matched_gt_idxes[i] = j

                full_track_idxes = torch.arange(len(track_instances.pred_logits)/2, dtype=torch.long,
                                                device=pred_logits_i.device)

                matched_track_idxes = (track_indices_flat >= 0)  # occu >=0表明该query为跟踪query
                prev_matched_indices = torch.stack(
                    [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]],
                    dim=1)  # 检测或跟踪与gt的对应关系

                # step2. select the unmatched slots.
                # note that the FP tracks whose obj_idxes are -2 will not be selected here.
                unmatched_track_idxes = full_track_idxes[track_indices_flat == -1]  # 获取检测query


                # step3. select the untracked gt instances (new tracks).
                tgt_indexes = track_instances.matched_gt_idxes
                tgt_indexes = tgt_indexes[tgt_indexes != -1]  # 获取跟踪query匹配GT，非新生儿（除了这些之外便是新生儿）

                tgt_state = torch.zeros(len(gt_indices_flat), device=pred_logits_i.device)
                tgt_state[tgt_indexes] = 1

                full_tgt_idxes = torch.arange(len(gt_indices_flat), device=pred_logits_i.device)
                untracked_tgt_indexes = full_tgt_idxes[tgt_state == 0]

                unmatched_tgt = {
                    'pred_logits': batch['cls'][untracked_tgt_indexes],
                    'pred_boxes': batch['bboxes'][untracked_tgt_indexes],
                    'track_id': batch['track_id'][untracked_tgt_indexes],

                }

                # step4. do matching between the unmatched slots and GTs.该过程就是DET匈牙利匹配过程

                unmatched_outputs = {
                    'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
                }

                new_track_indices = self.matcher(unmatched_outputs["pred_boxes"], unmatched_outputs["pred_logits"],
                                                 unmatched_tgt['pred_boxes'], unmatched_tgt['pred_logits'],
                                                 [len(untracked_tgt_indexes)])

                # indices = self.matcher(frame_res["pred_boxes"], frame_res["pred_logits"], batch['bboxes'], batch['cls'],
                #                        batch['gt_groups'])

                src_idx = new_track_indices[0][0]
                tgt_idx = new_track_indices[0][1]
                # concat src and tgt.
                new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                                  dim=1).to(pred_logits_i.device)
                # step5. update obj_idxes according to the new matching result. 分配GT的ID给track和GT所在的索引
                track_instances.obj_idxes[new_matched_indices[:, 0]] = batch['track_id'][
                    new_matched_indices[:, 1]].long()

                # 调整 new_matched_indices 的大小，使其匹配 prev_matched_indices,主要是RT-DETR的loss的匹配顺序和好像MOTR的不一样

                # new_matched_indices_adjusted = new_matched_indices.expand_as(prev_matched_indices)

                matched_indices_ = torch.cat([new_matched_indices, prev_matched_indices], dim=0)
                matched_indices = [(matched_indices_[:, 0], matched_indices_[:, 1])]
                track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]
                # 获取True值的索引
                true_indices = torch.nonzero(active_idxes, as_tuple=False)
                if len(true_indices) > 0:
                    for i in true_indices:
                        active_track_box = track_instances.pred_boxes[i]
                        # 计算交集和并集
                        try:
                            intersection = torch.min(active_track_box, gt_bboxes)
                        except:
                            intersection = 0

                            # 找到最大的 IoU
                            union = torch.max(active_track_box, gt_bboxes)
                            # 计算 IoU
                            iou = intersection.sum(dim=1) / union.sum(dim=1)
                            max_iou = iou.max()
                            track_instances.iou[i] = max_iou
                        # try:
                        #     union = torch.max(active_track_box, gt_bboxes)
                        #     # 计算 IoU
                        #     iou = intersection.sum(dim=1) / union.sum(dim=1)
                        #     max_iou = iou.max()
                        #     track_instances.iou[i] = max_iou
                        # except:
                        #     pass

                else:
                    active_track_boxes = None
        # if not self.training:
        track_instances = self.track_base.update(track_instances)
        tmp = {'detect_queries': track_instances,
               'track_queries': self.track_instances}
        out_track_instances = self.track_embed(tmp)
        frame_res['track_instances'] = out_track_instances
        # else:
        #     frame_res['track_instances'] = track_instances

        if self.training:
            return frame_res, matched_indices, unmatched_track_idxes

        return frame_res, None, None

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        from ultralytics.models.utils.ops import get_cdn_group

        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)

        '''enc_bboxes.size() in head torch.Size([1, 300, 4])
            dec_bboxes.size() in head torch.Size([6, 1, 492, 4])
            enc_scores.size() in head torch.Size([1, 300, 5])
            dec_scores.size() in head torch.Size([6, 1, 492, 5])'''

        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features

        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = len(feats)

        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)
        # dynamic anchors + static content
        enc_outputs_bboxes = self.enc_bbox_head(features) + anchors  # (bs, h*w, 4)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # Unsigmoided
        refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # refer_bbox = torch.gather(enc_outputs_bboxes, 1, topk_ind.reshape(bs, self.num_queries).unsqueeze(-1).repeat(1, 1, 4))

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        if self.training:
            refer_bbox = refer_bbox.detach()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class MYDecoder(nn.Module):
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
    ):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # backbone feature projection
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version, but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(nn.Conv2d(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = MOTRDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = MOTRTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        # 线性变换
        # self.reference_points = nn.Linear(hd, 4)  # 这里可能是4？

        self._reset_parameters()

    def forward(self, x, track_ref_pts=None, batch=None, is_first=False, pre_class=None, track_query_pos=None):
        from ultralytics.models.utils.ops import get_track_cdn_group

        # for l, feat in enumerate(x):
        #     src, mask = feat.decompose()
        #     srcs.append(self.input_proj[l](src))
        #     masks.append(mask)
        #     assert mask is not None
        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        bs, _, c = feats.shape

        # prepare denoising training

        if pre_class is not None:
            pre_cls_outputs = [torch.zeros((0, 2), device=pre_class.device)] * pre_class.shape[0]
            clses = [torch.zeros((0, 1), device=pre_class.device)] * pre_class.shape[0]
            for i, pre_cls in enumerate(pre_class):
                score, cls = pre_class[i].max(-1)  # (300, )

                pred = torch.cat([score[..., None], cls[..., None]], dim=-1)  # filter
                pre_cls_outputs[i] = pred
                clses[i] = cls.item()
            track_cls_embed = self.denoising_class_embed.weight[clses]  # bs*num * 2 * num_group, 256

            track_cls_embed = track_cls_embed.unsqueeze(dim=0)
            num_track_queries = track_cls_embed.shape[1]

        else:
            track_cls_embed = None
            num_track_queries = 0

        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_track_cdn_group(batch,
                                self.nc,
                                self.num_queries,
                                self.denoising_class_embed.weight,
                                self.num_denoising,
                                self.label_noise_ratio,
                                self.box_noise_scale,
                                self.training,
                                num_track_queries)

        embed, refer_bbox, enc_bboxes, enc_scores, track_ref_pts, query_pos = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox, track_ref_pts, is_first=is_first,
                                    track_embed=track_cls_embed, pre_class=pre_class, track_query_pos=track_query_pos)

        track_ref_pts.to(enc_bboxes.device)
        # track_query_pos = pos2posemb(track_ref_pts)

        # track_query_pos = track_query_pos.unsqueeze(0)
        #
        # track_query_embed, _ = torch.split(track_query_pos, c, dim=1)
        #
        # track_query_embed = track_query_pos.expand(bs, -1, -1)

        if not self.training:
            refer_bbox = refer_bbox.to(embed.dtype)
        #     enc_bboxes = enc_bboxes.to(embed.dtype)
        #     enc_scores = enc_scores.to(embed.dtype)
        #     track_query_embed = track_query_embed.to(embed.dtype)
        #     track_ref_pts = track_ref_pts.to(embed.dtype)

        # decoder  额，我这的定义和motr的不一样啊，我这的参数具体来说应该和上面那个detr的是一样的才对
        dec_bboxes, dec_scores, dec_output_embeding = self.decoder(embed,
                                                                   refer_bbox,
                                                                   feats,
                                                                   shapes,
                                                                   self.dec_bbox_head,
                                                                   self.dec_score_head,
                                                                   self.query_pos_head,
                                                                   attn_mask=attn_mask,
                                                                   track_query_embed=query_pos
                                                                   )

        # x = hs, inter_references, enc_bboxes, enc_scores, dn_meta

        # if track_ref_pts is None:
        #     reference_points = self.reference_points(track_query_embed).sigmoid()
        # else:
        reference_points = track_ref_pts.repeat(bs, 1, 1).sigmoid()

        init_reference_out = reference_points
        dec_scores_out = dec_scores

        ''' hs.size() in head torch.Size([6, 1, 498, 4])
            init_reference_out.size() in head torch.Size([1, 300, 2])
            inter_references_out.size() in head torch.Size([6, 1, 498, 5])
            enc_scores.size() in head torch.Size([1, 300, 5])
            'dn_num_group': 33, 'dn_num_split': [198, 300]
            '''

        ''' enc_bboxes.size() in head torch.Size([1, 300, 4])
            dec_bboxes.size() in head torch.Size([6, 1, 492, 4])
            enc_scores.size() in head torch.Size([1, 300, 5])
            dec_scores.size() in head torch.Size([6, 1, 492, 5])

        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta'''
        # # 使用 torch.isnan 函数找到NaN值的索引
        # nan_indices = torch.isnan(dec_bboxes)
        #
        # # 使用 torch.where 函数将NaN值替换为0
        # dec_bboxes = torch.where(nan_indices, torch.tensor(0.0), dec_bboxes)
        #
        # # 使用 torch.isnan 函数找到NaN值的索引
        # nan_indices = torch.isnan(dec_scores_out)
        #
        # # 使用 torch.where 函数将NaN值替换为0
        # dec_scores_out = torch.where(nan_indices, torch.tensor(0.0), dec_scores_out)
        x = dec_bboxes, dec_scores_out, enc_bboxes, enc_scores, dn_meta, init_reference_out, dec_output_embeding

        return x

        # if self.training:
        #     return x
        # # (bs, 300, 4+nc)
        # y = torch.cat((dec_bboxes.squeeze(0), dec_scores_out.squeeze(0)), -1)
        # return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):

        # Now apply the input_proj to the resized x
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]

        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None, track_ref_pts=None, is_first=False,
                           track_embed=None, pre_class=None, track_query_pos=None):
        bs = len(feats)

        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        # anchors, valid_mask = self._generate_anchors(shapes, dtype=torch.float32, device=feats.device)

        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # dynamic anchors + static content
        enc_outputs_bboxes = self.enc_bbox_head(features) + anchors  # (bs, h*w, 4)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        if track_ref_pts is None or is_first:
            refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1).to(features.device)
            query_pos = pos2posemb(refer_bbox)
        else:
            refer_bbox = track_ref_pts.view(bs, track_ref_pts.shape[0], -1).to(features.device)
            refer_bbox_1 = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1).to(
                features.device)
            # 保证是self.num_queries
            # refer_bbox_1 = refer_bbox_1[:, 0:self.num_queries - track_ref_pts.shape[0], :]
            refer_bbox = torch.cat([refer_bbox, refer_bbox_1], dim=1)
            if track_query_pos.device != refer_bbox_1.device:
                track_query_pos = track_query_pos.to(refer_bbox_1.device)
            query_pos = torch.cat([track_query_pos.unsqueeze(0), pos2posemb(refer_bbox_1)], dim=1)

            # if track_ref_pts.shape[0] < self.num_queries:
            #     refer_bbox_1 = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1).to(
            #         features.device)
            #     # 保证是self.num_queries
            #     refer_bbox_1 = refer_bbox_1[:, 0:self.num_queries - track_ref_pts.shape[0], :]
            #     refer_bbox = torch.cat([refer_bbox, refer_bbox_1], dim=1)
            #
            # else:
            #     refer_bbox = refer_bbox[:, 0:self.num_queries, :]

        # refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        enc_bboxes = refer_bbox.sigmoid()

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)

            query_pos = torch.cat([pos2posemb(dn_bbox), query_pos], dim=1)

        track_ref_pts = refer_bbox

        if self.training:
            refer_bbox = refer_bbox.detach()
            track_ref_pts = track_ref_pts.detach()

        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        if pre_class is not None:

            if pre_class.ndim == 1:
                pre_class = pre_class.unsqueeze(-1)
            if pre_class.device != enc_scores.device:
                temp_pre_class = pre_class.detach().to(enc_scores.device)
                enc_scores = torch.cat([temp_pre_class.unsqueeze(0), enc_scores], 1)
            else:
                enc_scores = torch.cat([pre_class.unsqueeze(0), enc_scores], 1)
        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()

        if track_embed is not None:
            embeddings = torch.cat([track_embed, embeddings], 1)
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores, track_ref_pts, query_pos

    # TODO
    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


from MOTR.models.structures import Instances


class RuntimeTrackerBase(object):  # 实际为一个跟踪ID分配器

    # MOT17这里是5比较合适
    def __init__(self, score_thresh=0.4, filter_score_thresh=0.5, miss_tolerance=5, training=False):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.max_obj_id_pre = 0
        self.training = training
        self.prev_track_ids = []

    def _filter_tracks(self, instances: Instances):
        num_boxes = instances.pred_boxes.shape[0]
        pred_boxes = instances.pred_boxes.clone().cpu()
        keep = torch.ones(num_boxes, dtype=torch.bool)
        for i in range(0, num_boxes):
            if keep[i]:
                for j in range(i + 1, num_boxes):
                    # if keep[j] and (instances.obj_idxes[j] not in self.prev_track_ids):
                    # if keep[j] and instances.obj_idxes[j] > self.max_obj_id_pre:
                    if keep[j]:
                        # iou = self._calculate_iou(pred_boxes, pred_boxes)
                        iou = self._calculate_iou(pred_boxes[i],
                                                  pred_boxes[j])
                        if iou > 0.8:
                            keep[j] = torch.BoolTensor([False])

        return keep

    def _calculate_iou(self, box1, box2):
        # 减小计算量
        if abs(box1[0] - box2[0]) > 0.5 * min(box1[0], box2[0]):
            return 0
        if abs(box1[1] - box2[1]) > 0.5 * min(box1[1], box2[1]):
            return 0
        # 计算两个边界框的交集
        intersection_x1 = torch.max(box1[0], box2[0])
        intersection_y1 = torch.max(box1[1], box2[1])
        intersection_x2 = torch.min(box1[0] + box1[2], box2[0] + box2[2])
        intersection_y2 = torch.min(box1[1] + box1[3], box2[1] + box2[3])

        # 计算交集区域的面积
        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0,
                                                                            intersection_y2 - intersection_y1)

        # 计算两个边界框的面积
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        # 计算IoU值
        iou = intersection_area / (box1_area + box2_area - intersection_area)

        return iou

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, g_size=1):
        try:
            assert track_instances.obj_idxes.shape[1] == 1
        except:
            track_instances.obj_idxes = torch.unsqueeze(track_instances.obj_idxes, 1)

        device = track_instances.scores.device
        track_instances.scores = track_instances.scores.detach()
        track_instances.obj_idxes = track_instances.obj_idxes.to(device).detach()
        num_queries = len(track_instances)
        # if self.training:
        #     active_idxes = track_instances.obj_idxes >= 0
        #     if torch.any(active_idxes).item():
        #         active_track_instances = track_instances[active_idxes]
        #     else:
        #         return track_instances
        #     import time
        #     start_time = time.time()
        #     if active_track_instances is not None:
        #         keep_mask = self._filter_tracks(active_track_instances)
        #         if len(keep_mask) != 0:
        #             try:
        #                 active_track_instances = active_track_instances[keep_mask]
        #             except:
        #                 # print(active_track_instances)
        #                 active_track_instances = active_track_instances
        #
        #     end_time = time.time()
        #     # print("耗时: {:.2f}ms".format((end_time - start_time) * 1000))
        #     tmp_num = 0
        #     return active_track_instances
        for i in range(len(track_instances.scores)):
            # print(track_instances.obj_idxes[i])
            # print(track_instances.scores[i])
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1

        active_track_idxes = track_instances.obj_idxes >= 0
        if any(active_track_idxes):
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]
        else:
            active_track_instances = track_instances
            return active_track_instances

        # active_track_instances = track_instances[track_instances.obj_idxes >= 0]
        import time
        start_time = time.time()
        if active_track_instances is not None:
            keep_mask = self._filter_tracks(active_track_instances)
            if len(keep_mask) != 0:
                try:
                    active_track_instances = active_track_instances[keep_mask]
                except:
                    active_track_instances = active_track_instances


        end_time = time.time()
        # print("过滤耗时: {:.2f}ms".format((end_time - start_time) * 1000))
        tmp_num = 0

        try:
            for i in range(len(active_track_instances.obj_idxes)):
                if active_track_instances.obj_idxes[i] > (self.max_obj_id_pre):
                    # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                    active_track_instances.obj_idxes[i] = self.max_obj_id_pre + tmp_num + 1
                    tmp_num += 1
        except:
            pass

        if active_track_instances is None:
            self.max_obj_id_pre = self.max_obj_id
        else:
            self.max_obj_id = max(active_track_instances.obj_idxes.cpu()) + 1
            self.max_obj_id_pre = self.max_obj_id - 1
            # self.prev_track_ids = active_track_instances.obj_idxes.clone().detach().cpu()
        return active_track_instances
