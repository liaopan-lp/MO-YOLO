# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, List

from MOTR.util import box_ops
from MOTR.util.misc import inverse_sigmoid
from MOTR.models.structures import Boxes, Instances, pairwise_iou
import math
from .fsqm import FSQM

def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances





class QueryInteractionBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()
        # 从参数中获取FSQM配置
        self.fsqm = FSQM(
            max_num_queries=300,
            feature_dim=dim_in,
            in_threshold=0.7,
            out_threshold=0.3
        )

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = 0.1
        self.fp_ratio = 0.1
        self.update_query_pos = args.update_query_pos
        self.iou_threshold = 0.6
        self.prev_track_instances_len = 0
        self.prev_track_ids = []

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def pos2posemb(self, pos, num_pos_feats=64, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=pos.dtype, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        posemb = pos[..., None] / dim_t
        posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
        return posemb

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
        try:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]
        except:
            # print(track_instances.obj_idxes)
            inactive_instances = []

        # add fp for each active track in a specific probability.
        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        try:
            # Use torch.multinomial to sample indices based on fp_prob
            selected_indices = torch.multinomial(fp_prob, len(active_track_instances), replacement=True)
        except:
            return active_track_instances
        selected_active_track_instances = active_track_instances[selected_indices]

        # fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        # selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]
                fp_track_instances.obj_idxes = fp_track_instances.obj_idxes.unsqueeze(1)
            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            # print(track_instances.iou.shape)
            active_idxes = (track_instances.obj_idxes[:, 0] >= 0) & (track_instances.iou > 0.5)


            active_track_instances = track_instances[active_idxes]
            active_track_instances.obj_idxes[active_track_instances.iou< 0.5] = -1

            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            try:
                active_idxes = (track_instances.obj_idxes[:, 0] >= 0)

                active_track_instances = track_instances[active_idxes]
            except:
                active_track_instances = None

        # print("active_track_instances",active_track_instances)
        # import time
        # start_time = time.time()
        # if active_track_instances is not None and self.prev_track_instances_len != 0:
        #     keep_mask = self._filter_tracks(active_track_instances)
        #     if len(keep_mask) != 0:
        #         active_track_instances = active_track_instances[keep_mask]
        #
        # end_time = time.time()
        # print("耗时: {:.2f}ms".format((end_time - start_time) * 1000))
        # if active_track_instances is None:
        #     self.prev_track_instances_len = 0
        #     self.prev_track_ids = []
        # else:
        #     self.prev_track_instances_len = len(active_track_instances)
        #     self.prev_track_ids = active_track_instances.obj_idxes.clone().detach()
        return active_track_instances

    def _filter_tracks(self, instances: Instances):
        num_boxes = instances.pred_boxes.shape[0]

        keep = torch.ones(num_boxes, dtype=torch.bool)
        for i in range(0, num_boxes):
            if keep[i]:
                for j in range(i + 1, num_boxes):
                    # if keep[j] and (instances.obj_idxes[j] not in self.prev_track_ids):
                    if keep[j]:
                        # iou = self._calculate_iou(instances.pred_boxes[i].cpu(), instances.pred_boxes[j].cpu())
                        iou = self._calculate_iou(instances.pred_boxes[i].clone().cpu(),
                                                  instances.pred_boxes[j].clone().cpu())
                        if iou > 0.75:
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

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        # 我这应该更新下ref_pts就行
        query_pos = self.pos2posemb(track_instances.ref_pts)
        # print('track_instances.ref_pts', track_instances.ref_pts.dtype)
        # print('query_pos', query_pos.dtype)
        dim = query_pos.shape[1]
        out_embed = track_instances.output_embedding
        if query_pos.dtype != out_embed.dtype:
            query_pos = query_pos.to(out_embed.dtype)
        # print('out_embed', out_embed.dtype)
        # query_pos = track_instances.query_pos[:, :dim // 2]
        # query_feat = track_instances.query_pos[:, dim // 2:]
        query_feat = track_instances.query_pos

        if query_pos.device != out_embed.device:
            query_pos = query_pos.to(out_embed.device)
        q = k = query_pos + out_embed

        tgt = out_embed
        # print(q.dtype)
        # print(k.dtype)
        # print(tgt.dtype)
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))

        if query_feat.dtype != query_feat2.dtype:
            query_feat = query_feat.to(query_feat2.dtype)
        try:
            query_feat = query_feat + self.dropout_feat2(query_feat2)
        except:
            query_feat = query_pos + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :4].detach().clone())
        return track_instances

    def forward(self, data) -> Instances:
        # try:
        #     active_track_instances = self._select_active_tracks(data)
        #     # 注意下啊，我这应该是用不到才对
        #     active_track_instances = self._update_track_embedding(active_track_instances)
        #     init_track_instances: Instances = data['init_track_instances']\

        #     # print(active_track_instances.get_fields())
        #     merged_track_instances = Instances.cat([active_track_instances, init_track_instances])
        # except:
        #     merged_track_instances = data['init_track_instances']
        detect_queries = data['detect_queries']
        track_queries = data['track_queries']

        # 使用FSQM进行查询内存管理
        updated_queries = self.fsqm.online_update(detect_queries, track_queries)

        # if self.training:
        #     active_track_instances = self._select_active_tracks(data)
        # else:
        #     active_track_instances = data['track_instances']
        # init_track_instances: Instances = data['init_track_instances']
        # # print(active_track_instances)
        # if updated_queries is not None:
        #     active_track_instances = self._update_track_embedding(updated_queries)
        #     # print(active_track_instances.get_fields())
        #     # init_track_instances.query_pos = init_track_instances.query_pos.unsqueeze(0)
        #     merged_track_instances = Instances.cat([active_track_instances, init_track_instances])
        #     # try:
        #     #     merged_track_instances = Instances.cat([active_track_instances, init_track_instances])
        #     # except:
        #     #     # print(active_track_instances)
        #     #     merged_track_instances = init_track_instances
        # else:
        #     merged_track_instances = init_track_instances
        # # merged_track_instances = active_track_instances
        # merged_track_instances.query_pos = merged_track_instances.query_pos
        return updated_queries


def build(args, layer_name, dim_in, hidden_dim, dim_out):
    interaction_layers = {
        'QIM': QueryInteractionModule,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, dim_out)
