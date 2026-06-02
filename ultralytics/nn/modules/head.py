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

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder', 'DecoderTracker', 'MOTRTrack'


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
        m = self  # self.model[-1]  # Detect() class
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class DecoderTracker(nn.Module):
    """
    DecoderTracker head with Fixed-Size Query Memory (FSQM).
    
    Implements:
    - Fixed-size track query memory pool (N_m queries)
    - Track initiation from high-confidence detections
    - Track termination after 3 consecutive low-confidence frames
    - Self-attention masking for inactive queries
    """
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), use_fsqm=True, d_model=256, aux_loss=False, nq=300, training_stage=3):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.aux_loss = aux_loss
        self.use_fsqm = use_fsqm  # whether to use Fixed-Size Query Memory
        self.is_first = True
        self.training_stage = training_stage  # 1=detection, 2=TBSP, 3=TALA
        self.tbsp_iou_threshold = 0.5  # IOU threshold for TBSP filtering
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

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Track instances storage
        self.track_instances = None
        self.track_instances_pre = None
        self.memory_bank = None

        # FSQM parameters (Fixed-Size Query Memory)
        self.tau_in = 0.5   # threshold for track initiation
        self.tau_out = 0.3  # threshold for track termination
        self.max_obj_id = 0 # global ID counter

    def _generate_empty_tracks(self, len_before=0):
        from MOTR.models.structures import Instances
        track_instances = Instances((1, 1))
        num_queries, dim = self.decoder.num_queries, self.decoder.hidden_dim * 2
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Fixed-size memory pool: N_m queries, all initialized to zero
        track_instances.ref_pts = torch.zeros(num_queries, 4, device=device)
        track_instances.query_pos = torch.zeros((num_queries, 256), dtype=torch.float, device=device)
        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)
        
        # Global ID pool: all inactive (-1)
        track_instances.obj_idxes = torch.full((len(track_instances), 1), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), 1), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.nc), dtype=torch.float, device=device)

        # FSQM: consecutive low-confidence frame counter
        track_instances.low_conf_count = torch.zeros((len(track_instances),), dtype=torch.long, device=device)

        return track_instances.to(device)

    def _generate_attn_mask(self):
        """
        Generate self-attention mask for FSQM.
        
        Mask size: (N_m + N) x (N_m + N) where N_m = track queries, N = detect queries.
        Uses vectorized operations for efficiency.
        Detect queries always have ID=0 (active).
        
        Mask_{ij} = -inf if ID_ext[i] == -1 OR ID_ext[j] == -1, else 0.
        """
        if self.track_instances is None:
            return None

        nm = self.nq  # N_m (track queries)
        nq = self.nq  # N (detect queries)
        device = self.track_instances.obj_idxes.device

        # Get track query IDs (first N_m entries only, even if track_instances was expanded)
        track_ids = self.track_instances.obj_idxes[:nm].view(-1)  # (N_m,)

        # Extend ID: track IDs + detect IDs (always active, ID=0)
        id_ext = torch.cat([track_ids, torch.zeros(nq, dtype=torch.long, device=device)], dim=0)

        # Generate mask using vectorized operations (much faster than nested loops)
        inactive_i = (id_ext == -1).unsqueeze(1).float()  # (N_m+N, 1)
        inactive_j = (id_ext == -1).unsqueeze(0).float()  # (1, N_m+N)
        inactive_mask = (inactive_i + inactive_j) > 0  # (N_m+N, N_m+N)

        # Use large negative number to avoid NaN in softmax
        mask = torch.where(inactive_mask,
                           torch.tensor(-1e4, device=device, dtype=torch.float32),
                           torch.tensor(0.0, device=device, dtype=torch.float32))

        if (id_ext == -1).any():
            return mask
        return None

    def _tbsp_filter(self, track_pred_boxes, track_pred_logits, det_pred_boxes, det_pred_logits):
        """
        TBSP: Tracking Box Selection Process.
        Filter detect queries that overlap with track queries by IOU threshold.
        
        For stage 2 training: detect queries with IOU > threshold against any track query are discarded.
        This prevents detect queries from being assigned to already-tracked objects.
        
        Args:
            track_pred_boxes: (N_m, 4) - predicted boxes from track queries (active ones)
            track_pred_logits: (N_m, nc) - predicted logits from track queries
            det_pred_boxes: (N_m, 4) - predicted boxes from detect queries
            det_pred_logits: (N_m, nc) - predicted logits from detect queries
            
        Returns:
            filtered_det_boxes, filtered_det_logits, keep_mask
        """
        # Find active track queries (obj_idxes >= 0)
        active_mask = self.track_instances.obj_idxes.view(-1) >= 0
        if not active_mask.any():
            # No active tracks, all detect queries pass through
            keep_mask = torch.ones(det_pred_boxes.shape[0], dtype=torch.bool, device=det_pred_boxes.device)
            return det_pred_boxes, det_pred_logits, keep_mask
        
        active_track_boxes = track_pred_boxes[active_mask]  # (n_active, 4)
        
        # Compute IOU between each detect query and all active track queries
        # det_pred_boxes: (N_det, 4), active_track_boxes: (N_track, 4)
        # Use pairwise_iou from structures
        det_boxes_xyxy = Boxes(det_pred_boxes)
        track_boxes_xyxy = Boxes(active_track_boxes)
        
        try:
            ious = pairwise_iou(det_boxes_xyxy, track_boxes_xyxy)  # (N_det, N_active)
            # Keep detect queries where max IOU with any track query <= threshold
            max_iou_per_det, _ = ious.max(dim=1)  # (N_det,)
            keep_mask = max_iou_per_det <= self.tbsp_iou_threshold
        except:
            keep_mask = torch.ones(det_pred_boxes.shape[0], dtype=torch.bool, device=det_pred_boxes.device)
        
        return det_pred_boxes, det_pred_logits, keep_mask

    def forward(self, x, batch=None, is_first=True):
        """Forward pass with FSQM: fixed-size query memory with attention masking."""

        shape = x[0].shape  # BCHW
        
        # Stage 1 (detection): always treat as first frame (no tracking)
        if self.training and self.training_stage == 1:
            self.is_first = True
        else:
            self.is_first = is_first

        if self.is_first or self.track_instances is None:
            # Initialize fixed-size track memory (all zeros, all IDs=-1)
            self.track_instances = self._generate_empty_tracks()
            self.track_base = RuntimeTrackerBase(training=self.training)
            self.track_base.clear()
            ref_pts = None
            pre_class = None
            track_query_pos = None
            # No masking needed for first frame (all tracks inactive)
            attn_mask = None
        else:
            # Use stored track queries (fixed-size pool)
            ref_pts = self.track_instances.ref_pts
            pre_class = self.track_instances.pred_logits
            track_query_pos = self.track_instances.query_pos
            if len(self.track_instances) <= 0:
                ref_pts = None
                pre_class = None
                track_query_pos = None
            # Generate attention mask based on track IDs (only when FSQM is enabled)
            attn_mask = self._generate_attn_mask() if self.use_fsqm else None

        [dec_bboxes, dec_scores, enc_bbox, enc_outputs_class, dn_meta, init_reference,
         dec_output_embeding] = self.decoder(x,
                                             track_query_pos=track_query_pos,
                                             track_ref_pts=ref_pts,
                                             batch=batch,
                                             is_first=self.is_first,
                                             pre_class=pre_class,
                                             attn_mask=attn_mask)

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
        """
        Update track instances with decoder outputs and manage track lifecycle via FSQM.
        
        Decoder input is [DN (if any), track_queries (N_m), detect_queries (N_m)].
        Decoder output is [DN (if any), track_queries (N_m), detect_queries (N_m)].
        
        We only update track_instances (N_m entries) with the track query portion (first N_m).
        For matching, we use all 2*N_m outputs.
        After QIM update, track_instances remains N_m entries.
        """
        dec_bboxes, dec_scores, enc_bbox, enc_outputs_class, enc_outputs_coord_unact, init_reference, dec_output_embeding = out

        # Split DN and detection outputs if DN metadata exists
        if enc_outputs_coord_unact is not None:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, enc_outputs_coord_unact['dn_num_split'], dim=2)
            _, init_reference = torch.split(init_reference, enc_outputs_coord_unact['dn_num_split'], dim=1)
            dn_scores, dec_scores = torch.split(dec_scores, enc_outputs_coord_unact['dn_num_split'], dim=2)

        # Compute reference points across all decoder layers
        outputs_coords = []
        for lvl in range(dec_bboxes.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = dec_bboxes[lvl - 1]
            from MOTR.util.misc import inverse_sigmoid
            reference = inverse_sigmoid(reference)
            outputs_coord = dec_bboxes[lvl]
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        if not self.training:
            init_reference = init_reference.clone().to(dec_bboxes.device)

        ref_pts_all = torch.cat([init_reference[None], dec_bboxes[:, :, :, :4]], dim=0).to(dec_scores.device)

        # Extract final layer outputs (all 2*N_m entries: track + detect)
        outputs_class = dec_scores
        all_pred_logits = outputs_class[-1]   # (bs, 2*N_m, nc)
        all_pred_boxes = outputs_coord[-1]    # (bs, 2*N_m, 4)
        all_ref_pts = ref_pts_all[-1]         # (bs, 2*N_m, 4)
        all_hs = dec_output_embeding[-1]      # (bs, 2*N_m, d)

        # Split into track queries (first N_m) and detect queries (last N_m)
        nm = self.nq
        track_pred_logits = all_pred_logits[:, :nm, :]    # (bs, N_m, nc)
        track_pred_boxes = all_pred_boxes[:, :nm, :]      # (bs, N_m, 4)
        track_hs = all_hs[:, :nm, :]                       # (bs, N_m, d)
        track_scores = track_pred_logits[0, :].sigmoid().max(dim=-1).values.detach()

        # Update track_instances with track query outputs only (N_m entries)
        self.track_instances.scores = track_scores
        self.track_instances.pred_logits = track_pred_logits[0]  # (N_m, nc)
        self.track_instances.pred_boxes = track_pred_boxes[0]    # (N_m, 4)
        self.track_instances.output_embedding = track_hs[0]      # (N_m, d)

        # Lifecycle management: FSQM or default tracker
        if self.use_fsqm:
            self._fsqm_lifecycle_update()
        else:
            self.track_base.update(self.track_instances)

        # Match with GT (training) or assign IDs (inference)
        matched_indices = None
        unmatched_track_idxes = None
        if batch is not None:
            pred_logits_i = self.track_instances.pred_logits  # (N_m, nc)
            pred_boxes_i = self.track_instances.pred_boxes    # (N_m, 4)
            self.track_instances.matched_gt_idxes[...] = -1

            if len(batch['track_id']) == 0:
                return [None, None]

            if is_first or not (self.track_instances.obj_idxes != -1).any():
                # First frame or no active tracks: match all with GT
                indices = self.matcher(track_pred_boxes.unsqueeze(0), track_pred_logits.unsqueeze(0),
                                       batch['bboxes'], batch['cls'], batch['gt_groups'])
                indices = [(ind[0].to(pred_logits_i.device), ind[1].to(pred_logits_i.device)) for ind in indices]

                for i, ind in enumerate(indices):
                    self.track_instances.matched_gt_idxes[ind[0]] = ind[1]
                    self.track_instances.obj_idxes[ind[0]] = batch['track_id'][ind[1]].long()

                # Update IoU for matched tracks
                active_idxes = torch.logical_and(
                    self.track_instances.obj_idxes[:, 0] >= 0,
                    self.track_instances.matched_gt_idxes >= 0
                )
                matched_indices = indices
            else:
                # Subsequent frames: match active tracks by ID, match remaining with GT
                active_idxes = (self.track_instances.obj_idxes >= 0).squeeze()
                gt_bboxes = batch['bboxes']
                gt_obj_idxes = batch['track_id']

                # Match by track ID
                track_indices_flat = self.track_instances.obj_idxes.view(-1)
                gt_indices_flat = gt_obj_idxes.view(-1)
                matching_indices = torch.nonzero(track_indices_flat[:, None] == gt_indices_flat)
                i, j = matching_indices[:, 0], matching_indices[:, 1]
                self.track_instances.matched_gt_idxes[i] = j

                # Split matched and unmatched tracks
                full_track_idxes = torch.arange(nm, dtype=torch.long, device=pred_logits_i.device)
                matched_track_idxes = (track_indices_flat >= 0)
                prev_matched_indices = torch.stack(
                    [full_track_idxes[matched_track_idxes],
                     self.track_instances.matched_gt_idxes[matched_track_idxes]], dim=1)

                # Unmatched tracks (detection queries) - these are the inactive slots
                unmatched_track_idxes = full_track_idxes[track_indices_flat == -1]

                # Unmatched GT (new objects)
                tgt_indexes = self.track_instances.matched_gt_idxes
                tgt_indexes = tgt_indexes[tgt_indexes != -1]
                tgt_state = torch.zeros(len(gt_indices_flat), device=pred_logits_i.device)
                tgt_state[tgt_indexes] = 1
                full_tgt_idxes = torch.arange(len(gt_indices_flat), device=pred_logits_i.device)
                untracked_tgt_indexes = full_tgt_idxes[tgt_state == 0]

                unmatched_tgt = {
                    'pred_logits': batch['cls'][untracked_tgt_indexes],
                    'pred_boxes': batch['bboxes'][untracked_tgt_indexes],
                    'track_id': batch['track_id'][untracked_tgt_indexes],
                }

                # TBSP filtering for stage 2: filter detect queries that overlap with active tracks
                if self.training and self.training_stage == 2:
                    _, _, tbsp_keep_mask = self._tbsp_filter(
                        track_pred_boxes=pred_boxes_i,
                        track_pred_logits=pred_logits_i,
                        det_pred_boxes=self.track_instances.pred_boxes[unmatched_track_idxes],
                        det_pred_logits=self.track_instances.pred_logits[unmatched_track_idxes]
                    )
                    # Only keep detect queries that passed TBSP filtering
                    unmatched_track_idxes = unmatched_track_idxes[tbsp_keep_mask]

                # Match unmatched detections with unmatched GT
                if len(unmatched_track_idxes) > 0 and len(untracked_tgt_indexes) > 0:
                    unmatched_outputs = {
                        'pred_logits': self.track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
                        'pred_boxes': self.track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
                    }

                    new_track_indices = self.matcher(unmatched_outputs["pred_boxes"], unmatched_outputs["pred_logits"],
                                                      unmatched_tgt['pred_boxes'], unmatched_tgt['pred_logits'],
                                                      [len(untracked_tgt_indexes)])

                    src_idx = new_track_indices[0][0]
                    tgt_idx = new_track_indices[0][1]
                    new_matched_indices = torch.stack(
                        [unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]], dim=1).to(pred_logits_i.device)
                else:
                    new_matched_indices = torch.zeros((0, 2), dtype=torch.long, device=pred_logits_i.device)

                # Assign IDs to newly matched detections
                self.track_instances.obj_idxes[new_matched_indices[:, 0]] = batch['track_id'][
                    new_matched_indices[:, 1]].long()

                # Combine matched indices
                matched_indices_ = torch.cat([new_matched_indices, prev_matched_indices], dim=0)
                matched_indices = [(matched_indices_[:, 0], matched_indices_[:, 1])]
                self.track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # Update track embeddings via QIM (TAN)
        # track_instances has exactly N_m entries; QIM updates query_pos and ref_pts
        tmp = {'detect_queries': self.track_instances, 'track_queries': self.track_instances}
        out_track_instances = self.track_embed(tmp)

        # Update track instances with QIM outputs (N_m entries preserved)
        self.track_instances.ref_pts = out_track_instances.ref_pts
        self.track_instances.query_pos = out_track_instances.query_pos

        if self.training:
            return [matched_indices, unmatched_track_idxes]
        return [None, None]

    def _fsqm_lifecycle_update(self):
        """
        FSQM online update: manage track lifecycle.
        
        - Track Termination: If confidence < tau_out for 3 consecutive frames, reset slot
        - Track Initiation: If detection confidence > tau_in, fill first inactive slot
        """
        track_instances = self.track_instances
        track_scores = track_instances.scores

        # --- Track Termination ---
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] >= 0:  # Active track
                if track_scores[i] < self.tau_out:
                    track_instances.low_conf_count[i] += 1
                    if track_instances.low_conf_count[i] >= 3:
                        # Reset to zero vector, ID = -1
                        d = track_instances.query_pos.shape[1]
                        track_instances.query_pos[i] = torch.zeros(d, device=track_instances.query_pos.device)
                        track_instances.ref_pts[i] = torch.zeros(4, device=track_instances.ref_pts.device)
                        track_instances.output_embedding[i] = torch.zeros(
                            track_instances.output_embedding.shape[1], device=track_instances.output_embedding.device)
                        track_instances.obj_idxes[i] = -1
                        track_instances.pred_boxes[i] = torch.zeros(4, device=track_instances.pred_boxes.device)
                        track_instances.scores[i] = 0.0
                        track_instances.low_conf_count[i] = 0
                else:
                    track_instances.low_conf_count[i] = 0  # Reset counter

        # --- Track Initiation ---
        # Find high-confidence detections not yet assigned
        det_indices = torch.where(
            (track_instances.obj_idxes.view(-1) == -1) & (track_scores > self.tau_in)
        )[0]

        for det_idx in det_indices:
            # Find first inactive slot
            inactive_slots = torch.where(track_instances.obj_idxes.view(-1) == -1)[0]
            if len(inactive_slots) == 0:
                break
            slot_idx = inactive_slots[0]

            # Skip if det_idx is itself the slot (already inactive)
            if det_idx == slot_idx:
                track_instances.obj_idxes[slot_idx] = self.max_obj_id
                self.max_obj_id += 1
                track_instances.scores[slot_idx] = track_scores[det_idx]
                track_instances.low_conf_count[slot_idx] = 0
                continue

            # Move detection to inactive slot
            track_instances.query_pos[slot_idx] = track_instances.query_pos[det_idx].clone()
            track_instances.ref_pts[slot_idx] = track_instances.ref_pts[det_idx].clone()
            track_instances.output_embedding[slot_idx] = track_instances.output_embedding[det_idx].clone()
            track_instances.pred_boxes[slot_idx] = track_instances.pred_boxes[det_idx].clone()
            track_instances.pred_logits[slot_idx] = track_instances.pred_logits[det_idx].clone()
            track_instances.scores[slot_idx] = track_scores[det_idx]
            track_instances.obj_idxes[slot_idx] = self.max_obj_id
            self.max_obj_id += 1
            track_instances.low_conf_count[slot_idx] = 0

            # Clear source slot
            d = track_instances.query_pos.shape[1]
            track_instances.query_pos[det_idx] = torch.zeros(d, device=track_instances.query_pos.device)
            track_instances.ref_pts[det_idx] = torch.zeros(4, device=track_instances.ref_pts.device)
            track_instances.output_embedding[det_idx] = torch.zeros(
                track_instances.output_embedding.shape[1], device=track_instances.output_embedding.device)
            track_instances.obj_idxes[det_idx] = -1
            track_instances.pred_boxes[det_idx] = torch.zeros(4, device=track_instances.pred_boxes.device)
            track_instances.scores[det_idx] = 0.0

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() class
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
        """Initialize the YOLO network with default parameters and Convolutional Layers."""
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

    def forward(self, x, track_ref_pts=None, batch=None, is_first=False, pre_class=None, track_query_pos=None,
                attn_mask=None):
        from ultralytics.models.utils.ops import get_track_cdn_group

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

        dn_embed, dn_bbox, attn_mask_dn, dn_meta = \
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

        if not self.training:
            refer_bbox = refer_bbox.to(embed.dtype)

        # Merge DN attention mask with FSQM attention mask
        # attn_mask_dn comes from DN group, attn_mask comes from FSQM
        merged_attn_mask = attn_mask  # FSQM mask (for track+det queries)
        if attn_mask_dn is not None and attn_mask is not None:
            # Both masks exist: need to merge them
            # DN mask handles DN-to-DN masking, FSQM mask handles track masking
            # For simplicity, use FSQM mask (DN queries are always active)
            merged_attn_mask = attn_mask
        elif attn_mask_dn is not None:
            merged_attn_mask = attn_mask_dn

        # decoder
        dec_bboxes, dec_scores, dec_output_embeding = self.decoder(embed,
                                                                    refer_bbox,
                                                                    feats,
                                                                    shapes,
                                                                    self.dec_bbox_head,
                                                                    self.dec_score_head,
                                                                    self.query_pos_head,
                                                                    attn_mask=merged_attn_mask,
                                                                    track_query_embed=query_pos
                                                                    )

        reference_points = track_ref_pts.repeat(bs, 1, 1).sigmoid()

        init_reference_out = reference_points
        dec_scores_out = dec_scores

        x = dec_bboxes, dec_scores_out, enc_bboxes, enc_scores, dn_meta, init_reference_out, dec_output_embeding

        return x

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
            refer_bbox = torch.cat([refer_bbox, refer_bbox_1], dim=1)
            if track_query_pos.device != refer_bbox_1.device:
                track_query_pos = track_query_pos.to(refer_bbox_1.device)
            query_pos = torch.cat([track_query_pos.unsqueeze(0), pos2posemb(refer_bbox_1)], dim=1)

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
                    if keep[j]:
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
        for i in range(len(track_instances.scores)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    track_instances.obj_idxes[i] = -1

        active_track_idxes = track_instances.obj_idxes >= 0
        if any(active_track_idxes):
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]
        else:
            active_track_instances = track_instances
            return active_track_instances

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
        tmp_num = 0

        try:
            for i in range(len(active_track_instances.obj_idxes)):
                if active_track_instances.obj_idxes[i] > (self.max_obj_id_pre):
                    active_track_instances.obj_idxes[i] = self.max_obj_id_pre + tmp_num + 1
                    tmp_num += 1
        except:
            pass

        if active_track_instances is None:
            self.max_obj_id_pre = self.max_obj_id
        else:
            self.max_obj_id = max(active_track_instances.obj_idxes.cpu()) + 1
            self.max_obj_id_pre = self.max_obj_id - 1
        return active_track_instances


# Alias for YAML compatibility (must be after DecoderTracker class definition)
MOTRTrack = DecoderTracker
