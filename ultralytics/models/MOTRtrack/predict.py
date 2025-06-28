# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results, TrackResults
from ultralytics.utils import ops


class TrackPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocess predictions and returns a list of Results objects."""
        try:
            nd = preds[0].shape[-1]
            bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
            track_instances = None
        except:
            nd = preds[0][0].shape[-1]
            bboxes, scores = preds[0][0].split((4, nd - 4), dim=-1)
            track_instances = preds[1]
        if track_instances is None:
            results = []
            for i, bbox in enumerate(bboxes):  # (300, 4)
                bbox = ops.xywh2xyxy(bbox)
                score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
                idx = score.squeeze(-1) > self.args.conf  # (300, )
                if self.args.classes is not None:
                    idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx

                pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
                orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
                oh, ow = orig_img.shape[:2]
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[..., [0, 2]] *= ow
                    pred[..., [1, 3]] *= oh
                path = self.batch[0]
                img_path = path[i] if isinstance(path, list) else path
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
            return results
        else:
            results = []
            try:
                active_track_instances = track_instances[track_instances.obj_idxes >= 0]
            except:
                active_track_instances = None

            if active_track_instances is not None:
                track_bbox = active_track_instances.pred_boxes
                track_scores = active_track_instances.scores
                if active_track_instances.pred_logits.ndim != 1:
                    _, track_clses = active_track_instances.pred_logits.max(-1, keepdim=True)
                else:
                    track_clses = torch.zeros(active_track_instances.pred_logits.shape[-1]).unsqueeze(1).to(track_scores.device)
                track_ids = active_track_instances.obj_idxes

                bbox = ops.xywh2xyxy(track_bbox)
                score = track_scores
                cls = track_clses
                id = track_ids
                idx = score.squeeze(-1) > self.args.conf  # (300, )
                if self.args.classes is not None:
                    idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
                score = score.unsqueeze(1)

                pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
                orig_img = orig_imgs[0] if isinstance(orig_imgs, list) else orig_imgs
                oh, ow = orig_img.shape[:2]
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[..., [0, 2]] *= ow
                    pred[..., [1, 3]] *= oh
                path = self.batch[0]
                img_path = path if isinstance(path, list) else path
                results.append(
                    TrackResults(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred, track_id=id))
                return results

            for i, bbox in enumerate(bboxes):  # (300, 4)
                bbox = ops.xywh2xyxy(bbox)
                score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
                idx = score.squeeze(-1) > self.args.conf  # (300, )
                if self.args.classes is not None:
                    idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
                pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
                orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
                oh, ow = orig_img.shape[:2]
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[..., [0, 2]] *= ow
                    pred[..., [1, 3]] *= oh
                path = self.batch[0]
                img_path = path[i] if isinstance(path, list) else path
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
            return results

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        # The size must be square(640) and scaleFilled.
        return [LetterBox(self.imgsz, auto=False, scaleFill=True)(image=x) for x in im]
