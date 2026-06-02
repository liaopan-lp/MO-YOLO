import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from . import _timing


class CLEAR(_BaseMetric):
    """Class which implements the CLEAR metrics (MOTA, MOTP, etc.).
    Based on: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188835 (Ristani et al.)
    and https://link.springer.com/article/10.1007/s11263-019-01215-5 (Luiten et al.)
    See also: https://github.com/JonathonLuiten/TrackEval
    """

    def __init__(self, config=None):
        super().__init__()
        self.plottable = False
        self.integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
        self.float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'sMOTA']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.float_fields + self.integer_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the CLEAR metrics for one sequence.

        Args:
            data (dict): Dictionary containing per-timestep tracking data with keys:
                - 'num_timesteps': int
                - 'gt_ids': list of arrays, each shape (N_t,) with gt track IDs per timestep
                - 'tracker_ids': list of arrays, each shape (M_t,) with tracker IDs per timestep
                - 'similarity_scores': list of arrays, each shape (N_t, M_t) with IOU scores
                - 'num_gt_dets': total number of gt detections
                - 'num_tracker_dets': total number of tracker detections
                - 'num_gt_ids': number of unique gt IDs
                - 'num_tracker_ids': number of unique tracker IDs
        """
        # Initialise results
        res = {}
        for field in self.integer_fields:
            res[field] = 0
        for field in self.float_fields:
            res[field] = 0.0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['CLR_FN'] = data['num_gt_dets']
            res['MOTA'] = -np.inf  # special case
            return res
        if data['num_gt_dets'] == 0:
            res['CLR_FP'] = data['num_tracker_dets']
            res['MOTA'] = -np.inf  # special case
            return res

        # Variables for tracking ID switches
        prev_to_cur_match = {}  # maps matched gt_id in prev frame -> tracker_id in prev frame
        # Also need reverse mapping for IDSW detection
        cur_match_gt_to_tracker = {}  # gt_id -> tracker_id at current timestep

        prev_gt_ids = np.array([], dtype=int)
        prev_tracker_ids = np.array([], dtype=int)

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # gt_ids_t and tracker_ids_t are arrays of track IDs for this timestep

            # Handle empty cases
            if len(gt_ids_t) == 0:
                res['CLR_FP'] += len(tracker_ids_t)
                prev_gt_ids = np.array([], dtype=int)
                prev_tracker_ids = np.array([], dtype=int)
                continue
            if len(tracker_ids_t) == 0:
                res['CLR_FN'] += len(gt_ids_t)
                # Check for fragmentation: gt tracks that were previously matched but now unmatched
                prev_gt_ids = np.array([], dtype=int)
                prev_tracker_ids = np.array([], dtype=int)
                continue

            # Flatten gt_ids_t and tracker_ids_t if needed
            gt_ids_flat = gt_ids_t.flatten() if hasattr(gt_ids_t, 'flatten') else np.array(gt_ids_t)
            tracker_ids_flat = tracker_ids_t.flatten() if hasattr(tracker_ids_t, 'flatten') else np.array(tracker_ids_t)

            # Get similarity scores (IOU matrix) for this timestep
            similarity = data['similarity_scores'][t]
            if similarity.ndim == 1:
                similarity = similarity.reshape(len(gt_ids_flat), len(tracker_ids_flat))

            # Hungarian matching using negative similarity (IOU)
            # We use a threshold of 0.5 for matching (standard CLEAR threshold)
            match_threshold = 0.5

            # Create cost matrix (negative IOU for minimization)
            cost_matrix = -similarity
            try:
                match_rows, match_cols = linear_sum_assignment(cost_matrix)
            except ValueError:
                # Handle edge cases
                match_rows = np.array([], dtype=int)
                match_cols = np.array([], dtype=int)

            # Filter matches by threshold
            actually_matched = similarity[match_rows, match_cols] >= match_threshold - np.finfo('float').eps
            match_rows = match_rows[actually_matched]
            match_cols = match_cols[actually_matched]

            num_matches = len(match_rows)
            res['CLR_TP'] += num_matches
            res['CLR_FN'] += len(gt_ids_flat) - num_matches
            res['CLR_FP'] += len(tracker_ids_flat) - num_matches

            # MOTP: sum of similarities for matched pairs
            if num_matches > 0:
                res['MOTP'] += sum(similarity[match_rows, match_cols])

            # Build current match mapping: gt_id -> tracker_id
            cur_match_gt_to_tracker = {}
            for r, c in zip(match_rows, match_cols):
                gt_id = int(gt_ids_flat[r])
                trk_id = int(tracker_ids_flat[c])
                cur_match_gt_to_tracker[gt_id] = trk_id

            # Count IDSW: a gt track was matched to a different tracker than previous frame
            for gt_id, trk_id in cur_match_gt_to_tracker.items():
                if gt_id in prev_to_cur_match:
                    if prev_to_cur_match[gt_id] != trk_id:
                        res['IDSW'] += 1

            # Count fragmentation: a gt track that was matched before, unmatched now, and will be matched again
            # Simplified version: count when a matched gt track becomes unmatched then re-matches
            for gt_id in prev_to_cur_match:
                if gt_id not in cur_match_gt_to_tracker:
                    # gt track was matched before but is now unmatched (either FN or still exists but unmatched)
                    if gt_id in gt_ids_flat:
                        res['Frag'] += 1

            # Update previous match mapping
            prev_to_cur_match = cur_match_gt_to_tracker.copy()
            prev_gt_ids = gt_ids_flat
            prev_tracker_ids = tracker_ids_flat

        # Calculate derived metrics
        # MOTP
        if res['CLR_TP'] > 0:
            res['MOTP'] = res['MOTP'] / res['CLR_TP']
        else:
            res['MOTP'] = 0.0

        # MOTA = 1 - (FN + FP + IDSW) / GT
        if data['num_gt_dets'] > 0:
            res['MOTA'] = 1.0 - (res['CLR_FN'] + res['CLR_FP'] + res['IDSW']) / data['num_gt_dets']
        else:
            res['MOTA'] = -np.inf

        # MODA = 1 - (FN + FP) / GT
        if data['num_gt_dets'] > 0:
            res['MODA'] = 1.0 - (res['CLR_FN'] + res['CLR_FP']) / data['num_gt_dets']
        else:
            res['MODA'] = -np.inf

        # CLR_Re (Recall) = TP / (TP + FN) = TP / GT
        if data['num_gt_dets'] > 0:
            res['CLR_Re'] = res['CLR_TP'] / data['num_gt_dets']
        else:
            res['CLR_Re'] = 0.0

        # CLR_Pr (Precision) = TP / (TP + FP)
        if data['num_tracker_dets'] > 0:
            res['CLR_Pr'] = res['CLR_TP'] / data['num_tracker_dets']
        else:
            res['CLR_Pr'] = 0.0

        # sMOTA = MOTA * MOTP
        if res['MOTA'] > -np.inf:
            res['sMOTA'] = res['MOTA'] * res['MOTP']
        else:
            res['sMOTA'] = 0.0

        # MT (Mostly Tracked), PT (Partially Tracked), ML (Mostly Lost)
        # A track is considered "mostly tracked" if >80% of its detections are tracked
        # "mostly lost" if <20% tracked, "partially tracked" otherwise
        gt_track_length = {}  # gt_id -> total frames it appears
        gt_track_matched = {}  # gt_id -> number of frames it was matched

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            gt_ids_flat = gt_ids_t.flatten() if hasattr(gt_ids_t, 'flatten') else np.array(gt_ids_t)
            tracker_ids_flat = tracker_ids_t.flatten() if hasattr(tracker_ids_t, 'flatten') else np.array(tracker_ids_t)

            for gt_id in gt_ids_flat:
                gt_id_int = int(gt_id)
                if gt_id_int not in gt_track_length:
                    gt_track_length[gt_id_int] = 0
                    gt_track_matched[gt_id_int] = 0
                gt_track_length[gt_id_int] += 1

            # Find matches for this timestep
            if len(gt_ids_flat) > 0 and len(tracker_ids_flat) > 0:
                similarity = data['similarity_scores'][t]
                if similarity.ndim == 1:
                    similarity = similarity.reshape(len(gt_ids_flat), len(tracker_ids_flat))

                cost_matrix = -similarity
                try:
                    match_rows, match_cols = linear_sum_assignment(cost_matrix)
                    actually_matched = similarity[match_rows, match_cols] >= 0.5 - np.finfo('float').eps
                    match_rows = match_rows[actually_matched]
                    match_cols = match_cols[actually_matched]
                    for r in match_rows:
                        gt_id_int = int(gt_ids_flat[r])
                        gt_track_matched[gt_id_int] += 1
                except (ValueError, IndexError):
                    pass

        for gt_id_int in gt_track_length:
            if gt_track_length[gt_id_int] == 0:
                continue
            ratio = gt_track_matched[gt_id_int] / gt_track_length[gt_id_int]
            if ratio >= 0.8:
                res['MT'] += 1
            elif ratio <= 0.2:
                res['ML'] += 1
            else:
                res['PT'] += 1

        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)

        # Recalculate derived float metrics from combined integer counts
        total_gt = sum([all_res[k].get('_total_gt', 0) for k in all_res.keys()])
        total_tracker = sum([all_res[k].get('_total_tracker', 0) for k in all_res.keys()])

        if res['CLR_TP'] > 0:
            total_motp_weighted = sum([all_res[k]['MOTP'] * all_res[k]['CLR_TP'] for k in all_res.keys()])
            res['MOTP'] = total_motp_weighted / res['CLR_TP']
        else:
            res['MOTP'] = 0.0

        if total_gt > 0:
            res['MOTA'] = 1.0 - (res['CLR_FN'] + res['CLR_FP'] + res['IDSW']) / total_gt
            res['MODA'] = 1.0 - (res['CLR_FN'] + res['CLR_FP']) / total_gt
            res['CLR_Re'] = res['CLR_TP'] / total_gt
        else:
            res['MOTA'] = -np.inf
            res['MODA'] = -np.inf
            res['CLR_Re'] = 0.0

        if total_tracker > 0:
            res['CLR_Pr'] = res['CLR_TP'] / total_tracker
        else:
            res['CLR_Pr'] = 0.0

        if res['MOTA'] > -np.inf:
            res['sMOTA'] = res['MOTA'] * res['MOTP']
        else:
            res['sMOTA'] = 0.0

        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values."""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                vals = [v[field] for v in all_res.values()
                        if not (v[field] == -np.inf or (isinstance(v[field], float) and np.isnan(v[field])))]
            else:
                vals = [v[field] for v in all_res.values()]
            res[field] = np.mean(vals) if vals else 0.0
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        # Recalculate derived float metrics
        total_gt = sum([all_res[k].get('_total_gt', 0) for k in all_res.keys()])
        if res['CLR_TP'] > 0:
            total_motp_weighted = sum([all_res[k]['MOTP'] * all_res[k]['CLR_TP'] for k in all_res.keys()])
            res['MOTP'] = total_motp_weighted / res['CLR_TP']
        else:
            res['MOTP'] = 0.0
        if total_gt > 0:
            res['MOTA'] = 1.0 - (res['CLR_FN'] + res['CLR_FP'] + res['IDSW']) / total_gt
        else:
            res['MOTA'] = -np.inf
        return res