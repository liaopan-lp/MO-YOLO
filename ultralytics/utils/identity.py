import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from . import _timing


class Identity(_BaseMetric):
    """Class which implements the Identity metrics (IDF1, IDR, IDP).
    Based on: https://arxiv.org/abs/1609.01775 (Ristani et al.)
    See also: https://github.com/JonathonLuiten/TrackEval
    """

    def __init__(self, config=None):
        super().__init__()
        self.plottable = False
        self.integer_fields = ['IDTP', 'IDFN', 'IDFP']
        self.float_fields = ['IDF1', 'IDR', 'IDP']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.float_fields + self.integer_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the Identity metrics for one sequence.

        The Identity metric computes a global minimum-cost bipartite matching between
        ground truth tracks and predicted tracks, where the cost is based on the
        temporal overlap (IOU over time) of track pairs.

        Args:
            data (dict): Dictionary containing per-timestep tracking data with keys:
                - 'num_timesteps': int
                - 'gt_ids': list of arrays with gt track IDs per timestep
                - 'tracker_ids': list of arrays with tracker IDs per timestep
                - 'similarity_scores': list of arrays with IOU scores per timestep
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
            res['IDFN'] = data['num_gt_dets']
            res['IDF1'] = 0.0
            res['IDR'] = 0.0
            res['IDP'] = 0.0
            return res
        if data['num_gt_dets'] == 0:
            res['IDFP'] = data['num_tracker_dets']
            res['IDF1'] = 0.0
            res['IDR'] = 0.0
            res['IDP'] = 0.0
            return res

        num_gt_ids = data['num_gt_ids']
        num_tracker_ids = data['num_tracker_ids']

        # Build ID-based confusion matrix:
        # For each (gt_id, tracker_id) pair, count:
        #   - IDTP: number of frames where both are present and matched (IOU >= threshold)
        #   - Also track the temporal overlap for global matching

        # Step 1: Build a matrix of pairwise ID associations
        # gt_id_mapping: map original gt_id to consecutive index
        all_gt_ids = set()
        all_tracker_ids = set()
        for gt_ids_t in data['gt_ids']:
            all_gt_ids.update(gt_ids_t.flatten().tolist() if hasattr(gt_ids_t, 'flatten') else list(gt_ids_t))
        for tracker_ids_t in data['tracker_ids']:
            all_tracker_ids.update(
                tracker_ids_t.flatten().tolist() if hasattr(tracker_ids_t, 'flatten') else list(tracker_ids_t))

        all_gt_ids = sorted([int(x) for x in all_gt_ids])
        all_tracker_ids = sorted([int(x) for x in all_tracker_ids])

        if len(all_gt_ids) == 0 or len(all_tracker_ids) == 0:
            res['IDFN'] = data['num_gt_dets']
            res['IDFP'] = data['num_tracker_dets']
            res['IDF1'] = 0.0
            res['IDR'] = 0.0
            res['IDP'] = 0.0
            return res

        gt_id_to_idx = {gt_id: i for i, gt_id in enumerate(all_gt_ids)}
        tracker_id_to_idx = {trk_id: i for i, trk_id in enumerate(all_tracker_ids)}

        n_gt = len(all_gt_ids)
        n_trk = len(all_tracker_ids)

        # Count temporal overlaps and matches for each (gt_id, tracker_id) pair
        # idtp_matrix[i,j] = number of frames where gt_id i and tracker_id j are both present AND matched
        idtp_matrix = np.zeros((n_gt, n_trk), dtype=np.float64)

        # gt_presence[i] = number of frames where gt_id i is present
        gt_presence = np.zeros(n_gt, dtype=np.float64)
        # tracker_presence[j] = number of frames where tracker_id j is present
        tracker_presence = np.zeros(n_trk, dtype=np.float64)

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            gt_ids_flat = gt_ids_t.flatten() if hasattr(gt_ids_t, 'flatten') else np.array(gt_ids_t)
            tracker_ids_flat = tracker_ids_t.flatten() if hasattr(tracker_ids_t, 'flatten') else np.array(tracker_ids_t)

            # Count presence
            for gt_id in gt_ids_flat:
                gt_idx = gt_id_to_idx.get(int(gt_id))
                if gt_idx is not None:
                    gt_presence[gt_idx] += 1

            for trk_id in tracker_ids_flat:
                trk_idx = tracker_id_to_idx.get(int(trk_id))
                if trk_idx is not None:
                    tracker_presence[trk_idx] += 1

            # Find matches at this timestep using Hungarian algorithm
            if len(gt_ids_flat) > 0 and len(tracker_ids_flat) > 0:
                similarity = data['similarity_scores'][t]
                if similarity.ndim == 1:
                    similarity = similarity.reshape(len(gt_ids_flat), len(tracker_ids_flat))

                match_threshold = 0.5  # standard IOU threshold
                cost_matrix = -similarity

                try:
                    match_rows, match_cols = linear_sum_assignment(cost_matrix)
                    actually_matched = similarity[match_rows, match_cols] >= match_threshold - np.finfo('float').eps
                    match_rows = match_rows[actually_matched]
                    match_cols = match_cols[actually_matched]

                    for r, c in zip(match_rows, match_cols):
                        gt_idx = gt_id_to_idx.get(int(gt_ids_flat[r]))
                        trk_idx = tracker_id_to_idx.get(int(tracker_ids_flat[c]))
                        if gt_idx is not None and trk_idx is not None:
                            idtp_matrix[gt_idx, trk_idx] += 1
                except (ValueError, IndexError):
                    pass

        # Step 2: Global bipartite matching between gt tracks and tracker tracks
        # The cost for matching gt_id i to tracker_id j is:
        #   cost(i,j) = -(idtp_matrix[i,j])  (negative because we minimize)
        # After matching, IDTP for a pair = idtp_matrix[i,j] for matched pairs

        # Pad the cost matrix to handle rectangular cases
        max_dim = max(n_gt, n_trk)
        cost_matrix = np.zeros((max_dim, max_dim), dtype=np.float64)
        cost_matrix[:n_gt, :n_trk] = -idtp_matrix

        try:
            match_rows, match_cols = linear_sum_assignment(cost_matrix)
        except ValueError:
            match_rows = np.array([], dtype=int)
            match_cols = np.array([], dtype=int)

        # Filter to valid matches (within actual gt/tracker dimensions)
        valid = (match_rows < n_gt) & (match_cols < n_trk)
        match_rows = match_rows[valid]
        match_cols = match_cols[valid]

        # Calculate IDTP, IDFN, IDFP
        idtp = 0
        for r, c in zip(match_rows, match_cols):
            idtp += idtp_matrix[r, c]

        idfn = sum(gt_presence) - idtp
        idfp = sum(tracker_presence) - idtp

        res['IDTP'] = idtp
        res['IDFN'] = idfn
        res['IDFP'] = idfp

        # Calculate IDF1, IDR, IDP
        if idtp > 0:
            res['IDR'] = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
            res['IDP'] = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
            res['IDF1'] = 2 * idtp / (2 * idtp + idfn + idfp) if (2 * idtp + idfn + idfp) > 0 else 0.0
        else:
            res['IDR'] = 0.0
            res['IDP'] = 0.0
            res['IDF1'] = 0.0

        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)

        # Recalculate derived float metrics from combined integer counts
        if res['IDTP'] > 0:
            res['IDR'] = res['IDTP'] / (res['IDTP'] + res['IDFN']) if (res['IDTP'] + res['IDFN']) > 0 else 0.0
            res['IDP'] = res['IDTP'] / (res['IDTP'] + res['IDFP']) if (res['IDTP'] + res['IDFP']) > 0 else 0.0
            res['IDF1'] = 2 * res['IDTP'] / (2 * res['IDTP'] + res['IDFN'] + res['IDFP']) if (
                                                                                                       2 * res['IDTP'] +
                                                                                                       res['IDFN'] +
                                                                                                       res['IDFP']) > 0 else 0.0
        else:
            res['IDR'] = 0.0
            res['IDP'] = 0.0
            res['IDF1'] = 0.0

        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values."""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                vals = [v[field] for v in all_res.values() if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0]
            else:
                vals = [v[field] for v in all_res.values()]
            res[field] = np.mean(vals) if vals else 0.0
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        if res['IDTP'] > 0:
            res['IDR'] = res['IDTP'] / (res['IDTP'] + res['IDFN']) if (res['IDTP'] + res['IDFN']) > 0 else 0.0
            res['IDP'] = res['IDTP'] / (res['IDTP'] + res['IDFP']) if (res['IDTP'] + res['IDFP']) > 0 else 0.0
            res['IDF1'] = 2 * res['IDTP'] / (2 * res['IDTP'] + res['IDFN'] + res['IDFP']) if (
                                                                                                       2 * res['IDTP'] +
                                                                                                       res['IDFN'] +
                                                                                                       res['IDFP']) > 0 else 0.0
        else:
            res['IDR'] = 0.0
            res['IDP'] = 0.0
            res['IDF1'] = 0.0
        return res