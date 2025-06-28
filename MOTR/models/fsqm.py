import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from MOTR.models.structures import Instances

class FSQM:
    """
    Fixed-Size Query Memory 模块，用于管理固定数量的跟踪查询，解决动态数据导致的推理效率问题
    """

    def __init__(self,
                 max_num_queries: int,
                 feature_dim: int,
                 in_threshold: float = 0.7,
                 out_threshold: float = 0.3,
                 consecutive_frames: int = 3):
        """
        初始化FSQM模块

        Args:
            max_num_queries: 最大查询数量，固定内存大小
            feature_dim: 查询特征维度
            in_threshold: 新查询注入阈值
            out_threshold: 查询移除阈值
            consecutive_frames: 连续低于阈值的帧数才移除
        """
        self.max_num_queries = max_num_queries
        self.feature_dim = feature_dim
        self.in_threshold = in_threshold
        self.out_threshold = out_threshold
        self.consecutive_frames = consecutive_frames

        # 初始化查询内存
        self.query_memory = torch.zeros(max_num_queries, feature_dim)  # 形状: [N, d]
        self.confidence = torch.zeros(max_num_queries)  # 置信度向量
        self.global_id_pool = list(range(max_num_queries))  # 全局ID池，初始为0到N-1
        self.ids = torch.ones(max_num_queries, dtype=torch.long) * -1  # 所有查询ID初始化为-1
        self.bounding_boxes = torch.zeros(max_num_queries, 4)  # 边界框坐标，初始为0

        # 连续低于阈值的帧数记录
        self.consecutive_low_frames = torch.zeros(max_num_queries, dtype=torch.int)

    def _find_first_inactive_query(self) -> int:
        """查找第一个非活跃查询的索引（ID为-1）"""
        inactive_indices = torch.where(self.ids == -1)[0]
        if len(inactive_indices) > 0:
            return inactive_indices[0].item()
        return -1  # 没有可用查询位置

    def inject_new_queries(self,
                           detect_queries: Instances,
                           frame_idx: int = 0) -> Instances:
        """
        注入新的检测查询到FSQM

        Args:
            detect_queries: 包含检测查询的Instances对象，需包含'scores'和'pred_boxes'
            frame_idx: 当前帧索引，用于日志记录

        Returns:
            包含更新后查询的Instances对象
        """
        if len(detect_queries) == 0:
            return detect_queries

        # 过滤出置信度高于阈值的检测查询
        valid_indices = torch.where(detect_queries.scores > self.in_threshold)[0]
        valid_queries = detect_queries[valid_indices]

        if len(valid_queries) == 0:
            return detect_queries

        updated_queries = []
        for i in range(len(valid_queries)):
            # 查找第一个非活跃查询位置
            inactive_idx = self._find_first_inactive_query()
            if inactive_idx == -1:
                # 内存已满，无法注入新查询
                break

            # 分配ID并更新内存
            self.ids[inactive_idx] = self.global_id_pool.pop(0)  # 从ID池获取ID
            self.query_memory[inactive_idx] = valid_queries.output_embedding[i]
            self.confidence[inactive_idx] = valid_queries.scores[i]
            self.bounding_boxes[inactive_idx] = valid_queries.pred_boxes[i]
            self.consecutive_low_frames[inactive_idx] = 0  # 重置连续低置信度帧数

            # 记录更新后的查询
            query_dict = {
                'output_embedding': self.query_memory[inactive_idx].unsqueeze(0),
                'scores': self.confidence[inactive_idx].unsqueeze(0),
                'pred_boxes': self.bounding_boxes[inactive_idx].unsqueeze(0),
                'obj_idxes': torch.tensor([self.ids[inactive_idx]]).unsqueeze(0)
            }
            updated_queries.append(Instances((0, 0), **query_dict))

        if updated_queries:
            return Instances.cat(updated_queries)
        return detect_queries

    def remove_inactive_queries(self) -> None:
        """移除连续多帧置信度低于阈值的非活跃查询"""
        low_confidence_indices = torch.where(self.confidence < self.out_threshold)[0]

        for idx in low_confidence_indices:
            self.consecutive_low_frames[idx] += 1
            if self.consecutive_low_frames[idx] >= self.consecutive_frames:
                # 连续多帧低于阈值，重置该查询
                self.query_memory[idx] = torch.zeros(self.feature_dim)
                self.confidence[idx] = 0.0
                self.global_id_pool.append(self.ids[idx].item())  # 回收ID
                self.ids[idx] = -1
                self.bounding_boxes[idx] = torch.zeros(4)
                self.consecutive_low_frames[idx] = 0

    def update_confidence(self,
                          track_queries: Instances,
                          frame_idx: int = 0) -> None:
        """
        更新跟踪查询的置信度

        Args:
            track_queries: 包含跟踪查询的Instances对象，需包含'scores'和'obj_idxes'
            frame_idx: 当前帧索引，用于日志记录
        """
        for i in range(len(track_queries)):
            query_id = track_queries.obj_idxes[i].item()
            if query_id >= 0 and query_id < self.max_num_queries:
                self.confidence[query_id] = track_queries.scores[i]
                self.bounding_boxes[query_id] = track_queries.pred_boxes[i]
                self.consecutive_low_frames[query_id] = 0  # 重置连续低置信度帧数

    def get_active_queries(self) -> Instances:
        """返回所有"""
        active_indices = torch.where(self.ids != -1)[0]
        if len(active_indices) == 0:
            return Instances((0, 0))

        active_embeddings = self.query_memory
        active_scores = self.confidence
        active_boxes = self.bounding_boxes
        active_ids = self.ids
        track_instances = Instances((1, 1))
        track_instances.output_embedding = active_embeddings
        track_instances.scores = active_scores
        track_instances.pred_boxes = active_boxes

        return Instances((0, 0),
                         output_embedding=active_embeddings,
                         scores=active_scores,
                         pred_boxes=active_boxes,
                         obj_idxes=active_ids.unsqueeze(1))

    def online_update(self,
                      detect_queries: Instances,
                      track_queries: Instances,
                      frame_idx: int = 0) -> Instances:
        """
        FSQM在线更新流程，包括注入新查询和移除非活跃查询

        Args:
            detect_queries: 检测查询Instances对象
            track_queries: 跟踪查询Instances对象
            frame_idx: 当前帧索引

        Returns:
            包含更新后所有活跃查询的Instances对象
        """
        # 1. 更新跟踪查询的置信度
        self.update_confidence(track_queries, frame_idx)

        # 2. 注入新的检测查询
        self.inject_new_queries(detect_queries, frame_idx)

        # 3. 移除非活跃查询
        self.remove_inactive_queries()

        # 4. 返回活跃查询
        return track_queries

    def reset(self) -> None:
        """重置FSQM内存，用于新视频序列的初始化"""
        self.query_memory = torch.zeros(self.max_num_queries, self.feature_dim)
        self.confidence = torch.zeros(self.max_num_queries)
        self.ids = torch.ones(self.max_num_queries, dtype=torch.long) * -1
        self.bounding_boxes = torch.zeros(self.max_num_queries, 4)
        self.consecutive_low_frames = torch.zeros(self.max_num_queries, dtype=torch.int)
        self.global_id_pool = list(range(self.max_num_queries))


# 使用示例
if __name__ == "__main__":
    # 初始化FSQM，最大查询数100，特征维度256
    fsqm = FSQM(max_num_queries=100, feature_dim=256)

    # 模拟检测查询（假设有3个新检测，置信度分别为0.8, 0.9, 0.6）
    detect_embeddings = torch.randn(3, 256)
    detect_scores = torch.tensor([0.8, 0.9, 0.6])
    detect_boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                 [0.5, 0.6, 0.7, 0.8],
                                 [0.2, 0.3, 0.4, 0.5]])

    detect_queries = Instances((0, 0),
                               output_embedding=detect_embeddings,
                               scores=detect_scores,
                               pred_boxes=detect_boxes)

    # 模拟跟踪查询（假设有2个跟踪查询）
    track_embeddings = torch.randn(2, 256)
    track_scores = torch.tensor([0.7, 0.5])
    track_boxes = torch.tensor([[0.15, 0.25, 0.35, 0.45],
                                [0.55, 0.65, 0.75, 0.85]])
    track_ids = torch.tensor([[10], [15]])  # 假设存在的ID

    track_queries = Instances((0, 0),
                              output_embedding=track_embeddings,
                              scores=track_scores,
                              pred_boxes=track_boxes,
                              obj_idxes=track_ids)

    # 执行在线更新
    active_queries = fsqm.online_update(detect_queries, track_queries, frame_idx=0)

    # 打印更新后的活跃查询数量
    print(f"活跃查询数量: {len(active_queries)}")