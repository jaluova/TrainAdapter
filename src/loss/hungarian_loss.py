"""
Hungarian Loss for Coordinate Point Regression
处理预测点与真值点的最优匹配问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


class HungarianPointLoss(nn.Module):
    """
    Hungarian Loss for Point Regression
    解决预测点与真值点的最优匹配问题，支持多点预测和多真值
    """
    def __init__(self, 
                 inside_bbox_weight=1.0,
                 outside_bbox_weight=0.1,
                 match_cost='euclidean',
                 boundary_penalty_weight=0.1,
                 loss_type='hungarian_point',
                 grid_size=11,
                 grid_pos_weight=4.0,
                 neighbor_soft_label_weight=0.3):
        super(HungarianPointLoss, self).__init__()
        
        self.inside_bbox_weight = inside_bbox_weight
        self.outside_bbox_weight = outside_bbox_weight
        self.match_cost = match_cost
        self.boundary_penalty_weight = boundary_penalty_weight
        self.loss_type = loss_type
        self.grid_size = grid_size
        self.grid_pos_weight = grid_pos_weight
        self.neighbor_soft_label_weight = neighbor_soft_label_weight
        
    def parse_coordinates_from_text(self, text_outputs, image_width, image_height):
        """
        从MLLM文本输出中提取坐标点
        
        Args:
            text_outputs: 模型生成的文本列表 [str1, str2, ...]
            image_width: 图像宽度（用于归一化）
            image_height: 图像高度（用于归一化）
            
        Returns:
            坐标点列表的列表 [[[x1,y1], [x2,y2], ...], ...]
        """
        import re
        
        all_points = []
        
        for text in text_outputs:
            points = []
            
            # 使用正则表达式匹配[x,y]格式
            # 支持格式：[123,456], [123.5, 456.7], [x:123, y:456]等
            patterns = [
                r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',  # [x,y]
                r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',  # (x,y)
                r'x\s*:\s*(\d+(?:\.\d+)?)\s*,\s*y\s*:\s*(\d+(?:\.\d+)?)',  # x:123, y:456
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        x = float(match[0])
                        y = float(match[1])
                        
                        # 过滤掉明显不合理的坐标（负值或过大）
                        if 0 <= x <= image_width * 2 and 0 <= y <= image_height * 2:
                            points.append([x, y])
                    except (ValueError, IndexError):
                        continue
            
            # 如果没有解析到点，尝试提取数字对
            if len(points) == 0:
                numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
                if len(numbers) >= 2 and len(numbers) % 2 == 0:
                    for i in range(0, len(numbers), 2):
                        try:
                            x = float(numbers[i])
                            y = float(numbers[i+1])
                            if 0 <= x <= image_width * 2 and 0 <= y <= image_height * 2:
                                points.append([x, y])
                        except (ValueError, IndexError):
                            continue
            
            all_points.append(points)
        
        return all_points
    
    def compute_boundary_penalty(self, points, image_width, image_height):
        """
        计算边界惩罚：鼓励预测点在图像范围内
        
        Args:
            points: 预测点 [[x1,y1], [x2,y2], ...]
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            边界惩罚值
        """
        if len(points) == 0:
            return 0.0
        
        penalty = 0.0
        for point in points:
            x, y = point
            
            # 左边界惩罚
            if x < 0:
                penalty += -x
            # 右边界惩罚
            elif x > image_width:
                penalty += x - image_width
            
            # 上边界惩罚
            if y < 0:
                penalty += -y
            # 下边界惩罚
            elif y > image_height:
                penalty += y - image_height
        
        return penalty / len(points)
    
    def compute_distance_matrix(self, pred_points, gt_points, image_width, image_height):
        """
        计算预测点与真值点之间的距离矩阵
        
        Args:
            pred_points: 预测点 [[x1,y1], ...]
            gt_points: 真值点 [[gx1,gy1], ...]
            
        Returns:
            代价矩阵 [num_pred, num_gt]
        """
        num_pred = len(pred_points)
        num_gt = len(gt_points)
        
        if num_pred == 0 or num_gt == 0:
            return torch.zeros(0, 0)
        
        # 转换为tensor
        pred_tensor = torch.tensor(pred_points, dtype=torch.float32)  # [num_pred, 2]
        gt_tensor = torch.tensor(gt_points, dtype=torch.float32)      # [num_gt, 2]
        
        # 计算距离矩阵
        if self.match_cost == 'euclidean':
            # 欧氏距离
            pred_expanded = pred_tensor.unsqueeze(1)  # [num_pred, 1, 2]
            gt_expanded = gt_tensor.unsqueeze(0)      # [1, num_gt, 2]
            distances = torch.norm(pred_expanded - gt_expanded, p=2, dim=-1)  # [num_pred, num_gt]
            
        elif self.match_cost == 'l1':
            # L1距离
            pred_expanded = pred_tensor.unsqueeze(1)  # [num_pred, 1, 2]
            gt_expanded = gt_tensor.unsqueeze(0)      # [1, num_gt, 2]
            distances = torch.sum(torch.abs(pred_expanded - gt_expanded), dim=-1)  # [num_pred, num_gt]
            
        elif self.match_cost == 'smooth_l1':
            # Smooth L1距离
            pred_expanded = pred_tensor.unsqueeze(1)  # [num_pred, 1, 2]
            gt_expanded = gt_tensor.unsqueeze(0)      # [1, num_gt, 2]
            diff = torch.abs(pred_expanded - gt_expanded)
            distances = torch.where(
                diff < 1.0,
                0.5 * diff ** 2,
                diff - 0.5
            ).sum(dim=-1)  # [num_pred, num_gt]
        
        else:
            raise ValueError(f"Unknown match cost: {self.match_cost}")
        
        # 添加边界惩罚到每个预测点
        for i in range(num_pred):
            boundary_penalty = self.compute_boundary_penalty([pred_points[i]], image_width, image_height)
            distances[i, :] += self.boundary_penalty_weight * boundary_penalty
        
        return distances
    
    def hungarian_matching(self, cost_matrix):
        """
        使用Hungarian算法找到最优匹配
        
        Args:
            cost_matrix: 代价矩阵 [num_pred, num_gt]
            
        Returns:
            matched_pred_indices: 匹配的预测点索引
            matched_gt_indices: 匹配的真值点索引
        """
        if cost_matrix.numel() == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        if linear_sum_assignment is None:
            return self.greedy_matching(cost_matrix)

        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix_np)
        return row_indices, col_indices

    def greedy_matching(self, cost_matrix):
        if cost_matrix.numel() == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        num_pred, num_gt = cost_matrix.shape
        used_pred = set()
        matched_pred = []
        matched_gt = []

        for gt_idx in range(num_gt):
            best_pred = None
            best_cost = None
            for pred_idx in range(num_pred):
                if pred_idx in used_pred:
                    continue
                cost = cost_matrix[pred_idx, gt_idx].item()
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_pred = pred_idx

            if best_pred is not None:
                used_pred.add(best_pred)
                matched_pred.append(best_pred)
                matched_gt.append(gt_idx)

        return np.array(matched_pred, dtype=np.int64), np.array(matched_gt, dtype=np.int64)

    def _tensor_loss(self, pred_points, gt_points_list, image_sizes, pred_logits=None):
        batch_size, num_pred, _ = pred_points.shape
        total_loss = pred_points.new_tensor(0.0)
        match_info = []

        for i in range(batch_size):
            image_width, image_height = image_sizes[i]
            pred_norm = pred_points[i]

            gt_points = gt_points_list[i]
            if len(gt_points) == 0:
                target_mask = torch.zeros(num_pred, device=pred_points.device)
                score_loss = pred_points.new_tensor(0.0)
                if pred_logits is not None:
                    score_loss = F.binary_cross_entropy_with_logits(pred_logits[i], target_mask)
                total_loss = total_loss + score_loss
                match_info.append({
                    'pred_points': pred_norm.detach().cpu().tolist(),
                    'pred_scores': torch.sigmoid(pred_logits[i]).detach().cpu().tolist() if pred_logits is not None else [],
                    'gt_points': gt_points,
                    'matched_pred_indices': [],
                    'matched_gt_indices': [],
                    'sample_loss': float(score_loss.detach().cpu()),
                    'coordinate_mode': 'normalized_grid',
                    'image_size': (image_width, image_height)
                })
                continue

            gt_tensor = torch.tensor(gt_points, dtype=pred_points.dtype, device=pred_points.device)
            cost_matrix = torch.cdist(pred_norm, gt_tensor, p=2)
            matched_pred_indices, matched_gt_indices = self.hungarian_matching(cost_matrix)

            matched_loss = pred_points.new_tensor(0.0)
            if len(matched_pred_indices) > 0:
                pred_idx_tensor = torch.as_tensor(matched_pred_indices, dtype=torch.long, device=pred_points.device)
                gt_idx_tensor = torch.as_tensor(matched_gt_indices, dtype=torch.long, device=pred_points.device)
                matched_loss = cost_matrix[pred_idx_tensor, gt_idx_tensor].mean()

            target_mask = torch.zeros(num_pred, device=pred_points.device)
            if len(matched_pred_indices) > 0:
                target_mask[torch.as_tensor(matched_pred_indices, dtype=torch.long, device=pred_points.device)] = 1.0

            score_loss = pred_points.new_tensor(0.0)
            if pred_logits is not None:
                score_loss = F.binary_cross_entropy_with_logits(pred_logits[i], target_mask)

            unmatched_gt_loss = pred_points.new_tensor(0.0)
            if len(gt_points) > len(matched_gt_indices):
                unmatched_gt_indices = sorted(set(range(len(gt_points))) - set(matched_gt_indices))
                if unmatched_gt_indices:
                    remaining_gt = gt_tensor[torch.as_tensor(unmatched_gt_indices, dtype=torch.long, device=pred_points.device)]
                    unmatched_gt_loss = torch.cdist(pred_norm, remaining_gt, p=2).min(dim=0).values.mean()

            sample_loss = matched_loss + 0.2 * unmatched_gt_loss + 0.1 * score_loss
            total_loss = total_loss + sample_loss

            match_info.append({
                'pred_points': pred_norm.detach().cpu().tolist(),
                'pred_scores': torch.sigmoid(pred_logits[i]).detach().cpu().tolist() if pred_logits is not None else [],
                'gt_points': gt_points,
                'matched_pred_indices': matched_pred_indices.tolist(),
                'matched_gt_indices': matched_gt_indices.tolist(),
                'sample_loss': float(sample_loss.detach().cpu()),
                'coordinate_mode': 'normalized_grid',
                'image_size': (image_width, image_height)
            })

        avg_loss = total_loss / max(batch_size, 1)
        return avg_loss, match_info

    def _grid_logits_to_points(self, grid_logits, top_k=4):
        top_k = max(1, min(top_k, grid_logits.shape[-1]))
        values, indices = torch.topk(grid_logits, k=top_k, dim=-1)
        ys = torch.div(indices, self.grid_size, rounding_mode='floor')
        xs = indices % self.grid_size
        denom = float(max(self.grid_size - 1, 1))
        points = torch.stack(
            [xs.to(grid_logits.dtype) / denom, ys.to(grid_logits.dtype) / denom],
            dim=-1
        )
        return points, values

    def _grid_classification_loss(self, pred_grid_logits, grid_targets, gt_points_list=None, top_k=4):
        pos_weight = pred_grid_logits.new_full((pred_grid_logits.shape[-1],), float(self.grid_pos_weight))
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_grid_logits,
            grid_targets,
            pos_weight=pos_weight,
            reduction='none'
        )
        sample_losses = bce_loss.mean(dim=-1)
        total_loss = sample_losses.mean()

        pred_points, pred_logits = self._grid_logits_to_points(pred_grid_logits, top_k=top_k)
        pred_scores = torch.sigmoid(pred_logits)
        match_info = []

        if gt_points_list is None:
            gt_points_list = [[] for _ in range(pred_grid_logits.shape[0])]

        for batch_idx in range(pred_grid_logits.shape[0]):
            gt_points = gt_points_list[batch_idx]
            gt_tensor = (
                torch.as_tensor(
                    gt_points,
                    dtype=pred_points.dtype,
                    device=pred_points.device
                )
                if gt_points else pred_points.new_zeros((0, 2))
            )
            min_grid_distance = None
            if gt_tensor.numel() > 0:
                min_grid_distance = torch.cdist(pred_points[batch_idx], gt_tensor, p=2).min().item()

            match_info.append({
                'pred_points': pred_points[batch_idx].detach().cpu().tolist(),
                'pred_scores': pred_scores[batch_idx].detach().cpu().tolist(),
                'gt_points': gt_points,
                'sample_loss': float(sample_losses[batch_idx].detach().cpu()),
                'coordinate_mode': 'normalized_grid',
                'loss_type': 'bce_grid',
                'min_grid_distance': min_grid_distance
            })

        return total_loss, match_info

    def _text_loss(self, pred_texts, gt_points_list, image_sizes):
        """
        前向传播，计算Hungarian Loss
        
        Args:
            pred_texts: 模型预测的文本列表 [str1, str2, ...]
            gt_points_list: 真值点列表的列表 [[[gx1,gy1], ...], ...]
            image_sizes: 图像尺寸列表 [(w1,h1), (w2,h2), ...]
            
        Returns:
            loss: 平均损失值
            match_info: 匹配信息（用于分析）
        """
        batch_size = len(pred_texts)
        total_loss = 0.0
        match_info = []
        
        for i in range(batch_size):
            # 解析预测坐标
            image_width, image_height = image_sizes[i]
            pred_points = self.parse_coordinates_from_text(
                [pred_texts[i]], image_width, image_height
            )[0]  # 取出batch中的第一个（只有一个文本）
            
            gt_points = gt_points_list[i]
            
            # 计算距离矩阵
            cost_matrix = self.compute_distance_matrix(
                pred_points, gt_points, image_width, image_height
            )
            
            # Hungarian匹配
            matched_pred_indices, matched_gt_indices = self.hungarian_matching(cost_matrix)
            
            # 计算匹配的损失
            matched_loss = 0.0
            if len(matched_pred_indices) > 0:
                matched_loss = cost_matrix[
                    torch.tensor(matched_pred_indices),
                    torch.tensor(matched_gt_indices)
                ].sum()
            
            # 处理未匹配的预测点（惩罚虚假预测）
            unmatched_pred_loss = 0.0
            if len(pred_points) > len(matched_pred_indices):
                all_pred_indices = set(range(len(pred_points)))
                unmatched_pred_indices = all_pred_indices - set(matched_pred_indices)
                
                for idx in unmatched_pred_indices:
                    # 惩罚：到最近真值点的距离或到边界的距离
                    if len(gt_points) > 0:
                        pred_tensor = torch.tensor(pred_points[idx], dtype=torch.float32)
                        gt_tensor = torch.tensor(gt_points, dtype=torch.float32)
                        min_dist = torch.norm(pred_tensor.unsqueeze(0) - gt_tensor, p=2, dim=-1).min()
                        unmatched_pred_loss += min_dist
                    else:
                        # 如果没有真值点，惩罚到边界的距离
                        x, y = pred_points[idx]
                        bound_dist = min(x, image_width-x, y, image_height-y)
                        unmatched_pred_loss += bound_dist
            
            # 处理未匹配的真值点（惩罚漏检）
            unmatched_gt_loss = 0.0
            if len(gt_points) > len(matched_gt_indices):
                all_gt_indices = set(range(len(gt_points)))
                unmatched_gt_indices = all_gt_indices - set(matched_gt_indices)
                
                for idx in unmatched_gt_indices:
                    if len(pred_points) > 0:
                        # 惩罚：到最近预测点的距离
                        gt_tensor = torch.tensor(gt_points[idx], dtype=torch.float32)
                        pred_tensor = torch.tensor(pred_points, dtype=torch.float32)
                        min_dist = torch.norm(gt_tensor.unsqueeze(0) - pred_tensor, p=2, dim=-1).min()
                        unmatched_gt_loss += min_dist
                    else:
                        # 如果没有预测点，惩罚为最大可能距离
                        unmatched_gt_loss += max(image_width, image_height)
            
            # 总损失（归一化）
            num_matches = len(matched_pred_indices)
            num_elements = num_matches + len(pred_points) - num_matches + len(gt_points) - num_matches
            
            if num_elements > 0:
                sample_loss = (matched_loss + unmatched_pred_loss + unmatched_gt_loss) / num_elements
            else:
                sample_loss = 0.0
            
            total_loss += sample_loss
            
            # 记录匹配信息
            match_info.append({
                'pred_points': pred_points,
                'gt_points': gt_points,
                'matched_pred_indices': matched_pred_indices,
                'matched_gt_indices': matched_gt_indices,
                'sample_loss': sample_loss.item() if isinstance(sample_loss, torch.Tensor) else sample_loss
            })
        
        # 平均损失
        avg_loss = total_loss / batch_size
        return avg_loss, match_info

    def forward(
        self,
        pred_texts=None,
        gt_points_list=None,
        image_sizes=None,
        pred_points=None,
        pred_logits=None,
        pred_grid_logits=None,
        grid_targets=None,
        top_k=4
    ):
        if pred_grid_logits is not None and grid_targets is not None:
            return self._grid_classification_loss(
                pred_grid_logits=pred_grid_logits,
                grid_targets=grid_targets,
                gt_points_list=gt_points_list,
                top_k=top_k
            )
        if pred_points is not None:
            return self._tensor_loss(pred_points, gt_points_list, image_sizes, pred_logits)
        return self._text_loss(pred_texts, gt_points_list, image_sizes)


class HungarianPointLossPyTorch(nn.Module):
    """
    PyTorch实现的Hungarian Loss（不使用scipy）
    适用于需要完全在GPU上运行的场景
    """
    def __init__(self, 
                 inside_bbox_weight=1.0,
                 outside_bbox_weight=0.1,
                 match_cost='euclidean',
                 boundary_penalty_weight=0.1):
        super(HungarianPointLossPyTorch, self).__init__()
        
        self.inside_bbox_weight = inside_bbox_weight
        self.outside_bbox_weight = outside_bbox_weight
        self.match_cost = match_cost
        self.boundary_penalty_weight = boundary_penalty_weight
    
    def greedy_matching(self, cost_matrix):
        """
        简化的贪心匹配算法（替代Hungarian）
        计算复杂度较低，适合大规模匹配
        """
        if cost_matrix.numel() == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        
        num_pred, num_gt = cost_matrix.shape
        
        # 贪心匹配：为每个真值点选择最近的预测点
        matched_pred_indices = []
        matched_gt_indices = []
        
        # 复制代价矩阵（避免修改原始数据）
        cost_matrix_copy = cost_matrix.clone()
        
        for gt_idx in range(num_gt):
            if cost_matrix_copy.numel() == 0:
                break
            
            # 找到最小代价的匹配
            min_val, min_idx = cost_matrix_copy.min(dim=0)
            min_val, min_pred_idx = min_val.min(dim=0)
            
            if min_val < float('inf'):
                matched_pred_indices.append(min_pred_idx.item())
                matched_gt_indices.append(gt_idx)
                
                # 将该预测点标记为已匹配（设为无穷大）
                cost_matrix_copy[min_pred_idx, :] = float('inf')
        
        return torch.tensor(matched_pred_indices, dtype=torch.long), \
               torch.tensor(matched_gt_indices, dtype=torch.long)
    
    def forward(self, pred_texts, gt_points_list, image_sizes):
        """
        前向传播（同HungarianPointLoss，但使用贪心匹配）
        """
        # 实现类似forward方法，但使用greedy_matching代替hungarian_matching
        pass  # 为简洁省略，实际使用时复制HungarianPointLoss的forward并替换匹配算法


# 测试代码
if __name__ == "__main__":
    import re
    
    # 测试坐标解析
    print("=== 测试坐标解析 ===")
    
    loss_fn = HungarianPointLoss()
    
    test_texts = [
        "目标位置在[125, 240]和[300, 410]",
        "坐标是(150.5, 320.7)",
        "x: 200, y: 450",
        "没有坐标",
        "位置[100,200][300,400]"
    ]
    
    for text in test_texts:
        points = loss_fn.parse_coordinates_from_text([text], 500, 500)[0]
        print(f"Text: '{text}' -> Points: {points}")
    
    # 测试Hungarian Loss
    print("\n=== 测试Hungarian Loss ===")
    
    pred_texts = [
        "目标在[100, 100]和[200, 200]",
        "位置是[150, 150]"
    ]
    
    gt_points_list = [
        [[95, 105], [195, 205]],  # 两个真值点
        [[145, 155]]              # 一个真值点
    ]
    
    image_sizes = [(500, 500), (500, 500)]
    
    loss, match_info = loss_fn(pred_texts, gt_points_list, image_sizes)
    print(f"Loss: {loss:.4f}")
    print(f"Match info: {match_info}")
