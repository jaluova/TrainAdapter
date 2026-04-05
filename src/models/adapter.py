"""
Coordinate Adapter: 主适配器模块
整合 GridEncoder、CrossAttention、GatedFusion 和任务头。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid_encoder import GridEncoder, FeatureProjector
from .cross_attention import CrossAttention, TextGuidedCrossAttention, GatedFusion, ResidualFFN


class AttentionPool(nn.Module):
    """对 token 序列做可学习 attention pooling。"""

    def __init__(self, dim):
        super().__init__()
        self.score_proj = nn.Linear(dim, 1)

    def forward(self, features, attention_mask=None):
        scores = self.score_proj(features).squeeze(-1)
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class BaseCoordinateAdapter(nn.Module):
    """
    Coordinate Adapter 基类。
    主干负责网格引导的视觉增强；任务头支持 grid logits 和旧点回归两种模式。
    """

    def __init__(
        self,
        visual_dim=768,
        grid_feature_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_grid_tokens=64,
        num_output_points=4,
        dropout=0.1,
        output_mode='grid_logits',
        grid_size=11,
        ffn_hidden_multiplier=4
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_output_points = num_output_points
        self.output_mode = output_mode
        self.grid_size = grid_size
        self.num_grid_logits = grid_size * grid_size

        self.grid_encoder = GridEncoder(
            input_channels=3,
            feature_dim=grid_feature_dim
        )
        self.grid_projector = FeatureProjector(
            input_dim=grid_feature_dim,
            output_dim=visual_dim,
            num_tokens=num_grid_tokens
        )
        self.cross_attention = TextGuidedCrossAttention(
            dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.gated_fusion = GatedFusion(
            dim=visual_dim,
            dropout=dropout
        )
        self.residual_ffn = ResidualFFN(
            dim=visual_dim,
            hidden_dim=hidden_dim * ffn_hidden_multiplier,
            dropout=dropout
        )

        self.text_pool = AttentionPool(visual_dim)
        self.visual_condition_proj = nn.Linear(visual_dim, visual_dim)
        self.text_condition_proj = nn.Linear(visual_dim, visual_dim)
        self.visual_query_proj = nn.Linear(visual_dim, visual_dim)
        self.text_key_proj = nn.Linear(visual_dim, visual_dim)
        self.text_value_proj = nn.Linear(visual_dim, visual_dim)
        self.text_context_proj = nn.Linear(visual_dim, visual_dim)
        self.token_modulation = nn.Sequential(
            nn.LayerNorm(visual_dim * 3),
            nn.Linear(visual_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim),
            nn.Sigmoid()
        )
        self.token_text_gate = nn.Sequential(
            nn.LayerNorm(visual_dim * 2),
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim * 2)
        )
        self.token_score = nn.Linear(visual_dim, 1)
        self.token_text_score = nn.Sequential(
            nn.LayerNorm(visual_dim * 2),
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # 2D位置编码: 让grid classifier知道每个logit对应的空间位置
        # 用 sinusoidal 初始化，让模型一开始就知道空间布局
        self.grid_position_embedding = nn.Parameter(
            self._build_2d_sinusoidal_embedding(grid_size, visual_dim)
        )

        self.grid_classifier = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_grid_logits)
        )

        self.point_head = nn.Sequential(
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_output_points * 3)
        )

        self._initialize_weights()

    @staticmethod
    def _build_2d_sinusoidal_embedding(grid_size, dim):
        """构建 2D sinusoidal 位置编码 [1, grid_size^2, dim]。"""
        import math
        num_positions = grid_size * grid_size
        embedding = torch.zeros(num_positions, dim)
        quarter = dim // 4
        for idx in range(num_positions):
            y = idx // grid_size
            x = idx % grid_size
            for d in range(quarter):
                freq = 1.0 / (10000.0 ** (2.0 * d / dim))
                embedding[idx, 4 * d] = math.sin(x * freq)
                embedding[idx, 4 * d + 1] = math.cos(x * freq)
                embedding[idx, 4 * d + 2] = math.sin(y * freq)
                embedding[idx, 4 * d + 3] = math.cos(y * freq)
        return embedding.unsqueeze(0)  # [1, grid_size^2, dim]

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, images, grid_images, visual_features, text_features=None):
        grid_features_map = self.grid_encoder(grid_images)
        grid_tokens = self.grid_projector(grid_features_map)
        enhanced_features = self.cross_attention(
            visual_features=visual_features,
            grid_features=grid_tokens,
            text_features=text_features
        )
        fused_features = self.gated_fusion(visual_features, enhanced_features)
        return self.residual_ffn(fused_features)

    def _pool_text_features(self, text_features=None, attention_mask=None, visual_summary=None):
        if text_features is None:
            if visual_summary is None:
                raise ValueError("visual_summary is required when text_features is None")
            return torch.zeros_like(visual_summary)
        return self.text_pool(text_features, attention_mask=attention_mask)

    def _build_text_conditioning(self, visual_features, text_features=None, attention_mask=None):
        if text_features is None:
            return torch.zeros_like(visual_features)

        query = self.visual_query_proj(visual_features)
        keys = self.text_key_proj(text_features)
        values = self.text_value_proj(text_features)
        scale = float(self.visual_dim) ** -0.5

        attention_scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool).unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, torch.finfo(attention_scores.dtype).min)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        text_context = torch.matmul(attention_weights, values)
        return self.text_context_proj(text_context)

    def predict_grid_logits(self, visual_features, text_features=None, attention_mask=None):
        visual_summary = visual_features.mean(dim=1)
        text_summary = self._pool_text_features(
            text_features=text_features,
            attention_mask=attention_mask,
            visual_summary=visual_summary
        )
        # token-level 文本条件化：每个视觉token独立地从文本token中提取相关信息
        # 这让"左边"的信号可以选择性地增强左侧区域的token
        text_condition = self._build_text_conditioning(
            visual_features,
            text_features=text_features,
            attention_mask=attention_mask
        )

        # 融合视觉特征和token-level文本条件（不再广播池化向量）
        conditioned_tokens = (
            self.visual_condition_proj(visual_features) +
            self.text_condition_proj(text_condition) +
            text_condition
        )
        token_modulation = self.token_modulation(
            torch.cat([visual_features, text_condition, text_condition], dim=-1)
        )
        text_gate = self.token_text_gate(
            torch.cat([text_condition, text_condition], dim=-1)
        )
        gate_scale, gate_bias = torch.chunk(text_gate, chunks=2, dim=-1)
        gate_scale = 0.5 * torch.tanh(gate_scale)
        gate_bias = 0.25 * torch.tanh(gate_bias)
        conditioned_tokens = F.gelu(conditioned_tokens) * (1.0 + token_modulation)
        conditioned_tokens = conditioned_tokens * (1.0 + gate_scale) + gate_bias
        token_logits = self.grid_classifier(conditioned_tokens)
        token_weight_logits = self.token_score(conditioned_tokens).squeeze(-1)
        token_weight_logits = token_weight_logits + self.token_text_score(
            torch.cat([conditioned_tokens, text_condition], dim=-1)
        ).squeeze(-1)
        token_weights = torch.softmax(token_weight_logits, dim=1)
        grid_logits = torch.sum(token_logits * token_weights.unsqueeze(-1), dim=1)

        # 用文本特征与位置编码的交互来产生空间偏置
        # 这让模型能学到"左边"偏好小x、"右边"偏好大x等空间对应关系
        pos_bias = torch.matmul(
            text_summary.unsqueeze(1),  # [B, 1, D]
            self.grid_position_embedding.transpose(-1, -2)  # [1, D, 121]
        ).squeeze(1)  # [B, 121]
        grid_logits = grid_logits + pos_bias

        return grid_logits

    def decode_grid_logits(self, grid_logits, top_k=None):
        top_k = top_k or self.num_output_points
        top_k = max(1, min(top_k, self.num_grid_logits))

        values, indices = torch.topk(grid_logits, k=top_k, dim=-1)
        ys = torch.div(indices, self.grid_size, rounding_mode='floor')
        xs = indices % self.grid_size

        denom = float(max(self.grid_size - 1, 1))
        points = torch.stack(
            [xs.to(grid_logits.dtype) / denom, ys.to(grid_logits.dtype) / denom],
            dim=-1
        )
        return points, values

    @staticmethod
    def select_dynamic_topk(
        pred_points,
        pred_logits,
        abs_threshold=0.35,
        rel_ratio=0.75,
        min_k=1,
        max_k=6
    ):
        if pred_logits.ndim != 1:
            raise ValueError("pred_logits must be a 1D tensor")
        if pred_points.ndim != 2:
            raise ValueError("pred_points must be a 2D tensor")
        if pred_points.shape[0] != pred_logits.shape[0]:
            raise ValueError("pred_points and pred_logits must contain the same number of candidates")

        num_candidates = pred_logits.shape[0]
        if num_candidates == 0:
            empty_points = pred_points.new_zeros((0, 2))
            empty_scores = pred_logits.new_zeros((0,))
            empty_indices = torch.zeros((0,), dtype=torch.long, device=pred_logits.device)
            return {
                'selected_points': empty_points,
                'selected_logits': empty_scores,
                'selected_scores': empty_scores,
                'selected_indices': empty_indices,
                'selected_k': 0,
                'candidate_scores': empty_scores,
            }

        max_k = max(1, min(int(max_k), num_candidates))
        min_k = max(1, min(int(min_k), max_k))

        sorted_logits, sorted_indices = torch.sort(pred_logits, descending=True)
        sorted_points = pred_points[sorted_indices]
        sorted_scores = torch.sigmoid(sorted_logits)

        top1_score = sorted_scores[0]
        threshold = torch.maximum(
            sorted_scores.new_tensor(float(abs_threshold)),
            top1_score * float(rel_ratio)
        )

        keep_mask = sorted_scores >= threshold
        keep_count = int(keep_mask.sum().item())
        selected_k = max(min_k, min(max_k, keep_count if keep_count > 0 else 1))

        return {
            'selected_points': sorted_points[:selected_k],
            'selected_logits': sorted_logits[:selected_k],
            'selected_scores': sorted_scores[:selected_k],
            'selected_indices': sorted_indices[:selected_k],
            'selected_k': selected_k,
            'candidate_scores': sorted_scores,
        }

    def decode_grid_logits_dynamic(
        self,
        grid_logits,
        abs_threshold=0.35,
        rel_ratio=0.75,
        min_k=1,
        max_k=6
    ):
        candidate_points, candidate_logits = self.decode_grid_logits(
            grid_logits,
            top_k=self.num_grid_logits
        )

        selected_points = []
        selected_logits = []
        selected_scores = []
        selected_indices = []
        selected_ks = []
        candidate_scores = []

        for points, logits in zip(candidate_points, candidate_logits):
            selected = self.select_dynamic_topk(
                points,
                logits,
                abs_threshold=abs_threshold,
                rel_ratio=rel_ratio,
                min_k=min_k,
                max_k=max_k
            )
            selected_points.append(selected['selected_points'])
            selected_logits.append(selected['selected_logits'])
            selected_scores.append(selected['selected_scores'])
            selected_indices.append(selected['selected_indices'])
            selected_ks.append(selected['selected_k'])
            candidate_scores.append(selected['candidate_scores'])

        return {
            'selected_points': selected_points,
            'selected_logits': selected_logits,
            'selected_scores': selected_scores,
            'selected_indices': selected_indices,
            'selected_ks': selected_ks,
            'candidate_points': candidate_points,
            'candidate_logits': candidate_logits,
            'candidate_scores': candidate_scores,
        }

    def predict_point_regression(self, visual_features, text_features=None, attention_mask=None):
        visual_summary = visual_features.mean(dim=1)
        text_summary = self._pool_text_features(
            text_features=text_features,
            attention_mask=attention_mask,
            visual_summary=visual_summary
        )

        fused_summary = torch.cat([visual_summary, text_summary], dim=-1)
        raw_outputs = self.point_head(fused_summary)
        raw_outputs = raw_outputs.view(-1, self.num_output_points, 3)

        pred_points = torch.sigmoid(raw_outputs[..., :2])
        pred_logits = raw_outputs[..., 2]
        return pred_points, pred_logits

    def predict_points(self, visual_features, text_features=None, attention_mask=None):
        if self.output_mode == 'grid_logits':
            grid_logits = self.predict_grid_logits(
                visual_features,
                text_features=text_features,
                attention_mask=attention_mask
            )
            return self.decode_grid_logits(grid_logits, top_k=self.num_output_points)

        return self.predict_point_regression(
            visual_features,
            text_features=text_features,
            attention_mask=attention_mask
        )

    def get_trainable_parameters(self):
        return list(self.parameters())

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params
        }


class CoordinateAdapter(BaseCoordinateAdapter):
    """标准版 Coordinate Adapter。"""

    def __init__(
        self,
        visual_dim=768,
        grid_feature_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_grid_tokens=64,
        num_output_points=4,
        dropout=0.1,
        output_mode='grid_logits',
        grid_size=11
    ):
        super().__init__(
            visual_dim=visual_dim,
            grid_feature_dim=grid_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_grid_tokens=num_grid_tokens,
            num_output_points=num_output_points,
            dropout=dropout,
            output_mode=output_mode,
            grid_size=grid_size,
            ffn_hidden_multiplier=4
        )


class LightweightCoordinateAdapter(BaseCoordinateAdapter):
    """轻量级 Coordinate Adapter。"""

    def __init__(
        self,
        visual_dim=768,
        grid_feature_dim=256,
        hidden_dim=256,
        num_heads=4,
        num_grid_tokens=25,
        num_output_points=4,
        dropout=0.1,
        output_mode='grid_logits',
        grid_size=11
    ):
        super().__init__(
            visual_dim=visual_dim,
            grid_feature_dim=grid_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_grid_tokens=num_grid_tokens,
            num_output_points=num_output_points,
            dropout=dropout,
            output_mode=output_mode,
            grid_size=grid_size,
            ffn_hidden_multiplier=2
        )


if __name__ == "__main__":
    print("=== 测试CoordinateAdapter ===")

    B, C, H, W = 2, 3, 448, 448
    N, D = 196, 768
    L = 16

    images = torch.randn(B, C, H, W)
    grid_images = torch.randn(B, C, H, W)
    visual_features = torch.randn(B, N, D)
    text_features = torch.randn(B, L, D)
    attention_mask = torch.ones(B, L, dtype=torch.long)

    adapter = CoordinateAdapter(output_mode='grid_logits')
    output_features = adapter(images, grid_images, visual_features, text_features=text_features)
    grid_logits = adapter.predict_grid_logits(output_features, text_features, attention_mask)
    pred_points, pred_logits = adapter.predict_points(output_features, text_features, attention_mask)

    print(f"Output features shape: {output_features.shape}")
    print(f"Grid logits shape: {grid_logits.shape}")
    print(f"Decoded points shape: {pred_points.shape}")
    print(f"Decoded logits shape: {pred_logits.shape}")
