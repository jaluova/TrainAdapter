"""
Coordinate Adapter: 主适配器模块
整合GridEncoder、CrossAttention、GatedFusion和ResidualFFN
"""
import torch
import torch.nn as nn
from .grid_encoder import GridEncoder, FeatureProjector
from .cross_attention import CrossAttention, GatedFusion, ResidualFFN


class CoordinateAdapter(nn.Module):
    """
    Coordinate Adapter主模块
    输入: 原始图像和网格图像
    输出: 增强的视觉特征
    """
    def __init__(self, 
                 visual_dim=768,
                 grid_feature_dim=512,
                 hidden_dim=512,
                 num_heads=8,
                 num_grid_tokens=64,
                 num_output_points=4,
                 dropout=0.1):
        super(CoordinateAdapter, self).__init__()
        
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_output_points = num_output_points
        
        # 1. Grid Encoder: 从网格图像提取特征
        self.grid_encoder = GridEncoder(
            input_channels=3,
            feature_dim=grid_feature_dim
        )
        
        # 2. Feature Projector: 投影网格特征到与视觉特征相同维度
        self.grid_projector = FeatureProjector(
            input_dim=grid_feature_dim,
            output_dim=visual_dim,
            num_tokens=num_grid_tokens
        )
        
        # 3. Cross Attention: 网格特征指导视觉特征增强
        self.cross_attention = CrossAttention(
            dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 4. Gated Fusion: 自适应融合原始特征和增强特征
        self.gated_fusion = GatedFusion(
            dim=visual_dim,
            dropout=dropout
        )
        
        # 5. Residual FFN: 残差前馈网络进一步处理
        self.residual_ffn = ResidualFFN(
            dim=visual_dim,
            hidden_dim=hidden_dim * 4,
            dropout=dropout
        )

        # 6. Point head: 从融合后的视觉特征和文本特征中直接预测坐标
        self.point_head = nn.Sequential(
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_output_points * 3)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, images, grid_images, visual_features):
        """
        前向传播
        
        Args:
            images: 原始图像 [B, 3, H, W] (备用，可用于后续扩展)
            grid_images: 网格图像 [B, 3, H, W]
            visual_features: Qwen2.5-VL视觉编码器输出的视觉特征 [B, N, D]
            
        Returns:
            增强的视觉特征 [B, N, D]
        """
        B, N, D = visual_features.shape
        
        # 1. 提取网格特征
        # [B, 3, H, W] -> [B, grid_feature_dim, h, w]
        grid_features_map = self.grid_encoder(grid_images)
        
        # 2. 投影网格特征到token序列
        # [B, grid_feature_dim, h, w] -> [B, num_grid_tokens, visual_dim]
        grid_tokens = self.grid_projector(grid_features_map)
        
        # 3. Cross Attention: 网格特征指导视觉特征增强
        # visual_features [B, N, D] + grid_tokens [B, M, D] -> enhanced_features [B, N, D]
        enhanced_features = self.cross_attention(
            visual_features=visual_features,
            grid_features=grid_tokens
        )
        
        # 4. Gated Fusion: 自适应融合原始特征和增强特征
        # visual_features [B, N, D] + enhanced_features [B, N, D] -> fused_features [B, N, D]
        fused_features = self.gated_fusion(visual_features, enhanced_features)
        
        # 5. Residual FFN: 进一步处理融合特征
        # fused_features [B, N, D] -> final_features [B, N, D]
        final_features = self.residual_ffn(fused_features)
        
        return final_features

    def predict_points(self, visual_features, text_features=None):
        """
        基于增强后的视觉特征预测坐标点和置信度

        Args:
            visual_features: [B, N, D]
            text_features: [B, L, D] or None

        Returns:
            pred_points: [B, K, 2]，归一化到[0, 1]
            pred_logits: [B, K]
        """
        visual_summary = visual_features.mean(dim=1)

        if text_features is None:
            text_summary = torch.zeros_like(visual_summary)
        else:
            text_summary = text_features.mean(dim=1)

        fused_summary = torch.cat([visual_summary, text_summary], dim=-1)
        raw_outputs = self.point_head(fused_summary)
        raw_outputs = raw_outputs.view(-1, self.num_output_points, 3)

        pred_points = torch.sigmoid(raw_outputs[..., :2])
        pred_logits = raw_outputs[..., 2]
        return pred_points, pred_logits
    
    def get_trainable_parameters(self):
        """
        获取可训练参数（用于优化器）
        返回所有Adapter参数的列表
        """
        return list(self.parameters())
    
    def get_parameter_count(self):
        """获取参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params
        }


class LightweightCoordinateAdapter(nn.Module):
    """
    轻量级Coordinate Adapter
    减少参数数量，适合资源受限场景
    """
    def __init__(self, 
                 visual_dim=768,
                 grid_feature_dim=256,
                 hidden_dim=256,
                 num_heads=4,
                 num_grid_tokens=32,
                 num_output_points=4,
                 dropout=0.1):
        super(LightweightCoordinateAdapter, self).__init__()
        
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_output_points = num_output_points
        
        # 1. 轻量级Grid Encoder
        self.grid_encoder = GridEncoder(
            input_channels=3,
            feature_dim=grid_feature_dim
        )
        
        # 2. 特征投影
        self.grid_projector = FeatureProjector(
            input_dim=grid_feature_dim,
            output_dim=visual_dim,
            num_tokens=num_grid_tokens
        )
        
        # 3. 轻量级Cross Attention
        self.cross_attention = CrossAttention(
            dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 4. 简化的Gated Fusion
        self.gated_fusion = GatedFusion(
            dim=visual_dim,
            dropout=dropout
        )
        
        # 5. 轻量级Residual FFN
        self.residual_ffn = ResidualFFN(
            dim=visual_dim,
            hidden_dim=hidden_dim * 2,  # 减小隐藏层维度
            dropout=dropout
        )

        self.point_head = nn.Sequential(
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_output_points * 3)
        )
        
    def forward(self, images, grid_images, visual_features):
        """前向传播（同CoordinateAdapter）"""
        grid_features_map = self.grid_encoder(grid_images)
        grid_tokens = self.grid_projector(grid_features_map)
        enhanced_features = self.cross_attention(visual_features, grid_tokens)
        fused_features = self.gated_fusion(visual_features, enhanced_features)
        final_features = self.residual_ffn(fused_features)
        return final_features

    def predict_points(self, visual_features, text_features=None):
        visual_summary = visual_features.mean(dim=1)

        if text_features is None:
            text_summary = torch.zeros_like(visual_summary)
        else:
            text_summary = text_features.mean(dim=1)

        fused_summary = torch.cat([visual_summary, text_summary], dim=-1)
        raw_outputs = self.point_head(fused_summary)
        raw_outputs = raw_outputs.view(-1, self.num_output_points, 3)

        pred_points = torch.sigmoid(raw_outputs[..., :2])
        pred_logits = raw_outputs[..., 2]
        return pred_points, pred_logits

    def get_trainable_parameters(self):
        return list(self.parameters())

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params
        }


if __name__ == "__main__":
    import torch
    
    # 测试代码
    print("=== 测试CoordinateAdapter ===")
    
    # 模拟输入
    B, C, H, W = 2, 3, 448, 448
    N, D = 196, 768  # Qwen2.5-VL视觉特征: 14x14=196 tokens, 每个token 768维
    
    images = torch.randn(B, C, H, W)
    grid_images = torch.randn(B, C, H, W)
    visual_features = torch.randn(B, N, D)
    
    # 创建Adapter
    adapter = CoordinateAdapter(
        visual_dim=768,
        grid_feature_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_grid_tokens=64,
        dropout=0.1
    )
    
    # 前向传播
    output_features = adapter(images, grid_images, visual_features)
    
    print(f"Input visual features shape: {visual_features.shape}")
    print(f"Output features shape: {output_features.shape}")
    
    # 参数统计
    param_count = adapter.get_parameter_count()
    print(f"Total parameters: {param_count['total']:,}")
    print(f"Trainable parameters: {param_count['trainable']:,}")
    
    # 测试轻量级版本
    print("\n=== 测试LightweightCoordinateAdapter ===")
    lightweight_adapter = LightweightCoordinateAdapter()
    lightweight_output = lightweight_adapter(images, grid_images, visual_features)
    print(f"Lightweight output shape: {lightweight_output.shape}")
    
    lightweight_param_count = lightweight_adapter.get_parameter_count()
    print(f"Lightweight total parameters: {lightweight_param_count['total']:,}")
    print(f"Lightweight trainable parameters: {lightweight_param_count['trainable']:,}")
