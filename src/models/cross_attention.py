"""
Cross-Attention Module: 网格特征指导视觉特征增强
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    跨注意力机制：使用网格特征作为Key和Value，视觉特征作为Query
    实现网格特征对视觉特征的增强
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q, K, V投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, visual_features, grid_features, attention_mask=None):
        """
        前向传播
        Args:
            visual_features: 视觉特征 [B, N, D]
            grid_features: 网格特征 [B, M, D]
            attention_mask: 注意力掩码 [B, N, M] 或 None
        Returns:
            增强的视觉特征 [B, N, D]
        """
        B, N, D = visual_features.shape
        _, M, _ = grid_features.shape
        
        # 残差连接
        residual = visual_features
        
        # 投影
        Q = self.q_proj(visual_features)    # [B, N, D]
        K = self.k_proj(grid_features)      # [B, M, D]
        V = self.v_proj(grid_features)      # [B, M, D]
        
        # 重塑为多头 [B, num_heads, seq_len, head_dim]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, M, d]
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, M, d]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, N, M]
        
        # 应用掩码
        if attention_mask is not None:
            # attention_mask: [B, N, M]
            attention_mask = attention_mask.unsqueeze(1)  # [B, 1, N, M]
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, h, N, M]
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到Value
        context = torch.matmul(attn_weights, V)  # [B, h, N, d]
        
        # 重塑回原始形状 [B, N, D]
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        
        # 输出投影
        output = self.out_proj(context)
        
        # 残差连接 + LayerNorm
        output = self.norm(residual + output)
        
        return output


class TextGuidedCrossAttention(nn.Module):
    """
    文本引导的跨注意力：在视觉×网格注意力基础上，增加视觉×文本注意力分支。
    让空间特征融合阶段就能感知"红色""左边"等语义。
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.grid_cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.text_cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        # 门控：决定每个视觉token应更多地信任网格还是文本增强
        self.merge_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, visual_features, grid_features, text_features=None, text_attention_mask=None):
        # 网格分支：视觉特征通过网格特征增强（已含残差+LayerNorm）
        grid_enhanced = self.grid_cross_attn(visual_features, grid_features)

        if text_features is None:
            return grid_enhanced

        # 文本分支：视觉特征通过文本特征增强（已含残差+LayerNorm）
        text_enhanced = self.text_cross_attn(visual_features, text_features)
        # 门控融合两个分支
        gate = self.merge_gate(torch.cat([grid_enhanced, text_enhanced], dim=-1))
        return gate * text_enhanced + (1 - gate) * grid_enhanced


class GatedFusion(nn.Module):
    """
    门控自适应融合机制：自适应平衡原始特征和增强特征
    """
    def __init__(self, dim, dropout=0.1):
        super(GatedFusion, self).__init__()
        
        self.dim = dim
        
        # 门控网络
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.gate_activation = nn.Sigmoid()
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, original_features, enhanced_features):
        """
        前向传播
        Args:
            original_features: 原始视觉特征 [B, N, D]
            enhanced_features: 增强的视觉特征 [B, N, D]
        Returns:
            融合特征 [B, N, D]
        """
        # 拼接特征
        concat_features = torch.cat([original_features, enhanced_features], dim=-1)  # [B, N, 2D]
        
        # 计算门控权重
        gate = self.gate_activation(self.gate_proj(concat_features))  # [B, N, D]
        
        # 融合特征
        fused_features = gate * enhanced_features + (1 - gate) * original_features  # [B, N, D]
        
        # 输出投影
        fused_features = self.out_proj(fused_features)
        fused_features = self.dropout(fused_features)
        
        # 残差连接 + LayerNorm
        output = self.norm(original_features + fused_features)
        
        return output


class ResidualFFN(nn.Module):
    """
    残差前馈网络：进一步增强融合特征
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super(ResidualFFN, self).__init__()
        
        self.dim = dim
        hidden_dim = hidden_dim or dim * 4
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [B, N, D]
        Returns:
            增强特征 [B, N, D]
        """
        return x + self.ffn(x)


if __name__ == "__main__":
    # 测试代码
    B, N, M, D = 2, 196, 64, 768
    
    visual_features = torch.randn(B, N, D)
    grid_features = torch.randn(B, M, D)
    
    # Cross Attention
    cross_attn = CrossAttention(dim=D, num_heads=8, dropout=0.1)
    enhanced_features = cross_attn(visual_features, grid_features)
    print(f"Enhanced features shape: {enhanced_features.shape}")
    
    # Gated Fusion
    gated_fusion = GatedFusion(dim=D, dropout=0.1)
    fused_features = gated_fusion(visual_features, enhanced_features)
    print(f"Fused features shape: {fused_features.shape}")
    
    # Residual FFN
    residual_ffn = ResidualFFN(dim=D, hidden_dim=3072, dropout=0.1)
    final_features = residual_ffn(fused_features)
    print(f"Final features shape: {final_features.shape}")
