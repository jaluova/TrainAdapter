"""
主训练脚本：训练Coordinate Adapter
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# 添加src到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.adapter import CoordinateAdapter, LightweightCoordinateAdapter
from data.dataset import CoordinateDataset, collate_fn_pad_batch
from loss.hungarian_loss import HungarianPointLoss
from training.trainer import CoordinateAdapterTrainer, create_optimizer_and_scheduler
from training.config import get_config, CONFIG_PRESETS
from utils.coordinate_parser import CoordinateParser


class FrozenBackbone(nn.Module):
    """
    统一的冻结特征抽取器接口。
    如果本地没有Qwen权重，则退化为轻量级mock backbone，方便先把训练链路跑通。
    """
    def __init__(self, hidden_size=768, vocab_size=32768):
        super().__init__()
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(64, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.text_embedding = nn.Embedding(vocab_size, hidden_size)

    def encode_image(self, images):
        features = self.vision_proj(images)
        return features.flatten(2).transpose(1, 2)

    def encode_text(self, input_ids, attention_mask):
        embeddings = self.text_embedding(input_ids)
        if attention_mask is None:
            return embeddings
        return embeddings * attention_mask.unsqueeze(-1)


def setup_transforms(config):
    """
    设置数据变换
    
    Args:
        config: 配置对象
        
    Returns:
        train_transform, val_transform
    """
    if config.data.use_data_augmentation:
        train_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ) if config.data.color_jitter else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # 验证集变换（不使用数据增强）
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def load_qwen_model(model_path, device):
    """
    加载Qwen2.5-VL模型
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        qwen_model, tokenizer
    """
    print(f"Loading Qwen2.5-VL from {model_path}")
    
    # 加载模型（简化版本，实际使用时需要完整加载）
    # 注意：这里需要根据Qwen2.5-VL的实际结构进行调整
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model path does not exist: {model_path}")

        target_dtype = torch.float16
        if getattr(device, 'type', str(device)) == 'cpu':
            target_dtype = torch.float32

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=target_dtype,
            device_map=None,
            local_files_only=True
        )
        
        processor = AutoProcessor.from_pretrained(model_path)

        class QwenBackboneWrapper(nn.Module):
            def __init__(self, qwen_model, qwen_processor):
                super().__init__()
                self.qwen_model = qwen_model
                self.qwen_processor = qwen_processor
                self.visual_dim = qwen_model.config.hidden_size
                self.text_dim = qwen_model.config.hidden_size
                self.merge_size = getattr(qwen_processor.image_processor, 'merge_size', 1)

            def encode_image(self, images):
                if hasattr(self.qwen_model, 'visual'):
                    pil_images = [self._tensor_to_pil(image) for image in images]
                    image_inputs = self.qwen_processor.image_processor(
                        pil_images,
                        return_tensors='pt'
                    )
                    pixel_values = image_inputs['pixel_values'].to(self.qwen_model.device)
                    image_grid_thw = image_inputs['image_grid_thw'].to(self.qwen_model.device)
                    outputs = self.qwen_model.visual(pixel_values, image_grid_thw)
                    return self._pack_visual_outputs(outputs, image_grid_thw)
                raise AttributeError("Qwen model does not expose a supported visual encoder")

            def encode_text(self, input_ids, attention_mask):
                if hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'embed_tokens'):
                    return self.qwen_model.model.embed_tokens(input_ids)
                if hasattr(self.qwen_model, 'get_input_embeddings'):
                    return self.qwen_model.get_input_embeddings()(input_ids)
                raise AttributeError("Qwen model does not expose a supported text embedding layer")

            def _tensor_to_pil(self, image_tensor):
                image = image_tensor.detach().float().cpu()

                # 尝试反归一化回接近原始 RGB，兼容当前数据增强配置
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
                image = image.clamp(0, 1)

                image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                return Image.fromarray(image)

            def _pack_visual_outputs(self, outputs, image_grid_thw):
                token_counts = (
                    image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
                ) // (self.merge_size ** 2)
                token_counts = token_counts.tolist()

                chunks = []
                start = 0
                for count in token_counts:
                    end = start + count
                    chunks.append(outputs[start:end])
                    start = end

                max_tokens = max(token_counts)
                padded = outputs.new_zeros((len(chunks), max_tokens, outputs.shape[-1]))
                for idx, chunk in enumerate(chunks):
                    padded[idx, :chunk.shape[0]] = chunk
                return padded

        print("Qwen2.5-VL loaded successfully")
        return QwenBackboneWrapper(model, processor).to(device), processor.tokenizer
        
    except Exception as e:
        print(f"Warning: Failed to load Qwen2.5-VL: {e}")
        print("Creating lightweight frozen backbone for smoke tests")
        return FrozenBackbone().to(device), None


def create_adapter(config):
    """
    创建Coordinate Adapter
    
    Args:
        config: 配置对象
        
    Returns:
        adapter
    """
    if config.model.adapter_type == 'lightweight':
        adapter = LightweightCoordinateAdapter(
            visual_dim=config.model.visual_dim,
            grid_feature_dim=config.model.grid_feature_dim,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_grid_tokens=config.model.num_grid_tokens,
            num_output_points=config.model.num_output_points,
            dropout=config.model.dropout
        )
    else:
        adapter = CoordinateAdapter(
            visual_dim=config.model.visual_dim,
            grid_feature_dim=config.model.grid_feature_dim,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_grid_tokens=config.model.num_grid_tokens,
            num_output_points=config.model.num_output_points,
            dropout=config.model.dropout
        )
    
    return adapter


def create_dataloaders(config, train_transform, val_transform):
    """
    创建数据加载器
    
    Args:
        config: 配置对象
        train_transform: 训练变换
        val_transform: 验证变换
        
    Returns:
        train_dataloader, val_dataloader
    """
    # 训练数据集
    train_dataset = CoordinateDataset(
        data_root=config.data.data_root,
        annotation_file=config.data.annotation_file,
        image_dir=config.data.image_dir,
        grid_image_dir=config.data.grid_image_dir,
        tokenizer_path=config.model.qwen_model_path,
        image_size=config.data.image_size,
        max_length=config.data.max_length,
        transform=train_transform,
        num_output_points=config.model.num_output_points,
        target_point_strategy=config.data.target_point_strategy,
        target_coordinate_mode=config.data.target_coordinate_mode
    )
    
    # 验证数据集（如果有验证集）
    val_dataset = None
    val_dataloader = None
    
    if os.path.exists(os.path.join(config.data.data_root, 'val_grefs_with_grids.json')):
        val_dataset = CoordinateDataset(
            data_root=config.data.data_root,
            annotation_file='val_grefs_with_grids.json',
            image_dir=config.data.image_dir,
            grid_image_dir=config.data.grid_image_dir,
            tokenizer_path=config.model.qwen_model_path,
            image_size=config.data.image_size,
            max_length=config.data.max_length,
            transform=val_transform,
            num_output_points=config.model.num_output_points,
            target_point_strategy=config.data.target_point_strategy,
            target_coordinate_mode=config.data.target_coordinate_mode
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_pad_batch
        )
    
    # 训练数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_pad_batch
    )
    
    return train_dataloader, val_dataloader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train Coordinate Adapter')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--preset', type=str, default='default', 
                       choices=list(CONFIG_PRESETS.keys()), help='预置配置')
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复')
    
    # 模型参数
    parser.add_argument('--adapter_type', type=str, choices=['standard', 'lightweight'], help='Adapter类型')
    parser.add_argument('--grid_feature_dim', type=int, help='网格特征维度')
    parser.add_argument('--hidden_dim', type=int, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, help='注意力头数')
    parser.add_argument('--num_grid_tokens', type=int, help='网格token数量')
    parser.add_argument('--num_output_points', type=int, help='每张图预测的坐标点数量')
    parser.add_argument('--dropout', type=float, help='Dropout率')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--annotation_file', type=str, help='标注文件')
    parser.add_argument('--batch_size', type=int, help='Batch大小')
    parser.add_argument('--use_negative_samples', action='store_true', help='使用负样本')
    
    # 训练参数
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--weight_decay', type=float, help='权重衰减')
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='梯度累积步数')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        config = get_config().load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = get_config(args.preset)
        print(f"Using {args.preset} preset")
    
    # 从命令行参数更新配置
    arg_updates = {
        'model.adapter_type': args.adapter_type,
        'model.grid_feature_dim': args.grid_feature_dim,
        'model.hidden_dim': args.hidden_dim,
        'model.num_heads': args.num_heads,
        'model.num_grid_tokens': args.num_grid_tokens,
        'model.num_output_points': args.num_output_points,
        'model.dropout': args.dropout,
        'data.data_root': args.data_root,
        'data.annotation_file': args.annotation_file,
        'training.batch_size': args.batch_size,
        'training.lr': args.lr,
        'training.weight_decay': args.weight_decay,
        'training.num_epochs': args.num_epochs,
        'training.gradient_accumulation_steps': args.gradient_accumulation_steps
    }
    config.update_from_args(arg_updates)
    
    # 设置保存目录
    if args.save_dir:
        config.logging.save_dir = args.save_dir
    
    # 设置设备
    if args.device:
        config.device = args.device
    
    # 设置恢复检查点
    if args.resume:
        config.resume_from = args.resume
    
    # 打印配置
    print("=" * 50)
    print("Training Configuration:")
    print("=" * 50)
    print(f"Adapter type: {config.model.adapter_type}")
    print(f"Grid feature dim: {config.model.grid_feature_dim}")
    print(f"Hidden dim: {config.model.hidden_dim}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.lr}")
    print(f"Num epochs: {config.training.num_epochs}")
    print(f"Target strategy: {config.data.target_point_strategy}")
    print(f"Coordinate mode: {config.data.target_coordinate_mode}")
    print(f"Save dir: {config.logging.save_dir}")
    print(f"Device: {config.device}")
    print("=" * 50)
    
    # 创建保存目录
    os.makedirs(config.logging.save_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(config.logging.save_dir, 'config.json')
    config.save(config_path)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载Qwen2.5-VL模型
    qwen_model, tokenizer = load_qwen_model(config.model.qwen_model_path, device)
    if hasattr(qwen_model, 'visual_dim'):
        config.model.visual_dim = qwen_model.visual_dim

    # 创建Coordinate Adapter
    adapter = create_adapter(config)
    adapter.to(device)
    
    print(f"Adapter parameters: {adapter.get_parameter_count()}")
    
    # 设置数据变换
    train_transform, val_transform = setup_transforms(config)
    
    # 创建数据加载器
    train_dataloader, val_dataloader = create_dataloaders(
        config, train_transform, val_transform
    )
    
    print(f"Train dataset size: {len(train_dataloader.dataset)}")
    if val_dataloader:
        print(f"Val dataset size: {len(val_dataloader.dataset)}")
    
    # 创建损失函数
    loss_fn = HungarianPointLoss(
        inside_bbox_weight=config.training.inside_bbox_weight,
        outside_bbox_weight=config.training.outside_bbox_weight,
        match_cost=config.training.match_cost,
        boundary_penalty_weight=config.training.boundary_penalty_weight
    )
    
    # 创建优化器和调度器
    total_steps = len(train_dataloader) * config.training.num_epochs
    train_config_dict = dict(config.training.__dict__)
    train_config_dict['total_steps'] = total_steps

    optimizer, scheduler = create_optimizer_and_scheduler(
        adapter, 
        train_config_dict
    )
    
    # 创建训练器
    trainer = CoordinateAdapterTrainer(
        adapter=adapter,
        qwen_model=qwen_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        save_dir=config.logging.save_dir,
        max_grad_norm=config.training.max_grad_norm,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_interval=config.logging.log_interval,
        eval_interval=config.logging.eval_interval,
        save_interval=config.logging.save_interval
    )
    
    # 开始训练
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50 + "\n")
    
    trainer.train(
        num_epochs=config.training.num_epochs,
        resume_from=config.resume_from
    )
    
    print("\n" + "=" * 50)
    print("Training Completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
