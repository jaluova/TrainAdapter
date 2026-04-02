"""
训练配置文件
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _default_data_root():
    return os.environ.get('TRAIN_ADAPTER_DATA_ROOT', os.path.join(_project_root(), 'Data'))


def _default_model_path():
    return os.environ.get(
        'TRAIN_ADAPTER_QWEN_PATH',
        os.path.join(_project_root(), 'Qwen2.5-VL-7B-Instruct')
    )


def _default_save_dir():
    return os.environ.get('TRAIN_ADAPTER_SAVE_DIR', os.path.join(_default_data_root(), 'train_outputs'))


@dataclass
class ModelConfig:
    """模型配置"""
    # Adapter配置
    adapter_type: str = 'standard'  # 'standard' or 'lightweight'
    visual_dim: int = 768
    grid_feature_dim: int = 512
    hidden_dim: int = 512
    num_heads: int = 8
    num_grid_tokens: int = 64
    dropout: float = 0.1
    
    # Qwen2.5-VL配置
    qwen_model_path: str = field(default_factory=_default_model_path)
    freeze_qwen: bool = True
    num_output_points: int = 4
    output_mode: str = 'grid_logits'  # 'grid_logits' or 'point_regression'
    grid_size: int = 11


@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = field(default_factory=_default_data_root)
    annotation_file: str = 'grefs_with_grids.json'
    image_dir: str = 'images'
    grid_image_dir: str = 'grid_images'
    image_size: tuple = (448, 448)
    max_length: int = 512
    
    # 数据增强
    use_data_augmentation: bool = True
    color_jitter: bool = True
    random_resized_crop: bool = True
    horizontal_flip: bool = False  # 坐标会变化，慎用

    # 监督目标
    target_point_strategy: str = 'fps'
    target_coordinate_mode: str = 'normalized_grid'
    split_by_image: bool = True
    val_ratio: float = 0.2
    relation_query_oversample: bool = True
    relation_query_weight: float = 1.2
    relation_keywords: tuple = (
        'left', 'right', 'top', 'bottom', 'front', 'behind', 'between', 'with', 'and',
        'center', 'middle', 'near', 'nearest', 'closest', 'far', 'furthest',
        'first', 'second', 'third', 'fourth', 'last'
    )
    ordinal_keywords: tuple = (
        'first', 'second', 'third', 'fourth', 'fifth', 'last',
        'leftmost', 'rightmost', 'furthest', 'nearest'
    )
    multi_entity_keywords: tuple = (
        ' and ', ' with ', ' between ', ' beside ', ' next to '
    )
    difficulty_oversample: bool = True
    spatial_query_weight: float = 1.8
    ordinal_query_weight: float = 2.2
    multi_entity_query_weight: float = 2.0
    multi_query_image_weight: float = 1.3
    
    # 负样本
    use_negative_samples: bool = False
    negative_sample_ratio: float = 0.2


@dataclass
class TrainingConfig:
    """训练配置"""
    # 优化器
    lr: float = 3e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    
    # 训练参数
    batch_size: int = 8
    num_epochs: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 学习率调度
    use_scheduler: bool = True
    warmup_steps: int = 200
    warmup_ratio: float = 0.05
    scheduler_type: str = 'cosine'  # 'cosine' or 'linear'
    early_stop_patience_evals: int = 3
    
    # Loss配置
    loss_type: str = 'bce_grid'
    inside_bbox_weight: float = 1.0
    outside_bbox_weight: float = 0.1
    match_cost: str = 'euclidean'  # 'euclidean', 'l1', 'smooth_l1'
    boundary_penalty_weight: float = 0.1
    grid_pos_weight: float = 5.0
    neighbor_soft_label_weight: float = 0.15
    ranking_margin: float = 0.2
    ranking_loss_weight: float = 0.2
    use_primary_grid_target: bool = True
    use_amp: bool = False


@dataclass
class LoggingConfig:
    """日志配置"""
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 500
    preview_interval: int = 0
    save_dir: str = field(default_factory=_default_save_dir)
    
    # WandB配置（可选）
    use_wandb: bool = False
    wandb_project: str = 'coordinate-adapter'
    wandb_name: Optional[str] = None


@dataclass
class Config:
    """主配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 设备配置
    device: str = 'cuda'
    seed: int = 42
    
    # 恢复训练
    resume_from: Optional[str] = None
    resume_as_init: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'logging': self.logging.__dict__,
            'device': self.device,
            'seed': self.seed,
            'resume_from': self.resume_from,
            'resume_as_init': self.resume_as_init
        }
    
    def save(self, save_path: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Config saved to {save_path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典加载配置"""
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        
        config.device = config_dict.get('device', 'cuda')
        config.seed = config_dict.get('seed', 42)
        config.resume_from = config_dict.get('resume_from')
        config.resume_as_init = config_dict.get('resume_as_init', False)
        
        return config
    
    @classmethod
    def load(cls, load_path: str):
        """从文件加载配置"""
        with open(load_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update_from_args(self, args: Dict[str, Any]):
        """从命令行参数更新配置"""
        for key, value in args.items():
            if value is not None:
                if key.startswith('model.'):
                    setattr(self.model, key[6:], value)
                elif key.startswith('data.'):
                    setattr(self.data, key[5:], value)
                elif key.startswith('training.'):
                    setattr(self.training, key[9:], value)
                elif key.startswith('logging.'):
                    setattr(self.logging, key[8:], value)
                else:
                    setattr(self, key, value)


def get_default_config():
    """获取默认配置"""
    config = Config()
    config.model.adapter_type = 'lightweight'
    config.model.grid_feature_dim = 256
    config.model.hidden_dim = 256
    config.model.num_heads = 4
    config.model.num_grid_tokens = 25
    config.model.output_mode = 'grid_logits'

    config.data.target_point_strategy = 'all'
    config.data.split_by_image = True
    config.data.val_ratio = 0.2
    config.data.relation_query_oversample = True
    config.data.relation_query_weight = 1.2

    config.training.batch_size = 8
    config.training.gradient_accumulation_steps = 1
    config.training.lr = 3e-5
    config.training.num_epochs = 4
    config.training.loss_type = 'bce_grid'
    config.training.grid_pos_weight = 5.0
    config.training.neighbor_soft_label_weight = 0.15
    config.training.ranking_margin = 0.2
    config.training.ranking_loss_weight = 0.2
    config.training.use_primary_grid_target = True
    config.training.use_amp = False
    config.training.early_stop_patience_evals = 3
    config.logging.eval_interval = 500
    config.logging.save_interval = 500
    config.logging.preview_interval = 0
    return config


def get_lightweight_config():
    """获取轻量级配置（适合资源受限场景）"""
    config = get_default_config()
    
    # 模型配置
    config.model.adapter_type = 'lightweight'
    config.model.grid_feature_dim = 256
    config.model.hidden_dim = 256
    config.model.num_heads = 4
    config.model.num_grid_tokens = 25
    
    return config


def get_high_performance_config():
    """获取高性能配置（追求最佳效果）"""
    config = get_default_config()
    
    # 模型配置
    config.model.adapter_type = 'standard'
    config.model.grid_feature_dim = 512
    config.model.hidden_dim = 512
    config.model.num_heads = 8
    config.model.num_grid_tokens = 64
    
    # 数据配置
    config.data.use_data_augmentation = True
    config.data.use_negative_samples = False
    config.data.negative_sample_ratio = 0.3
    
    # 训练配置
    config.training.batch_size = 8
    config.training.num_epochs = 16
    config.training.lr = 5e-5
    config.training.warmup_steps = 400
    
    return config


# 预定义配置字典
CONFIG_PRESETS = {
    'default': get_default_config,
    'lightweight': get_lightweight_config,
    'high_performance': get_high_performance_config
}


def get_config(preset='default', **kwargs):
    """
    获取配置
    
    Args:
        preset: 预置配置名称
        **kwargs: 额外配置参数
        
    Returns:
        配置对象
    """
    if preset in CONFIG_PRESETS:
        config = CONFIG_PRESETS[preset]()
    else:
        config = get_default_config()
        print(f"Unknown preset '{preset}', using default config")
    
    # 更新配置
    config.update_from_args(kwargs)
    
    return config


if __name__ == "__main__":
    # 测试配置
    print("=== 测试默认配置 ===")
    config = get_default_config()
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    
    print("\n=== 测试轻量级配置 ===")
    lightweight_config = get_lightweight_config()
    print(f"Batch size: {lightweight_config.training.batch_size}")
    print(f"Hidden dim: {lightweight_config.model.hidden_dim}")
    
    print("\n=== 测试高性能配置 ===")
    hp_config = get_high_performance_config()
    print(f"Batch size: {hp_config.training.batch_size}")
    print(f"Num epochs: {hp_config.training.num_epochs}")
    
    # 保存配置
    config.save('test_config.json')
    
    # 加载配置
    loaded_config = Config.load('test_config.json')
    print(f"\nLoaded config device: {loaded_config.device}")
