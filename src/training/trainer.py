"""
训练器：负责训练Coordinate Adapter
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import json
import logging
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


class CoordinateAdapterTrainer:
    """
    Coordinate Adapter训练器
    """
    def __init__(self, 
                 adapter,
                 qwen_model,
                 train_dataloader,
                 val_dataloader=None,
                 optimizer=None,
                 scheduler=None,
                 loss_fn=None,
                 device='cuda',
                 save_dir='outputs',
                 max_grad_norm=1.0,
                 gradient_accumulation_steps=1,
                 log_interval=10,
                 eval_interval=500,
                 save_interval=1000):
        """
        Args:
            adapter: Coordinate Adapter模型
            qwen_model: Qwen2.5-VL模型（冻结）
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_fn: 损失函数
            device: 设备
            save_dir: 保存目录
            max_grad_norm: 梯度裁剪阈值
            gradient_accumulation_steps: 梯度累积步数
            log_interval: 日志间隔
            eval_interval: 验证间隔
            save_interval: 保存间隔
        """
        self.adapter = adapter.to(device)
        self.qwen_model = qwen_model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'train_outputs'), exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 记录模型信息
        self.logger.info(f"Adapter parameters: {self.adapter.get_parameter_count()}")
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_acc_5 = -1.0
        self.qualitative_top_k = min(4, getattr(self.adapter, 'num_output_points', 4))
        self.qualitative_panel_indices = self._select_qualitative_indices()
        
        # 冻结Qwen模型
        self._freeze_qwen_model()
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(
            self.save_dir, 
            'logs', 
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _freeze_qwen_model(self):
        """冻结Qwen2.5-VL模型参数"""
        if hasattr(self.qwen_model, 'parameters'):
            for param in self.qwen_model.parameters():
                param.requires_grad = False
        
        self.qwen_model.eval()
        self.logger.info("Qwen2.5-VL model frozen and set to eval mode")
    
    def save_checkpoint(self, step, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'best_loss': self.best_loss,
            'best_acc_5': self.best_acc_5
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'checkpoints', f'checkpoint_step_{step}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at step {step} with loss {loss:.4f}")
        
        self.logger.info(f"Saved checkpoint at step {step}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.adapter.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.best_acc_5 = checkpoint.get('best_acc_5', self.best_acc_5)
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _select_qualitative_indices(self, count=6):
        """固定一组验证样本，避免每次评估都换图。"""
        if self.val_dataloader is None or not hasattr(self.val_dataloader, 'dataset'):
            return []

        dataset = self.val_dataloader.dataset
        if len(dataset) == 0:
            return []

        count = min(count, len(dataset))
        if count == len(dataset):
            return list(range(len(dataset)))
        if count == 1:
            return [0]

        step = (len(dataset) - 1) / float(count - 1)
        return sorted({round(i * step) for i in range(count)})

    def _load_font(self, size, bold=False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    pass
        return ImageFont.load_default()

    def _rank_predictions(self, pred_points, pred_scores, top_k=None):
        top_k = self.qualitative_top_k if top_k is None else top_k
        top_k = max(1, min(top_k, len(pred_points)))
        scored = list(zip(pred_points, pred_scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _normalized_points_to_grid_pixels(self, points, image_size, border_size=28):
        width, height = image_size
        pixel_points = []
        for point in points:
            x = border_size + float(point[0]) * width
            y = border_size + float(point[1]) * height
            pixel_points.append((x, y))
        return pixel_points

    def _draw_point(self, draw, point, color, label, radius=8, fill=True):
        x, y = point
        bbox = [x - radius, y - radius, x + radius, y + radius]
        if fill:
            draw.ellipse(bbox, fill=color, outline="white", width=2)
        else:
            draw.ellipse(bbox, outline=color, width=3)
        font = self._load_font(18, bold=True)
        draw.text((x + radius + 3, y - radius - 3), label, fill=color, font=font)

    def _save_qualitative_panel(self, step):
        if self.val_dataloader is None or not self.qualitative_panel_indices:
            return None

        from data.dataset import collate_fn_pad_batch

        dataset = self.val_dataloader.dataset
        panel_dir = os.path.join(self.save_dir, 'qualitative_panels', f'step_{step:06d}')
        os.makedirs(panel_dir, exist_ok=True)

        manifest = {
            'step': step,
            'epoch': self.epoch,
            'indices': self.qualitative_panel_indices,
            'coordinate_mode': 'normalized_grid',
            'top_k': self.qualitative_top_k,
        }

        title_font = self._load_font(26, bold=True)
        body_font = self._load_font(18)
        small_font = self._load_font(16)

        with torch.no_grad():
            for sample_idx in self.qualitative_panel_indices:
                item = dataset[sample_idx]
                batch = collate_fn_pad_batch([item])
                pred_points, pred_logits = self.forward_batch(batch)
                pred_points = pred_points[0].detach().cpu().tolist()
                pred_scores = torch.sigmoid(pred_logits[0]).detach().cpu().tolist()
                ranked = self._rank_predictions(pred_points, pred_scores)

                image_id = dataset.samples[sample_idx]['image_id']
                image_size = item['image_size']
                original_path = os.path.join(dataset.data_root, dataset.image_dir, image_id)
                grid_path = os.path.join(dataset.data_root, dataset.grid_image_dir, os.path.basename(dataset.samples[sample_idx]['grid_image_path']))

                original_img = Image.open(original_path).convert('RGB')
                if os.path.exists(grid_path):
                    grid_img = Image.open(grid_path).convert('RGB')
                else:
                    grid_img = Image.new('RGB', (original_img.width + 56, original_img.height + 56), 'white')
                    grid_img.paste(original_img, (28, 28))

                gt_pixel_points = self._normalized_points_to_grid_pixels(item['gt_points'], image_size)
                pred_pixel_points = self._normalized_points_to_grid_pixels([point for point, _ in ranked], image_size)

                panel_gap = 30
                header_h = 140
                footer_h = 130
                canvas_w = original_img.width + grid_img.width + panel_gap * 3
                canvas_h = max(original_img.height, grid_img.height) + header_h + footer_h
                canvas = Image.new('RGB', (canvas_w, canvas_h), '#f5f7fb')
                draw = ImageDraw.Draw(canvas)

                draw.text((30, 20), "Validation Qualitative Panel", fill="#18212f", font=title_font)
                draw.text((30, 56), f"Image: {image_id}", fill="#334155", font=body_font)
                draw.text((30, 84), f"Query: {item['query']}", fill="#1f2937", font=body_font)

                left_x = panel_gap
                top_y = header_h
                right_x = left_x + original_img.width + panel_gap
                canvas.paste(original_img, (left_x, top_y))
                canvas.paste(grid_img, (right_x, top_y))

                draw.text((left_x, top_y - 28), "Original Image", fill="#1f2937", font=body_font)
                draw.text((right_x, top_y - 28), "Grid Image + GT / Predictions", fill="#1f2937", font=body_font)

                overlay = ImageDraw.Draw(canvas)
                for idx, point in enumerate(gt_pixel_points, start=1):
                    self._draw_point(overlay, (right_x + point[0], top_y + point[1]), "#1d4ed8", f"G{idx}", radius=7, fill=True)

                for idx, ((point_x, point_y), (_, score)) in enumerate(zip(pred_pixel_points, ranked), start=1):
                    self._draw_point(overlay, (right_x + point_x, top_y + point_y), "#dc2626", f"P{idx}", radius=10, fill=False)
                    overlay.text((right_x + point_x + 16, top_y + point_y + 10), f"{score:.2f}", fill="#b91c1c", font=small_font)

                footer_y = top_y + max(original_img.height, grid_img.height) + 24
                draw.rounded_rectangle([30, footer_y, canvas_w - 30, canvas_h - 24], radius=18, fill="white", outline="#d7deea", width=2)
                draw.text((50, footer_y + 18), "Readout", fill="#162033", font=body_font)
                draw.text((50, footer_y + 50), "Blue dots are sampled supervision points; red circles are top confidence predictions.", fill="#334155", font=small_font)
                draw.text((50, footer_y + 74), "Scores come from sigmoid(pred_logits); training and evaluation both operate in normalized grid coordinates.", fill="#334155", font=small_font)

                output_prefix = os.path.join(panel_dir, f"sample_{sample_idx:04d}_{image_id}")
                canvas.save(f"{output_prefix}.png")
                with open(f"{output_prefix}.json", 'w', encoding='utf-8') as f:
                    json.dump({
                        'image_id': image_id,
                        'query': item['query'],
                        'image_size': list(image_size),
                        'ground_truth_points_normalized': item['gt_points'],
                        'predicted_points_normalized': [[round(float(x), 4), round(float(y), 4)] for x, y in [point for point, _ in ranked]],
                        'prediction_scores': [round(float(score), 4) for _, score in ranked],
                    }, f, ensure_ascii=False, indent=2)

        with open(os.path.join(panel_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return panel_dir
    
    def forward_batch(self, batch):
        """
        前向计算：提取冻结特征，经过Adapter预测坐标点
        
        Args:
            batch: 批次数据
            
        Returns:
            pred_points: [B, K, 2]
            pred_logits: [B, K]
        """
        # 提取批次数据
        images = batch['image'].to(self.device)
        grid_images = batch['grid_image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        batch_size = images.shape[0]
        adapter_dtype = next(self.adapter.parameters()).dtype
        
        # 1. 视觉编码（冻结）
        with torch.no_grad():
            visual_features = self.qwen_model.encode_image(images)
            if visual_features.dtype != adapter_dtype:
                visual_features = visual_features.to(dtype=adapter_dtype)
        
        # 2. Adapter增强（可训练）
        enhanced_features = self.adapter(images, grid_images, visual_features)
        
        # 3. 文本编码（冻结）
        with torch.no_grad():
            text_embeddings = self.qwen_model.encode_text(input_ids, attention_mask)
            if text_embeddings.dtype != adapter_dtype:
                text_embeddings = text_embeddings.to(dtype=adapter_dtype)

        pred_points, pred_logits = self.adapter.predict_points(enhanced_features, text_embeddings)
        return pred_points, pred_logits
    
    def train_step(self, batch):
        """
        单步训练
        
        Args:
            batch: 批次数据
            
        Returns:
            loss: 损失值
        """
        # 设置为训练模式
        self.adapter.train()
        
        pred_points, pred_logits = self.forward_batch(batch)
        
        # 获取真值数据
        gt_points_list = batch['gt_points']
        image_sizes = batch['image_size']
        
        # 计算损失
        loss, match_info = self.loss_fn(
            pred_points=pred_points,
            pred_logits=pred_logits,
            gt_points_list=gt_points_list,
            image_sizes=image_sizes
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.adapter.parameters(), 
                    self.max_grad_norm
                )
            
            # 更新参数
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item(), match_info
    
    def evaluate(self):
        """
        验证模型
        
        Returns:
            avg_loss: 平均损失
            metrics: 评估指标
        """
        if self.val_dataloader is None:
            return None, None
        
        self.adapter.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_match_info = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc='Evaluating'):
                pred_points, pred_logits = self.forward_batch(batch)
                
                # 获取真值数据
                gt_points_list = batch['gt_points']
                image_sizes = batch['image_size']
                
                # 计算损失
                loss, match_info = self.loss_fn(
                    pred_points=pred_points,
                    pred_logits=pred_logits,
                    gt_points_list=gt_points_list,
                    image_sizes=image_sizes
                )
                
                total_loss += loss.item()
                total_samples += len(batch['image'])
                all_match_info.extend(match_info)
        
        avg_loss = total_loss / len(self.val_dataloader)
        
        # 计算评估指标
        metrics = self._compute_metrics(all_match_info)
        panel_dir = self._save_qualitative_panel(self.global_step)
        if panel_dir:
            metrics['qualitative_panel_dir'] = panel_dir
        
        return avg_loss, metrics
    
    def _compute_metrics(self, match_info):
        """
        计算评估指标
        
        Args:
            match_info: 匹配信息列表
            
        Returns:
            metrics: 指标字典
        """
        total_l1_error = 0.0
        total_samples = 0
        acc_5 = 0
        acc_10 = 0
        
        for info in match_info:
            pred_points = info['pred_points']
            gt_points = info['gt_points']
            pred_scores = info.get('pred_scores', [1.0] * len(pred_points))
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                continue

            ranked_points = [point for point, _ in self._rank_predictions(pred_points, pred_scores)]
            
            # 计算最近距离
            for gt_point in gt_points:
                min_dist = float('inf')
                for pred_point in ranked_points:
                    dist = np.linalg.norm(np.array(pred_point) - np.array(gt_point))
                    min_dist = min(min_dist, dist)
                
                total_l1_error += min_dist
                total_samples += 1
                
                # 计算准确率
                if min_dist < 0.05:
                    acc_5 += 1
                if min_dist < 0.1:
                    acc_10 += 1
        
        metrics = {
            'l1_error': total_l1_error / total_samples if total_samples > 0 else 0.0,
            'acc_5': acc_5 / total_samples if total_samples > 0 else 0.0,
            'acc_10': acc_10 / total_samples if total_samples > 0 else 0.0,
            'total_samples': total_samples
        }
        
        return metrics
    
    def train(self, num_epochs, resume_from=None):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            resume_from: 从检查点恢复训练
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info(f"Start training for {num_epochs} epochs")
        self.optimizer.zero_grad(set_to_none=True)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            # 训练
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f'Training Epoch {epoch + 1}')):
                try:
                    loss, match_info = self.train_step(batch)
                    
                    epoch_loss += loss
                    num_batches += 1
                    self.global_step += 1
                    
                    # 日志
                    if self.global_step % self.log_interval == 0:
                        self.logger.info(
                            f"Step {self.global_step}, Loss: {loss:.4f}, "
                            f"Avg Loss: {epoch_loss / num_batches:.4f}"
                        )
                    
                    # 验证
                    if self.val_dataloader and self.global_step % self.eval_interval == 0:
                        val_loss, metrics = self.evaluate()
                        if val_loss is not None:
                            self.logger.info(
                                f"Validation - Loss: {val_loss:.4f}, "
                                f"L1 Error: {metrics['l1_error']:.4f}, "
                                f"Acc@5: {metrics['acc_5']:.2%}, "
                                f"Acc@10: {metrics['acc_10']:.2%}"
                            )
                            if metrics.get('qualitative_panel_dir'):
                                self.logger.info(f"Saved qualitative panel to {metrics['qualitative_panel_dir']}")
                            
                            # 保存最佳模型
                            if (
                                metrics['acc_5'] > self.best_acc_5 or
                                (metrics['acc_5'] == self.best_acc_5 and val_loss < self.best_loss)
                            ):
                                self.best_acc_5 = metrics['acc_5']
                                self.best_loss = val_loss
                                self.save_checkpoint(self.global_step, val_loss, is_best=True)
                    
                    # 保存检查点
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint(self.global_step, loss)
                
                except Exception as e:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.logger.error(f"Error at step {self.global_step}: {str(e)}")
                    continue
            
            #  epoch结束
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.logger.info(f"Epoch {epoch + 1} completed, Avg Loss: {avg_epoch_loss:.4f}")
            
            # 保存epoch检查点
            self.save_checkpoint(self.global_step, avg_epoch_loss)
        
        self.logger.info("Training completed!")


def create_optimizer_and_scheduler(adapter, train_config):
    """
    创建优化器和学习率调度器
    
    Args:
        adapter: Adapter模型
        train_config: 训练配置
        
    Returns:
        optimizer, scheduler
    """
    # 优化器
    optimizer = AdamW(
        adapter.get_trainable_parameters(),
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay'],
        betas=train_config.get('betas', (0.9, 0.999))
    )
    
    # 学习率调度器
    scheduler = None
    if train_config.get('use_scheduler', True):
        warmup_steps = train_config.get('warmup_steps', 200)
        total_steps = train_config.get('total_steps', 10000)
        warmup_ratio = train_config.get('warmup_ratio', 0.05)
        warmup_steps = max(int(warmup_steps), int(total_steps * warmup_ratio), 1)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                decay_steps = max(total_steps - warmup_steps, 1)
                progress = (step - warmup_steps) / decay_steps
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


if __name__ == "__main__":
    # 测试代码
    print("=== 测试训练器 ===")
    
    # 模拟配置
    train_config = {
        'lr': 1e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'use_scheduler': True,
        'warmup_steps': 500,
        'total_steps': 10000
    }
    
    # 创建模拟模型（需要替换为实际模型）
    class MockAdapter:
        def get_trainable_parameters(self):
            return [torch.nn.Parameter(torch.randn(10, 10))]
        
        def get_parameter_count(self):
            return {'total': 100, 'trainable': 100}
        
        def to(self, device):
            return self
        
        def train(self):
            pass
        
        def eval(self):
            pass
    
    class MockQwenModel:
        def to(self, device):
            return self
        
        def eval(self):
            pass
    
    # 创建组件
    adapter = MockAdapter()
    qwen_model = MockQwenModel()
    
    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(adapter, train_config)
    
    print(f"Optimizer: {optimizer}")
    print(f"Scheduler: {scheduler}")
