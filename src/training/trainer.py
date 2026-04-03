"""
训练器：负责训练Coordinate Adapter
"""
import os
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
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
                 save_interval=500,
                 preview_interval=0,
                 loss_type='hungarian_point',
                 use_amp=False,
                 early_stop_patience_evals=0):
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
        self.preview_interval = max(int(preview_interval or 0), 0)
        self.loss_type = loss_type
        self.use_amp = bool(use_amp and str(device).startswith('cuda'))
        self.early_stop_patience_evals = max(int(early_stop_patience_evals or 0), 0)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.resume_reset_optimizer = self._read_env_flag('TRAIN_ADAPTER_RESUME_RESET_OPTIMIZER', default=True)
        self.resume_reset_scheduler = self._read_env_flag('TRAIN_ADAPTER_RESUME_RESET_SCHEDULER', default=True)
        self.stop_on_nonfinite = self._read_env_flag('TRAIN_ADAPTER_STOP_ON_NONFINITE', default=True)
        self.override_lr = self._read_env_float('TRAIN_ADAPTER_OVERRIDE_LR')
        
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
        self.best_acc_1grid = -1.0
        self.best_acc_5 = -1.0
        self.best_acc_top4 = -1.0
        self._accumulated_batches = 0
        self._stale_validation_count = 0
        self.qualitative_top_k = min(4, getattr(self.adapter, 'num_output_points', 4))
        self.qualitative_panel_rotation = 0
        self._last_qualitative_rotation_step = None
        self.qualitative_panel_indices = self._select_qualitative_indices(rotation=self.qualitative_panel_rotation)
        
        # 冻结Qwen模型
        self._freeze_qwen_model()

    @staticmethod
    def _read_env_flag(name, default=False):
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _read_env_float(name):
        value = os.environ.get(name)
        if value is None or value == '':
            return None
        return float(value)
    
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
            'best_acc_1grid': self.best_acc_1grid,
            'best_acc_5': self.best_acc_5,
            'best_acc_top4': self.best_acc_top4,
            'loss_type': self.loss_type,
            'accumulated_batches': self._accumulated_batches
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'checkpoints', f'checkpoint_step_{step}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at step {step} with loss {loss:.4f}")

        self._prune_regular_checkpoints(keep_last=3)
        
        self.logger.info(f"Saved checkpoint at step {step}")

    def _prune_regular_checkpoints(self, keep_last=3):
        """只保留最近 keep_last 个普通 checkpoint，best_model.pth 永远保留。"""
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            return

        pattern = re.compile(r"checkpoint_step_(\d+)\.pth$")
        checkpoints = []
        for name in os.listdir(checkpoint_dir):
            match = pattern.match(name)
            if not match:
                continue
            checkpoints.append((int(match.group(1)), os.path.join(checkpoint_dir, name)))

        checkpoints.sort(key=lambda item: item[0], reverse=True)
        for _, path in checkpoints[max(keep_last, 0):]:
            try:
                os.remove(path)
                self.logger.info(f"Removed old checkpoint: {os.path.basename(path)}")
            except FileNotFoundError:
                continue
    
    def _set_learning_rate(self, lr_value):
        if lr_value is None or self.optimizer is None:
            return
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_value

    def load_checkpoint(self, checkpoint_path, resume_as_init=False):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        strict = not resume_as_init
        missing_keys, unexpected_keys = self.adapter.load_state_dict(
            checkpoint['model_state_dict'],
            strict=strict
        )
        if missing_keys:
            self.logger.info(f"Missing adapter keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            self.logger.info(f"Unexpected adapter keys when loading checkpoint: {unexpected_keys}")
        if self.optimizer and not self.resume_reset_optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.logger.info("Skipped optimizer state restore; using fresh optimizer state")
        
        if self.scheduler and not self.resume_reset_scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif self.scheduler and self.resume_reset_scheduler:
            self.logger.info("Skipped scheduler state restore; using fresh scheduler state")
        
        if resume_as_init:
            self.global_step = 0
            self.epoch = 0
            self.best_loss = float('inf')
            self.best_acc_1grid = -1.0
            self.best_acc_5 = -1.0
            self.best_acc_top4 = -1.0
            self._stale_validation_count = 0
            self._accumulated_batches = 0
            self.logger.info("Loaded checkpoint as initialization only; reset optimizer/scheduler progress and best metrics")
        else:
            self.global_step = checkpoint['step']
            self.epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.best_acc_1grid = checkpoint.get('best_acc_1grid', self.best_acc_1grid)
            self.best_acc_5 = checkpoint.get('best_acc_5', self.best_acc_5)
            self.best_acc_top4 = checkpoint.get('best_acc_top4', self.best_acc_top4)
            self._accumulated_batches = 0
        if self.override_lr is not None:
            self._set_learning_rate(self.override_lr)
            self.logger.info(f"Overrode optimizer lr to {self.override_lr}")
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _is_better_validation(self, metrics, val_loss):
        return (
            metrics['acc_1grid'] > self.best_acc_1grid or
            (
                metrics['acc_1grid'] == self.best_acc_1grid and
                metrics['acc_top4'] > self.best_acc_top4
            ) or
            (
                metrics['acc_1grid'] == self.best_acc_1grid and
                metrics['acc_top4'] == self.best_acc_top4 and
                val_loss < self.best_loss
            )
        )

    def _ensure_finite_tensor(self, value, name):
        if torch.is_tensor(value):
            is_finite = torch.isfinite(value).all()
            if bool(is_finite):
                return
            raise FloatingPointError(f"Non-finite tensor detected in {name}")

    def _ensure_finite_outputs(self, outputs):
        for key, value in outputs.items():
            if value is not None:
                self._ensure_finite_tensor(value, key)

    def _optimizer_step(self):
        if self._accumulated_batches <= 0:
            return

        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.max_grad_norm)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulated_batches = 0

    def _unwrap_dataset(self, dataset):
        if isinstance(dataset, Subset):
            return dataset.dataset, dataset.indices
        return dataset, list(range(len(dataset)))

    def _dataset_sample_meta(self, dataset, idx):
        base_dataset, indices = self._unwrap_dataset(dataset)
        base_idx = indices[idx]
        return base_dataset.samples[base_idx], base_dataset, base_idx

    def _select_qualitative_indices(self, count=6, rotation=0):
        """固定一组更分散的验证样本，优先保证不同图片与不同难度类型。"""
        if self.val_dataloader is None or not hasattr(self.val_dataloader, 'dataset'):
            return []

        dataset = self.val_dataloader.dataset
        if len(dataset) == 0:
            return []

        count = min(count, len(dataset))
        buckets = {
            'spatial_relation': [],
            'ordinal': [],
            'multi_entity': [],
            'easy_salient': [],
            'other': [],
        }

        for idx in range(len(dataset)):
            sample_meta, _, _ = self._dataset_sample_meta(dataset, idx)
            difficulty = sample_meta.get('difficulty_tag') or 'other'
            if difficulty not in buckets:
                difficulty = 'other'
            buckets[difficulty].append((idx, sample_meta))

        def pick_evenly_spaced(entries, desired_count, used_images, used_indices, rotation_offset):
            if desired_count <= 0 or not entries:
                return []

            available = [
                (idx, meta) for idx, meta in entries
                if idx not in used_indices and meta.get('image_id') not in used_images
            ]
            if not available:
                available = [(idx, meta) for idx, meta in entries if idx not in used_indices]
            if not available:
                return []

            desired_count = min(desired_count, len(available))
            start = int(rotation_offset) % len(available)
            rotated_available = available[start:] + available[:start]
            if desired_count >= len(available):
                chosen = rotated_available[:desired_count]
            else:
                step = len(rotated_available) / float(desired_count)
                chosen = []
                for i in range(desired_count):
                    pick_idx = min(int(i * step), len(rotated_available) - 1)
                    chosen.append(rotated_available[pick_idx])

            results = []
            for idx, meta in chosen:
                if idx in used_indices:
                    continue
                image_id = meta.get('image_id')
                if image_id in used_images and any(m.get('image_id') != image_id for _, m in available):
                    continue
                used_indices.add(idx)
                if image_id:
                    used_images.add(image_id)
                results.append(idx)
            return results

        selected = []
        used_indices = set()
        used_images = set()

        preferred_order = ['spatial_relation', 'ordinal', 'multi_entity', 'easy_salient']
        remaining_slots = count
        non_empty_bucket_count = sum(1 for name in preferred_order if buckets[name])

        for bucket_name in preferred_order:
            entries = buckets[bucket_name]
            if not entries or remaining_slots <= 0:
                continue
            target = 1 if non_empty_bucket_count >= remaining_slots else min(2, remaining_slots)
            picked = pick_evenly_spaced(
                entries,
                target,
                used_images,
                used_indices,
                rotation_offset=rotation + len(selected)
            )
            selected.extend(picked)
            remaining_slots = count - len(selected)

        if len(selected) < count:
            combined_pool = []
            for bucket_name in preferred_order + ['other']:
                combined_pool.extend(buckets[bucket_name])
            selected.extend(
                pick_evenly_spaced(
                    combined_pool,
                    count - len(selected),
                    used_images,
                    used_indices,
                    rotation_offset=rotation + len(selected)
                )
            )

        return sorted(selected[:count])

    def _rotate_qualitative_indices_if_needed(self, step):
        if self.val_dataloader is None:
            return
        if self._last_qualitative_rotation_step is None:
            self._last_qualitative_rotation_step = step
            return
        if step == self._last_qualitative_rotation_step:
            return

        self.qualitative_panel_rotation += 1
        self.qualitative_panel_indices = self._select_qualitative_indices(
            rotation=self.qualitative_panel_rotation
        )
        self._last_qualitative_rotation_step = step
        self.logger.info(
            f"Rotated qualitative preview group to round {self.qualitative_panel_rotation}: "
            f"{self.qualitative_panel_indices}"
        )

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

    def _build_grid_visual(self, original_img, border_size=28, grid_divisions=10):
        """根据原图动态生成带网格底图，避免依赖可能错配的磁盘网格图。"""
        width, height = original_img.size
        canvas = Image.new('RGB', (width + border_size * 2, height + border_size * 2), 'white')
        canvas.paste(original_img, (border_size, border_size))

        draw = ImageDraw.Draw(canvas)
        grid_color = '#94a3b8'
        axis_color = '#475569'
        label_font = self._load_font(14)

        for i in range(grid_divisions + 1):
            x = border_size + (width * i / grid_divisions)
            y = border_size + (height * i / grid_divisions)
            color = axis_color if i in (0, grid_divisions) else grid_color
            line_width = 2 if i in (0, grid_divisions) else 1
            draw.line([(x, border_size), (x, border_size + height)], fill=color, width=line_width)
            draw.line([(border_size, y), (border_size + width, y)], fill=color, width=line_width)
            if i < grid_divisions:
                draw.text((x + 4, 6), str(i), fill=axis_color, font=label_font)
                draw.text((6, y + 2), str(i), fill=axis_color, font=label_font)

        return canvas

    def _save_qualitative_panel(self, step, panel_root='qualitative_panels', panel_title='Validation Qualitative Panel'):
        if self.val_dataloader is None or not self.qualitative_panel_indices:
            return None

        self._rotate_qualitative_indices_if_needed(step)

        from data.dataset import collate_fn_pad_batch

        dataset = self.val_dataloader.dataset
        panel_dir = os.path.join(self.save_dir, panel_root, f'step_{step:06d}')
        os.makedirs(panel_dir, exist_ok=True)

        manifest = {
            'step': step,
            'epoch': self.epoch,
            'indices': self.qualitative_panel_indices,
            'coordinate_mode': 'normalized_grid',
            'top_k': self.qualitative_top_k,
            'panel_root': panel_root,
            'panel_title': panel_title,
            'rotation_round': self.qualitative_panel_rotation,
        }

        title_font = self._load_font(26, bold=True)
        body_font = self._load_font(18)
        small_font = self._load_font(16)

        with torch.no_grad():
            for sample_idx in self.qualitative_panel_indices:
                item = dataset[sample_idx]
                batch = collate_fn_pad_batch([item])
                outputs = self.forward_batch(batch)
                pred_points = outputs['pred_points'][0].detach().cpu().tolist()
                pred_scores = torch.sigmoid(outputs['pred_logits'][0]).detach().cpu().tolist()
                ranked = self._rank_predictions(pred_points, pred_scores)

                sample_meta, base_dataset, _ = self._dataset_sample_meta(dataset, sample_idx)
                image_id = sample_meta['image_id']
                image_size = item['image_size']
                original_path = os.path.join(base_dataset.data_root, base_dataset.image_dir, image_id)

                original_img = Image.open(original_path).convert('RGB')
                grid_img = self._build_grid_visual(original_img)

                gt_pixel_points = self._normalized_points_to_grid_pixels(item['gt_points'], image_size)
                pred_pixel_points = self._normalized_points_to_grid_pixels([point for point, _ in ranked], image_size)

                panel_gap = 30
                header_h = 140
                footer_h = 130
                canvas_w = original_img.width + grid_img.width + panel_gap * 3
                canvas_h = max(original_img.height, grid_img.height) + header_h + footer_h
                canvas = Image.new('RGB', (canvas_w, canvas_h), '#f5f7fb')
                draw = ImageDraw.Draw(canvas)

                draw.text((30, 20), panel_title, fill="#18212f", font=title_font)
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
                draw.text((50, footer_y + 50), "Blue dots are target grid points; red circles are top confidence predictions decoded from grid logits.", fill="#334155", font=small_font)
                draw.text((50, footer_y + 74), "Scores come from sigmoid(top-k logits); training and evaluation both operate in normalized grid coordinates.", fill="#334155", font=small_font)

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
        前向计算：提取冻结特征，经过Adapter预测网格 logits / 坐标点
        
        Args:
            batch: 批次数据
            
        Returns:
            outputs: 包含 pred_points / pred_logits / pred_grid_logits
        """
        images = batch['image'].to(self.device)
        grid_images = batch['grid_image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        adapter_dtype = next(self.adapter.parameters()).dtype

        autocast_enabled = self.use_amp and str(self.device).startswith('cuda')
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            with torch.no_grad():
                visual_features = self.qwen_model.encode_image(images)
                if visual_features.dtype != adapter_dtype:
                    visual_features = visual_features.to(dtype=adapter_dtype)

            enhanced_features = self.adapter(images, grid_images, visual_features)

            with torch.no_grad():
                text_embeddings = self.qwen_model.encode_text(input_ids, attention_mask)
                if text_embeddings.dtype != adapter_dtype:
                    text_embeddings = text_embeddings.to(dtype=adapter_dtype)

            if getattr(self.adapter, 'output_mode', 'point_regression') == 'grid_logits':
                pred_grid_logits = self.adapter.predict_grid_logits(
                    enhanced_features,
                    text_features=text_embeddings,
                    attention_mask=attention_mask
                )
                pred_points, pred_logits = self.adapter.decode_grid_logits(
                    pred_grid_logits,
                    top_k=self.qualitative_top_k
                )
            else:
                pred_grid_logits = None
                pred_points, pred_logits = self.adapter.predict_point_regression(
                    enhanced_features,
                    text_features=text_embeddings,
                    attention_mask=attention_mask
                )

        return {
            'pred_points': pred_points,
            'pred_logits': pred_logits,
            'pred_grid_logits': pred_grid_logits
        }
    
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
        
        outputs = self.forward_batch(batch)
        self._ensure_finite_outputs(outputs)
        gt_points_list = batch['gt_points']
        grid_targets = batch['grid_target'].to(self.device)
        image_sizes = batch['image_size']

        if outputs['pred_grid_logits'] is not None and self.loss_type == 'bce_grid':
            loss, match_info = self.loss_fn(
                pred_grid_logits=outputs['pred_grid_logits'],
                grid_targets=grid_targets,
                gt_points_list=gt_points_list,
                top_k=self.qualitative_top_k
            )
        else:
            loss, match_info = self.loss_fn(
                pred_points=outputs['pred_points'],
                pred_logits=outputs['pred_logits'],
                gt_points_list=gt_points_list,
                image_sizes=image_sizes
            )

        self._ensure_finite_tensor(loss, 'train_loss')

        for sample_idx, info in enumerate(match_info):
            info['is_relation_query'] = bool(batch['is_relation_query'][sample_idx].item())
            info['query'] = batch['query'][sample_idx]
            info['image_id'] = batch['image_id'][sample_idx]

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accumulated_batches += 1
        if self._accumulated_batches >= self.gradient_accumulation_steps:
            self._optimizer_step()

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
                outputs = self.forward_batch(batch)
                self._ensure_finite_outputs(outputs)
                gt_points_list = batch['gt_points']
                grid_targets = batch['grid_target'].to(self.device)
                image_sizes = batch['image_size']

                if outputs['pred_grid_logits'] is not None and self.loss_type == 'bce_grid':
                    loss, match_info = self.loss_fn(
                        pred_grid_logits=outputs['pred_grid_logits'],
                        grid_targets=grid_targets,
                        gt_points_list=gt_points_list,
                        top_k=self.qualitative_top_k
                    )
                else:
                    loss, match_info = self.loss_fn(
                        pred_points=outputs['pred_points'],
                        pred_logits=outputs['pred_logits'],
                        gt_points_list=gt_points_list,
                        image_sizes=image_sizes
                    )
                self._ensure_finite_tensor(loss, 'val_loss')

                for sample_idx, info in enumerate(match_info):
                    info['is_relation_query'] = bool(batch['is_relation_query'][sample_idx].item())
                    info['query'] = batch['query'][sample_idx]
                    info['image_id'] = batch['image_id'][sample_idx]
                
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
        all_min_distances = []
        acc_1grid = 0
        acc_top4 = 0
        relation_acc_top4 = 0
        relation_count = 0
        total_samples = 0

        for info in match_info:
            pred_points = info['pred_points']
            gt_points = info['gt_points']
            if len(pred_points) == 0 or len(gt_points) == 0:
                continue

            pred_arr = np.asarray(pred_points, dtype=np.float32)
            gt_arr = np.asarray(gt_points, dtype=np.float32)
            distances = np.linalg.norm(pred_arr[:, None, :] - gt_arr[None, :, :], axis=-1)
            min_distance = float(distances.min())
            all_min_distances.append(min_distance)
            total_samples += 1

            top1_hit = bool((distances[0] < 1e-6).any())
            top4_hit = bool((distances[:min(4, len(pred_points))] < 1e-6).any())
            acc_1grid += int(top1_hit)
            acc_top4 += int(top4_hit)

            if info.get('is_relation_query', False):
                relation_count += 1
                relation_acc_top4 += int(top4_hit)

        metrics = {
            'mean_min_grid_distance': float(np.mean(all_min_distances)) if all_min_distances else 0.0,
            'acc_1grid': acc_1grid / total_samples if total_samples > 0 else 0.0,
            'acc_top4': acc_top4 / total_samples if total_samples > 0 else 0.0,
            'relation_acc_top4': relation_acc_top4 / relation_count if relation_count > 0 else 0.0,
            'total_samples': total_samples,
            'relation_samples': relation_count,
        }

        metrics['l1_error'] = metrics['mean_min_grid_distance']
        metrics['acc_5'] = metrics['acc_top4']
        metrics['acc_10'] = metrics['acc_top4']
        return metrics
    
    def train(self, num_epochs, resume_from=None, resume_as_init=False):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            resume_from: 从检查点恢复训练
        """
        if resume_from:
            self.load_checkpoint(resume_from, resume_as_init=resume_as_init)
        start_epoch = 0 if (resume_from and resume_as_init) else (self.epoch if resume_from else 0)
        
        self.logger.info(f"Start training for {num_epochs} epochs")
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulated_batches = 0
        
        should_stop_early = False
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            relation_epoch_losses = []
            
            # 训练
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f'Training Epoch {epoch + 1}')):
                try:
                    loss, match_info = self.train_step(batch)
                    
                    epoch_loss += loss
                    num_batches += 1
                    self.global_step += 1

                    relation_sample_losses = [
                        info['sample_loss'] for info in match_info
                        if info.get('is_relation_query', False)
                    ]
                    if relation_sample_losses:
                        relation_epoch_losses.extend(relation_sample_losses)
                    
                    # 日志
                    if self.global_step % self.log_interval == 0:
                        log_message = (
                            f"Step {self.global_step}, Loss: {loss:.4f}, "
                            f"Avg Loss: {epoch_loss / num_batches:.4f}"
                        )
                        if relation_sample_losses:
                            log_message += f", Relation Loss: {np.mean(relation_sample_losses):.4f}"
                        self.logger.info(log_message)

                    if self.preview_interval > 0 and self.global_step % self.preview_interval == 0:
                        preview_dir = self._save_qualitative_panel(
                            self.global_step,
                            panel_root='preview_panels',
                            panel_title='Training Preview Panel'
                        )
                        if preview_dir:
                            self.logger.info(f"Saved preview panel to {preview_dir}")
                    
                    # 验证
                    if self.val_dataloader and self.global_step % self.eval_interval == 0:
                        val_loss, metrics = self.evaluate()
                        if val_loss is not None:
                            self.logger.info(
                                f"Validation - Loss: {val_loss:.4f}, "
                                f"Mean Min Grid Distance: {metrics['mean_min_grid_distance']:.4f}, "
                                f"Acc@1Grid: {metrics['acc_1grid']:.2%}, "
                                f"Acc@Top4: {metrics['acc_top4']:.2%}, "
                                f"Relation Acc@Top4: {metrics['relation_acc_top4']:.2%}"
                            )
                            if metrics.get('qualitative_panel_dir'):
                                self.logger.info(f"Saved qualitative panel to {metrics['qualitative_panel_dir']}")
                            
                            # 保存最佳模型
                            if self._is_better_validation(metrics, val_loss):
                                self.best_acc_1grid = metrics['acc_1grid']
                                self.best_acc_5 = metrics['acc_top4']
                                self.best_acc_top4 = metrics['acc_top4']
                                self.best_loss = val_loss
                                self._stale_validation_count = 0
                                self.save_checkpoint(self.global_step, val_loss, is_best=True)
                            else:
                                self._stale_validation_count += 1
                                if self.early_stop_patience_evals > 0:
                                    self.logger.info(
                                        f"No validation improvement for {self._stale_validation_count} eval(s); "
                                        f"patience={self.early_stop_patience_evals}"
                                    )
                                    if self._stale_validation_count >= self.early_stop_patience_evals:
                                        self.logger.info("Early stopping triggered by validation plateau")
                                        should_stop_early = True
                    
                    # 保存检查点
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint(self.global_step, loss)

                    if should_stop_early:
                        break
                
                except Exception as e:
                    self.optimizer.zero_grad(set_to_none=True)
                    self._accumulated_batches = 0
                    self.logger.error(f"Error at step {self.global_step}: {str(e)}")
                    if isinstance(e, FloatingPointError) and self.stop_on_nonfinite:
                        raise
                    continue

            if self._accumulated_batches > 0:
                self._optimizer_step()

            if should_stop_early:
                break
            
            #  epoch结束
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            if relation_epoch_losses:
                self.logger.info(
                    f"Epoch {epoch + 1} completed, Avg Loss: {avg_epoch_loss:.4f}, "
                    f"Relation Avg Loss: {np.mean(relation_epoch_losses):.4f}"
                )
            else:
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
