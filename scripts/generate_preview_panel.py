import argparse
import json
import math
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from data.dataset import CoordinateDataset, collate_fn_pad_batch, split_indices_by_image
from loss.hungarian_loss import HungarianPointLoss
from train import create_adapter, load_qwen_model, setup_transforms
from training.config import Config
from training.trainer import CoordinateAdapterTrainer, create_optimizer_and_scheduler


def build_val_dataloader(config, val_transform):
    full_dataset = CoordinateDataset(
        data_root=config.data.data_root,
        annotation_file=config.data.annotation_file,
        image_dir=config.data.image_dir,
        grid_image_dir=config.data.grid_image_dir,
        tokenizer_path=config.model.qwen_model_path,
        image_size=config.data.image_size,
        max_length=config.data.max_length,
        transform=val_transform,
        num_output_points=config.model.num_output_points,
        target_point_strategy=config.data.target_point_strategy,
        target_coordinate_mode=config.data.target_coordinate_mode,
        output_mode=config.model.output_mode,
        grid_size=config.model.grid_size,
        neighbor_soft_label_weight=config.training.neighbor_soft_label_weight,
        use_primary_grid_target=config.training.use_primary_grid_target,
        relation_keywords=config.data.relation_keywords,
        ordinal_keywords=config.data.ordinal_keywords,
        multi_entity_keywords=config.data.multi_entity_keywords,
    )

    if config.data.split_by_image:
        _, val_indices = split_indices_by_image(
            full_dataset.samples,
            val_ratio=config.data.val_ratio,
            seed=config.seed,
        )
        if not val_indices:
            raise RuntimeError("No validation indices available for preview generation")
        val_dataset = Subset(full_dataset, val_indices)
    else:
        val_dataset = full_dataset

    return DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_pad_batch,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate TrainAdapter preview panel")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--panel-root", default="preview_panels", help="Output panel subdir")
    parser.add_argument("--panel-title", default="Training Preview Panel", help="Panel title")
    parser.add_argument("--clear-existing", action="store_true", help="Remove existing panel dir before generation")
    parser.add_argument("--step", type=int, default=None, help="Optional preview step number")
    args = parser.parse_args()

    config = Config.load(args.config)
    device = torch.device(args.device)

    train_transform, val_transform = setup_transforms(config)
    val_dataloader = build_val_dataloader(config, val_transform)

    qwen_model, _ = load_qwen_model(config.model.qwen_model_path, device)
    if hasattr(qwen_model, "visual_dim"):
        config.model.visual_dim = qwen_model.visual_dim

    adapter = create_adapter(config)

    loss_fn = HungarianPointLoss(
        inside_bbox_weight=config.training.inside_bbox_weight,
        outside_bbox_weight=config.training.outside_bbox_weight,
        match_cost=config.training.match_cost,
        boundary_penalty_weight=config.training.boundary_penalty_weight,
        loss_type=config.training.loss_type,
        grid_size=config.model.grid_size,
        grid_pos_weight=config.training.grid_pos_weight,
        neighbor_soft_label_weight=config.training.neighbor_soft_label_weight,
        ranking_margin=config.training.ranking_margin,
        ranking_loss_weight=config.training.ranking_loss_weight,
    )

    train_config_dict = dict(config.training.__dict__)
    train_config_dict["total_steps"] = max(
        math.ceil(len(val_dataloader) / max(config.training.gradient_accumulation_steps, 1)),
        1,
    )
    optimizer, scheduler = create_optimizer_and_scheduler(adapter, train_config_dict)

    trainer = CoordinateAdapterTrainer(
        adapter=adapter,
        qwen_model=qwen_model,
        train_dataloader=val_dataloader,
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
        save_interval=config.logging.save_interval,
        preview_interval=getattr(config.logging, "preview_interval", 0),
        loss_type=config.training.loss_type,
        use_amp=config.training.use_amp,
        early_stop_patience_evals=config.training.early_stop_patience_evals,
    )
    trainer.load_checkpoint(args.checkpoint, resume_as_init=True)

    panel_base_dir = os.path.join(config.logging.save_dir, args.panel_root)
    if args.clear_existing:
        shutil.rmtree(panel_base_dir, ignore_errors=True)
    os.makedirs(panel_base_dir, exist_ok=True)

    step = args.step
    if step is None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        step = int(checkpoint.get("step", 999999))

    panel_dir = trainer._save_qualitative_panel(
        step=step,
        panel_root=args.panel_root,
        panel_title=args.panel_title,
    )
    if not panel_dir:
        raise RuntimeError("Preview panel generation returned no output")

    manifest_path = os.path.join(panel_dir, "manifest.json")
    print(panel_dir)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
