"""
训练结果可视化脚本：
- 从现有 checkpoint 加载 Adapter
- 复用真实 Qwen / fallback backbone 跑若干样本
- 将真值网格点与预测点叠加到网格图上
- 导出 PNG + JSON，方便快速判断模型学到了什么
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# 添加 src 到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import CoordinateDataset, collate_fn_pad_batch
from train import create_adapter, load_qwen_model, setup_transforms
from training.config import Config, get_config


BORDER_SIZE = 28
GRID_DIVISIONS = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TrainAdapter qualitative results")
    parser.add_argument("--checkpoint", type=str, default=None, help="Adapter checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Config json path saved during training")
    parser.add_argument("--preset", type=str, default="default", help="Preset name when config is absent")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root")
    parser.add_argument("--annotation_file", type=str, default=None, help="Override annotation file")
    parser.add_argument("--qwen_model_path", type=str, default=None, help="Override Qwen path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--output_dir", type=str, default="visualizations/qualitative", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=6, help="How many samples to visualize")
    parser.add_argument("--indices", type=str, default=None, help="Comma separated dataset indices")
    parser.add_argument("--query_contains", type=str, default=None, help="Only keep samples whose query contains this text")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sample picking")
    parser.add_argument("--top_k", type=int, default=4, help="How many predicted points to draw")
    return parser.parse_args()


def load_config(args):
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = get_config(args.preset)

    if args.data_root:
        config.data.data_root = args.data_root
    if args.annotation_file:
        config.data.annotation_file = args.annotation_file
    if args.qwen_model_path:
        config.model.qwen_model_path = args.qwen_model_path
    if args.device:
        config.device = args.device
    return config


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def parse_indices(indices_arg):
    if not indices_arg:
        return None
    values = []
    for chunk in indices_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def build_dataset(config):
    _, val_transform = setup_transforms(config)
    dataset = CoordinateDataset(
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
        target_coordinate_mode=config.data.target_coordinate_mode
    )
    return dataset


def select_indices(dataset, args):
    explicit = parse_indices(args.indices)
    if explicit is not None:
        return [idx for idx in explicit if 0 <= idx < len(dataset)]

    candidates = list(range(len(dataset)))
    if args.query_contains:
        needle = args.query_contains.lower()
        candidates = [
            idx for idx in candidates
            if needle in dataset.samples[idx]["query"].lower()
        ]

    if not candidates:
        return []

    count = min(args.num_samples, len(candidates))
    if count == len(candidates):
        return candidates

    if count == 1:
        return [candidates[0]]

    step = (len(candidates) - 1) / float(count - 1)
    selected = sorted({candidates[round(i * step)] for i in range(count)})
    while len(selected) < count:
        for idx in candidates:
            if idx not in selected:
                selected.append(idx)
            if len(selected) == count:
                break
    return sorted(selected)


def load_adapter_checkpoint(adapter, checkpoint_path, device):
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    adapter.load_state_dict(state_dict, strict=False)
    return checkpoint


def run_prediction(adapter, qwen_model, batch, device):
    images = batch["image"].to(device)
    grid_images = batch["grid_image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    adapter_dtype = next(adapter.parameters()).dtype

    with torch.no_grad():
        visual_features = qwen_model.encode_image(images)
        if visual_features.dtype != adapter_dtype:
            visual_features = visual_features.to(dtype=adapter_dtype)

        enhanced_features = adapter(images, grid_images, visual_features)

        text_embeddings = qwen_model.encode_text(input_ids, attention_mask)
        if text_embeddings.dtype != adapter_dtype:
            text_embeddings = text_embeddings.to(dtype=adapter_dtype)

        pred_points, pred_logits = adapter.predict_points(enhanced_features, text_embeddings)

    return pred_points[0].detach().cpu(), pred_logits[0].detach().cpu()


def normalized_points_to_pixels(points, image_size):
    width, height = image_size
    pixel_points = []
    for point in points:
        x = float(point[0]) * width
        y = float(point[1]) * height
        pixel_points.append((x, y))
    return pixel_points


def grid_points_to_pixels(grid_points, image_size):
    return prediction_points_to_grid_pixels(grid_points, image_size)


def prediction_points_to_grid_pixels(points, image_size):
    width, height = image_size
    pixel_points = []
    for point in points:
        x = BORDER_SIZE + float(point[0]) * width
        y = BORDER_SIZE + float(point[1]) * height
        pixel_points.append((x, y))
    return pixel_points


def draw_point(draw, point, color, label, radius=8, fill=True):
    x, y = point
    bbox = [x - radius, y - radius, x + radius, y + radius]
    if fill:
        draw.ellipse(bbox, fill=color, outline="white", width=2)
    else:
        draw.ellipse(bbox, outline=color, width=3)
    font = load_font(18, bold=True)
    draw.text((x + radius + 3, y - radius - 3), label, fill=color, font=font)


def wrap_text(text, width=42):
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= width:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def render_visual(sample, pred_points, pred_logits, output_path, top_k):
    image_id = sample["image_id"]
    query = sample["query"]
    gt_grid_points = sample["gt_points"]
    image_size = sample["image_size"]
    data_root = sample["data_root"]

    original_path = os.path.join(data_root, "images", image_id)
    grid_path = os.path.join(data_root, "grid_images", image_id)

    original_img = Image.open(original_path).convert("RGB")
    if os.path.exists(grid_path):
        grid_img = Image.open(grid_path).convert("RGB")
    else:
        grid_img = Image.new("RGB", (original_img.width + 2 * BORDER_SIZE, original_img.height + 2 * BORDER_SIZE), "white")
        grid_img.paste(original_img, (BORDER_SIZE, BORDER_SIZE))

    scores = torch.sigmoid(pred_logits).tolist()
    ranked = sorted(
        list(enumerate(zip(pred_points.tolist(), scores))),
        key=lambda item: item[1][1],
        reverse=True
    )[:max(1, top_k)]

    pred_norm_points = [item[1][0] for item in ranked]
    pred_scores = [item[1][1] for item in ranked]
    pred_pixel_points = prediction_points_to_grid_pixels(pred_norm_points, image_size)
    gt_pixel_points = grid_points_to_pixels(gt_grid_points, image_size)

    panel_gap = 30
    header_h = 150
    footer_h = 170
    canvas_w = original_img.width + grid_img.width + panel_gap * 3
    canvas_h = max(original_img.height, grid_img.height) + header_h + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(30, bold=True)
    body_font = load_font(18)
    small_font = load_font(16)

    draw.text((30, 22), "TrainAdapter Qualitative Result", fill="#18212f", font=title_font)
    draw.text((30, 62), f"Image: {image_id}", fill="#4a5568", font=body_font)

    y = 96
    for line in wrap_text(f"Query: {query}", width=70):
        draw.text((30, y), line, fill="#222b38", font=body_font)
        y += 24

    left_x = panel_gap
    top_y = header_h
    right_x = left_x + original_img.width + panel_gap

    canvas.paste(original_img, (left_x, top_y))
    canvas.paste(grid_img, (right_x, top_y))
    draw.rectangle(
        [left_x - 1, top_y - 1, left_x + original_img.width + 1, top_y + original_img.height + 1],
        outline="#cbd5e0",
        width=2
    )
    draw.rectangle(
        [right_x - 1, top_y - 1, right_x + grid_img.width + 1, top_y + grid_img.height + 1],
        outline="#cbd5e0",
        width=2
    )

    draw.text((left_x, top_y - 28), "Original Image", fill="#1f2937", font=body_font)
    draw.text((right_x, top_y - 28), "Grid Image + Ground Truth / Predictions", fill="#1f2937", font=body_font)

    overlay = ImageDraw.Draw(canvas)
    for idx, point in enumerate(gt_pixel_points, start=1):
        draw_point(overlay, (right_x + point[0], top_y + point[1]), "#1d4ed8", f"G{idx}", radius=7, fill=True)

    for idx, (point, score) in enumerate(zip(pred_pixel_points, pred_scores), start=1):
        draw_point(overlay, (right_x + point[0], top_y + point[1]), "#dc2626", f"P{idx}", radius=10, fill=False)
        overlay.text(
            (right_x + point[0] + 16, top_y + point[1] + 10),
            f"{score:.2f}",
            fill="#b91c1c",
            font=small_font
        )

    legend_y = top_y + max(original_img.height, grid_img.height) + 24
    overlay.rounded_rectangle(
        [30, legend_y, canvas_w - 30, canvas_h - 24],
        radius=18,
        fill="white",
        outline="#d7deea",
        width=2
    )
    overlay.text((50, legend_y + 18), "How To Read This Figure", fill="#162033", font=body_font)
    legend_lines = [
        "Blue filled dots (G1, G2...) are normalized supervision points sampled from the gRefCOCO target region.",
        "Red circles (P1, P2...) are the adapter's top predicted points ranked by sigmoid confidence.",
        "If red circles gather around the blue cluster, the model has learned to align text with the target region.",
        "Training, evaluation, and these overlays all use the same normalized grid coordinate system."
    ]
    text_y = legend_y + 52
    for line in legend_lines:
        overlay.text((50, text_y), line, fill="#334155", font=small_font)
        text_y += 24

    canvas.save(output_path)

    metadata = {
        "image_id": image_id,
        "query": query,
        "image_size": list(image_size),
        "ground_truth_points_normalized": gt_grid_points,
        "ground_truth_pixel_points": [[round(x, 2), round(y, 2)] for x, y in gt_pixel_points],
        "predicted_points_normalized": [[round(float(x), 4), round(float(y), 4)] for x, y in pred_norm_points],
        "predicted_points_on_grid_image": [[round(x, 2), round(y, 2)] for x, y in pred_pixel_points],
        "prediction_scores": [round(float(score), 4) for score in pred_scores]
    }
    with open(str(Path(output_path).with_suffix(".json")), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    config = load_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config)
    chosen_indices = select_indices(dataset, args)

    if not chosen_indices:
        raise SystemExit("No samples matched the selection criteria.")

    qwen_model, _ = load_qwen_model(config.model.qwen_model_path, device)
    if hasattr(qwen_model, "visual_dim"):
        config.model.visual_dim = qwen_model.visual_dim
    adapter = create_adapter(config).to(device)
    adapter.eval()
    qwen_model.eval()

    checkpoint_meta = load_adapter_checkpoint(adapter, args.checkpoint, device)

    manifest = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "data_root": config.data.data_root,
        "annotation_file": config.data.annotation_file,
        "selected_indices": chosen_indices,
        "device": str(device),
        "checkpoint_step": checkpoint_meta.get("step") if isinstance(checkpoint_meta, dict) else None,
        "checkpoint_loss": checkpoint_meta.get("loss") if isinstance(checkpoint_meta, dict) else None,
    }

    for sample_idx in chosen_indices:
        item = dataset[sample_idx]
        sample_meta = dict(dataset.samples[sample_idx])
        batch = collate_fn_pad_batch([item])
        pred_points, pred_logits = run_prediction(adapter, qwen_model, batch, device)
        render_item = {
            "image_id": sample_meta["image_id"],
            "query": item["query"],
            "gt_points": item["gt_points"],
            "image_size": item["image_size"],
            "data_root": config.data.data_root,
        }
        output_path = output_dir / f"sample_{sample_idx:04d}_{sample_meta['image_id']}"
        render_visual(render_item, pred_points, pred_logits, str(output_path.with_suffix(".png")), args.top_k)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chosen_indices)} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
