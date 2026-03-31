# GridGround: Spatial Grid Adapter for Qwen2.5-VL

GridGround is a lightweight spatial adapter for referring-expression localization on top of a frozen `Qwen2.5-VL` backbone. The project takes paired original images and grid-overlaid images, learns dense `11 x 11` grid supervision from `grefs_with_grids.json`, and predicts the most likely target grid locations for a natural-language query.

This repository was previously documented as `Coordinate Adapter`. The codebase name remains `TrainAdapter`, while the formal project name used in this README is `GridGround`.

## Current Status

As of `2026-03-31`, the project has already completed a real remote A100 training loop with the actual `Qwen2.5-VL-7B-Instruct` vision-language weights rather than the fallback mock backbone.

- Verified remote environment: `A100 40GB`
- Verified model path: `/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct`
- Verified 100-image smoke test: completed end-to-end
- Verified 1000-image fast-validation subset: built and training
- Verified local monitor: `http://127.0.0.1:4173`

Latest confirmed fast-validation run snapshot during the current round:

- Dataset: `1000` unique images / `4512` annotations
- Runtime split: `3065` train / `756` val
- Experiment dir: `/root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331`
- Training command: `python train.py --preset default --batch_size 4 --num_epochs 6 --device cuda`
- Recent validation: `Acc@1Grid 21.70%`, `Acc@Top4 53.53%`, `Relation Acc@Top4 53.47%`

## What Changed In This Round

- Switched the main task from sparse point regression to dense grid classification.
- The adapter now predicts `121` grid logits instead of only a few regressed points.
- Training now uses full `grid_points` supervision with `BCEWithLogitsLoss`.
- Added neighbor soft labels to reduce one-cell-off noise.
- Added relation-query oversampling and image-level train/val splitting.
- Enabled AMP in training config.
- Added a local web monitor for remote training status.
- Fixed several runtime issues found on the remote `torch 2.1.2` environment.

## Core Design

### Inputs

Each sample contains:

- `image`: resized original image
- `grid_image`: the same image with an overlaid coordinate grid
- `query`: referring expression text
- `grid_points`: one or more target grid points derived from masks

### Model

The trainable part is a lightweight adapter on top of a frozen Qwen backbone:

- `src/models/grid_encoder.py`: extracts grid-aware spatial features
- `src/models/cross_attention.py`: fuses visual and grid tokens
- `src/models/adapter.py`: standard and lightweight adapter variants

The default path now uses:

- frozen `Qwen2.5-VL` for image/text features
- lightweight adapter for efficiency on small and medium subsets
- `grid_logits` output mode for dense `11 x 11` prediction

### Loss And Decoding

- Training target: `121`-dimensional multi-hot or soft-label grid target
- Main loss: `BCEWithLogitsLoss`
- Positive weighting: `grid_pos_weight=4.0`
- Neighbor smoothing: `neighbor_soft_label_weight=0.3`
- Visualization/inference: decode sigmoid logits with top-k grid points

## Repository Layout

```text
TrainAdapter/
├── README.md
├── Adapter_README.md
├── requirements.txt
├── monitor/
│   ├── index.html
│   └── server.py
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── visualize_training_results.py
│   ├── data/
│   │   └── dataset.py
│   ├── loss/
│   │   └── hungarian_loss.py
│   ├── models/
│   │   ├── adapter.py
│   │   ├── cross_attention.py
│   │   └── grid_encoder.py
│   └── training/
│       ├── config.py
│       └── trainer.py
└── utils/
    ├── process_images.py
    ├── add_grid.py
    ├── process_masks_final_v3.py
    └── visualize_grefs_queries.py
```

## Environment Variables

The project is designed to work cleanly with runtime path overrides:

```bash
export TRAIN_ADAPTER_DATA_ROOT=/root/autodl-tmp/Data
export TRAIN_ADAPTER_QWEN_PATH=/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct
export TRAIN_ADAPTER_SAVE_DIR=/root/autodl-tmp/Data/train_outputs/minimal_a100_smoketest
```

These variables are read in `src/training/config.py`.

## Installation

Recommended remote environment used in the verified A100 runs:

```bash
conda create -p /root/autodl-tmp/conda-envs/adapter python=3.10 -y
conda activate /root/autodl-tmp/conda-envs/adapter

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 transformers==4.51.3
pip install -r requirements.txt
```

## Data Preparation

Expected runtime layout:

```text
/root/autodl-tmp/
├── TrainAdapter/
├── Data/
│   ├── images/
│   ├── grid_images/
│   ├── grefs(unc).json
│   ├── grefs_with_grids.json
│   └── train_outputs/
├── Data1000/
│   ├── images/
│   ├── grid_images/
│   └── grefs_with_grids.json
└── modelscope/
    └── Qwen2.5-VL-7B-Instruct/
```

To build processed training data:

```bash
python utils/process_images.py
python utils/add_grid.py
python utils/process_masks_final_v3.py
```

The processed annotation file consumed by training is `grefs_with_grids.json`.

## Training

From `src/`:

```bash
python train.py --preset default --batch_size 4 --num_epochs 6 --device cuda
```

Useful overrides:

```bash
python train.py \
  --preset default \
  --data_root /root/autodl-tmp/Data1000 \
  --save_dir /root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331 \
  --batch_size 4 \
  --num_epochs 6 \
  --device cuda
```

Resume from a checkpoint:

```bash
python train.py \
  --config /root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331/config.json \
  --resume /root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331/checkpoints/checkpoint_step_766.pth \
  --device cuda
```

## Visualization

Generate qualitative panels from a checkpoint:

```bash
python src/visualize_training_results.py \
  --checkpoint /root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331/checkpoints/checkpoint_step_766.pth \
  --config /root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331/config.json \
  --data_root /root/autodl-tmp/Data1000 \
  --output_dir visualizations/qualitative_fast1000 \
  --num_samples 6 \
  --top_k 4
```

The script overlays:

- ground-truth grid points
- top-k predicted grid points
- per-sample JSON metadata for later review

## Inference

```bash
python src/inference.py \
  --adapter_path /path/to/checkpoint.pth \
  --image_path /path/to/image.jpg \
  --query "the man on the left"
```

The inference path is compatible with both:

- current `grid_logits` checkpoints
- older point-regression checkpoints through the compatibility branch

## Local Training Monitor

The local monitor is a small polling web service that fetches remote status through SSH and renders:

- latest step and loss
- validation metrics
- GPU usage
- running training processes
- recent checkpoints
- log tail

Start it locally:

```bash
export TRAIN_MONITOR_HOST=region-9.autodl.pro
export TRAIN_MONITOR_PORT=22457
export TRAIN_MONITOR_USER=root
export TRAIN_MONITOR_PASSWORD='your-password'
export TRAIN_MONITOR_LOG_PATH=/root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331/nohup.log
export TRAIN_MONITOR_SAVE_DIR=/root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331
export TRAIN_MONITOR_PORT_LOCAL=4173

python monitor/server.py
```

Then open:

```text
http://127.0.0.1:4173
```

## Important Compatibility Fixes

The following fixes were required to make the verified A100 run stable:

- `src/models/adapter.py`: replaced `torch.gelu` with `torch.nn.functional.gelu`
- `src/models/adapter.py`: set lightweight `num_grid_tokens=25` to match actual pooled token geometry
- `src/training/config.py`: aligned lightweight/default preset token count to `25`
- `src/loss/hungarian_loss.py`: moved GT tensors to prediction device inside grid loss

## Recommended Next Experiments

If the goal is the fastest signal on whether the grid-classification idea works, the current order of experiments is:

1. Finish the `1000`-image fast-validation run.
2. Inspect the fixed 6-image qualitative panel after each checkpoint.
3. If the trend is positive, scale to `3000` images before touching the model design again.
4. Only after that, consider moving from lightweight adapter back to the standard adapter.

## License / Usage Note

This repository depends on external model and dataset licenses, especially:

- `Qwen2.5-VL-7B-Instruct`
- COCO / gRefCOCO related assets

Please verify their respective terms before redistribution or production use.
