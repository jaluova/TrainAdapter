# TrainAdapter

TrainAdapter 是一个面向 referring expression localization 的训练与推理仓库。它以冻结的 `Qwen2.5-VL` 作为视觉语言主干，在其上训练一个轻量可学习的空间适配器，让模型根据文本查询在图像中预测目标所在的网格位置。

当前代码主线已经从早期的“点回归”转向 `11 x 11` 网格分类。仓库中仍保留了一些旧命名，比如 `CoordinateAdapter`、`TrainAdapter`，但实际默认流程是基于网格 logits 的定位方案。本文档只描述当前项目文件夹里的源码与脚本，不展开 `release/` 目录下的打包产物。

## 项目能做什么

- 训练一个冻结 Qwen 主干上的空间 Adapter
- 使用 `grefs_with_grids.json` 进行网格监督训练
- 支持 `grid_logits` 和旧版 `point_regression` 两种输出模式
- 支持单图命令行推理
- 支持常驻本机的 HTTP 推理服务
- 支持基于 Gradio 的网页 Demo
- 支持导出训练结果可视化面板
- 附带数据预处理、加网格、标注转换和数据检查脚本

## 核心思路

每条样本通常包含以下信息：

- 原始图像 `image`
- 叠加坐标网格后的图像 `grid_image`
- 文本查询 `query`
- 目标网格点 `grid_points`

默认模型路径由几部分组成：

1. `Qwen2.5-VL` 提供冻结的视觉和文本特征。
2. `GridEncoder` 从带网格的图像中提取空间先验。
3. `TextGuidedCrossAttention` 融合视觉、网格和文本信息。
4. Adapter 输出 `121` 维网格 logits，再解码为 top-k 网格点。

默认训练目标是稠密网格监督：

- loss 类型默认是 `bce_grid`
- 支持正样本加权 `grid_pos_weight`
- 支持邻域软标签 `neighbor_soft_label_weight`
- 支持 Gaussian 软标签
- 支持图像级 train/val 划分，避免同图不同 query 泄漏
- 支持对空间关系、颜色、多实体、顺序类 query 做加权采样

## 代码结构

```text
TrainAdapter/
├── README.md
├── requirements.txt
├── monitor/
│   └── server.py
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── inference_service.py
│   ├── web_demo.py
│   ├── check_data.py
│   ├── visualize_training_results.py
│   ├── data/
│   │   └── dataset.py
│   ├── loss/
│   │   └── hungarian_loss.py
│   ├── models/
│   │   ├── adapter.py
│   │   ├── cross_attention.py
│   │   └── grid_encoder.py
│   ├── training/
│   │   ├── config.py
│   │   ├── trainer.py
│   │   └── trainer.py.patch
│   └── utils/
│       └── coordinate_parser.py
└── utils/
    ├── process_images.py
    ├── add_grid.py
    ├── process_masks_final_v3.py
    ├── get_image_size.py
    ├── check_grefs.py
    ├── query_image_in_json.py
    └── visualize_grefs_queries.py
```

其中最常用的入口文件是：

- `src/train.py`：训练入口
- `src/inference.py`：单图推理入口
- `src/inference_service.py`：常驻 HTTP 推理服务
- `src/web_demo.py`：Gradio Demo
- `src/visualize_training_results.py`：checkpoint 可视化

## 环境依赖

建议使用 Python 3.10。安装方式：

```bash
pip install -r requirements.txt
```

`requirements.txt` 里包含的主要依赖有：

- `torch`
- `torchvision`
- `transformers`
- `qwen-vl-utils`
- `opencv-python`
- `Pillow`
- `matplotlib`
- `scikit-learn`
- `tensorboard`
- `wandb`
- `gradio`

如果你要加载真实的 Qwen 权重，请确保本地已有可用的 `Qwen2.5-VL-7B-Instruct` 目录。

## 默认路径与环境变量

配置定义在 `src/training/config.py`。默认会优先读取这些环境变量：

```bash
export TRAIN_ADAPTER_DATA_ROOT=/path/to/Data
export TRAIN_ADAPTER_QWEN_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export TRAIN_ADAPTER_SAVE_DIR=/path/to/train_outputs/run_name
```

默认行为如下：

- `data_root` 默认为 `<repo>/Data`
- `qwen_model_path` 默认为 `<repo>/Qwen2.5-VL-7B-Instruct`
- `save_dir` 默认为 `<data_root>/train_outputs`

如果没有找到 Qwen 权重，`src/train.py` 里的 `load_qwen_model()` 会退化到一个轻量 `FrozenBackbone`，便于先跑通 smoke test，但这不代表真实效果。

## 数据准备

默认训练使用的数据文件是：

```text
Data/
├── images/
├── grid_images/
└── grefs_with_grids.json
```

训练时最重要的几个字段是：

- `image_id`
- `file_name`
- `query`
- `grid_points`

仓库自带了三个常用预处理脚本：

### 1. 处理原图尺寸

把图像缩放到 28 的整数倍，便于后续网格和视觉主干对齐：

```bash
python utils/process_images.py
```

### 2. 生成带网格的图像

在原图外围加白边，并叠加 `0..10` 的横纵坐标：

```bash
python utils/add_grid.py
```

### 3. 从分割标注生成 `grefs_with_grids.json`

把实例 mask 映射成离散网格点监督：

```bash
python utils/process_masks_final_v3.py
```

### 4. 检查数据完整性

```bash
cd src
python check_data.py
```

注意：上面几个预处理脚本里仍写死了一些原始数据路径，真正使用前通常需要先按你的机器目录修改脚本中的路径常量。

## 配置预设

`src/training/config.py` 提供了 3 个 preset：

- `default`
- `lightweight`
- `high_performance`

默认配置特点：

- `adapter_type=lightweight`
- `output_mode=grid_logits`
- `grid_size=11`
- `num_output_points=4`
- `batch_size=8`
- `lr=3e-5`

如果只是先确认链路是否能跑通，建议从 `default` 或 `lightweight` 开始。

## 训练

在仓库根目录执行：

```bash
python src/train.py --preset default --device cuda
```

更完整的例子：

```bash
python src/train.py \
  --preset default \
  --data_root /path/to/Data \
  --save_dir /path/to/train_outputs/exp1 \
  --batch_size 8 \
  --num_epochs 4 \
  --lr 3e-5 \
  --device cuda
```

常用参数：

- `--config`：直接加载已有 `config.json`
- `--preset`：选择预设
- `--resume`：从 checkpoint 恢复
- `--resume_as_init`：只加载模型权重，把旧 checkpoint 当成新实验初始化
- `--adapter_type`：`standard` 或 `lightweight`
- `--data_root`
- `--annotation_file`
- `--batch_size`
- `--lr`
- `--num_epochs`

训练输出默认会写到：

```text
<save_dir>/
├── checkpoints/
├── logs/
└── train_outputs/
```

其中：

- `checkpoints/best_model.pth` 是最佳模型
- `config.json` 会随训练一起保存
- 日志里会记录 step loss 和 validation 指标

## 从 checkpoint 继续或迁移实验

继续训练：

```bash
python src/train.py \
  --config /path/to/config.json \
  --resume /path/to/checkpoint_step_xxx.pth \
  --device cuda
```

把旧模型当作新实验初始化：

```bash
python src/train.py \
  --config /path/to/config.json \
  --resume /path/to/checkpoint_or_best_model.pth \
  --resume_as_init \
  --save_dir /path/to/new_run \
  --device cuda
```

`resume_as_init` 会重置优化器、调度器、epoch、step 和最佳指标，只复用模型权重。

## 单图推理

最简单的命令行推理：

```bash
python src/inference.py \
  --adapter_path /path/to/best_model.pth \
  --config /path/to/config.json \
  --image_path /path/to/image.jpg \
  --query "the man on the left" \
  --device cuda
```

如果你已经有单独的网格图，也可以显式传入：

```bash
python src/inference.py \
  --adapter_path /path/to/best_model.pth \
  --config /path/to/config.json \
  --image_path /path/to/image.jpg \
  --grid_image_path /path/to/grid_image.jpg \
  --query "the red bottle" \
  --return_normalized
```

常用推理参数：

- `--return_text`
- `--return_normalized`
- `--save_pred`
- `--save_dir`
- `--dynamic_topk`
- `--dynamic_abs_threshold`
- `--dynamic_rel_ratio`
- `--dynamic_min_k`
- `--dynamic_max_k`

如果不提供 `grid_image_path`，推理代码会直接复制原图作为网格图输入，因此建议在正式评估时尽量提供和训练一致的网格图。

## 本地推理服务

`src/inference_service.py` 会常驻加载模型，对外暴露一个轻量 HTTP 接口。

启动方式：

```bash
python src/inference_service.py \
  --adapter_path /path/to/best_model.pth \
  --config /path/to/config.json \
  --device cuda \
  --host 127.0.0.1 \
  --port 8765
```

健康检查：

```bash
curl http://127.0.0.1:8765/health
```

`POST /predict` 请求体需要包含：

- `image_base64`
- `query`
- 可选的动态 top-k 参数

返回值中会包含：

- 归一化坐标
- 绝对像素坐标
- 分数
- 选点模式
- 标注后的结果图 `annotated_image_base64`

## Gradio Demo

`src/web_demo.py` 是一个前端很轻的上传式 Demo，它会把图片和 query 转发给上面的推理服务。

先启动推理服务，再启动 Demo：

```bash
python src/web_demo.py \
  --inference_url http://127.0.0.1:8765 \
  --host 127.0.0.1 \
  --port 7860
```

页面能力包括：

- 上传图片
- 输入 query
- 可选动态 top-k
- 查看标注图
- 查看坐标表格
- 查看原始 JSON 返回

## 训练结果可视化

可以从已有 checkpoint 导出定性结果，方便检查模型到底学到了什么：

```bash
python src/visualize_training_results.py \
  --checkpoint /path/to/checkpoint_step_xxx.pth \
  --config /path/to/config.json \
  --data_root /path/to/Data \
  --output_dir visualizations/qualitative \
  --num_samples 6 \
  --top_k 4
```

这个脚本会导出：

- 叠加了真值与预测点的可视化图片
- 每个样本对应的 JSON
- 便于人工检查的结果目录

## 训练监控脚本

仓库里有一个 `monitor/server.py`，用于通过 SSH 轮询远端训练状态并提供本地 HTTP 服务。

它依赖这些环境变量：

```bash
export TRAIN_MONITOR_HOST=your.remote.host
export TRAIN_MONITOR_PORT=22
export TRAIN_MONITOR_USER=root
export TRAIN_MONITOR_PASSWORD=your_password
export TRAIN_MONITOR_LOG_PATH=/remote/path/nohup.log
export TRAIN_MONITOR_SAVE_DIR=/remote/path/train_outputs/exp1
export TRAIN_MONITOR_PORT_LOCAL=4173
```

然后运行：

```bash
python monitor/server.py
```

需要注意的是，当前仓库里只有 `monitor/server.py`，它引用的 `monitor/index.html` 并不在现有文件夹中。所以这部分更适合视为“后端状态采集脚本仍在，前端静态页未随当前目录一起保留”。

## 当前仓库里几个容易混淆的点

- `CoordinateAdapter` 是历史命名，当前默认任务已经是网格分类，不只是坐标回归。
- `src/training/trainer.py.patch` 是一个补丁文件，不是主运行入口。
- `release/` 目录存在，但这份 README 不把它当作当前主开发流程的一部分。
- 某些数据处理脚本仍带有硬编码路径，迁移机器时通常需要先改路径。

## 建议的最小跑通流程

1. 准备 `Qwen2.5-VL` 权重目录和 `Data/`。
2. 生成或检查 `images/`、`grid_images/`、`grefs_with_grids.json`。
3. 进入 `src/` 后运行 `python check_data.py`。
4. 运行 `python src/train.py --preset default --device cuda`。
5. 用 `python src/inference.py` 验证单图推理。
6. 需要可视化时再运行 `src/visualize_training_results.py`。

## License

本仓库依赖外部模型和数据集，请自行确认以下资源的使用条款：

- `Qwen2.5-VL`
- COCO / gRefCOCO 相关数据

如果你要把这个项目用于公开发布、二次分发或线上服务，建议先完成一次许可证与数据合规检查。
