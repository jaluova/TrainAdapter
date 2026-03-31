# TrainAdapter 训练排查与修复报告

## 1. 任务背景

本次目标是按照仓库说明跑通 `TrainAdapter` 的训练流程，并定位训练阶段的实际阻塞点。

排查分为两部分：

- 本地静态检查与代码修复
- 云端环境联调与最小数据集验证


## 2. 原始问题概览

在原始代码和云端环境中，训练无法直接启动，主要问题包括：

1. 本地与云端环境不一致
   - 本地没有 `torch`
   - 云端最初没有 `transformers`
   - 安装 `requirements.txt` 后默认装到了 `transformers 5.4.0`，与云端 `torch 2.1.2+cu118` 不兼容

2. 默认路径写死
   - 代码大量使用 `/root/autodl-tmp/...` 固定路径
   - 后续修改后默认路径切到了仓库内相对路径，需要通过环境变量显式指定云端真实路径

3. 训练链路设计存在根本问题
   - 原实现是 “模型生成文本 -> 正则解析坐标 -> 计算损失”
   - 该链路不可微，无法把 loss 回传到 Adapter

4. 数据前处理依赖不明确
   - 训练需要 `gRefCOCO` 风格数据
   - 当前云端已有的 `refcoco_train100` 是对话式 bbox 数据，不能直接用于本仓库现有标注脚本

5. 多处代码级错误
   - 命令行参数未正确覆盖配置
   - `cross_attention.py` 缺失 `F` 导入
   - `trainer.py` 对 mock Qwen 调用 `.parameters()` 会报错
   - `grid_points` 数据结构是三层列表，而 loss 期望二维点列表


## 3. 本地代码修改说明

本次实际修改了以下文件：

- [train.py](/Users/unf01d/teacher/TrainAdapter/src/train.py)
- [training/trainer.py](/Users/unf01d/teacher/TrainAdapter/src/training/trainer.py)
- [training/config.py](/Users/unf01d/teacher/TrainAdapter/src/training/config.py)
- [data/dataset.py](/Users/unf01d/teacher/TrainAdapter/src/data/dataset.py)
- [models/adapter.py](/Users/unf01d/teacher/TrainAdapter/src/models/adapter.py)
- [models/cross_attention.py](/Users/unf01d/teacher/TrainAdapter/src/models/cross_attention.py)
- [loss/hungarian_loss.py](/Users/unf01d/teacher/TrainAdapter/src/loss/hungarian_loss.py)

### 3.1 `src/train.py`

主要修改：

- 增加 `FrozenBackbone`，在本地缺少真实 Qwen 权重时提供轻量级 fallback
- 修复命令行参数到嵌套配置对象的映射
- 为 `adapter` 增加 `num_output_points` 配置支持
- 加入对本地 Qwen 路径不存在时的显式报错与 fallback
- 优化优化器配置传参逻辑

结果：

- `--batch_size`、`--num_epochs`、`--data_root` 等参数终于生效
- 没有 Qwen 时也能做 smoke test

### 3.2 `src/training/trainer.py`

主要修改：

- 将原来的“生成文本坐标”流程改为“直接预测坐标点”
- 新增 `forward_batch`
- 对 `qwen_model.parameters()` 调用增加保护
- 增加训练前和异常 batch 后的 `zero_grad`
- 修复 scheduler 中对 `np` 的使用

结果：

- 训练链路变为可回传梯度的监督学习流程
- mock/fallback backbone 不再在冻结阶段崩溃

### 3.3 `src/models/adapter.py`

主要修改：

- 为标准版和轻量版 Adapter 增加 `point_head`
- 新增 `predict_points`
- 补齐轻量版缺失的 `get_trainable_parameters` 和 `get_parameter_count`

结果：

- Adapter 能直接输出归一化点坐标和点置信度
- 轻量版训练不会再因为接口缺失报错

### 3.4 `src/loss/hungarian_loss.py`

主要修改：

- 保留原文本解析 loss 作为兼容分支
- 新增 tensor 版 `_tensor_loss`
- 支持直接接收 `pred_points` 和 `pred_logits`
- `scipy` 不可用时，增加贪心匹配 fallback

结果：

- 训练使用可微分的点匹配损失
- 不再依赖“文本解析坐标”进行训练

### 3.5 `src/data/dataset.py`

主要修改：

- 新增 `SimpleTokenizer` fallback
- tokenizer 加载失败时不直接中断
- 新增 `_normalize_grid_points`，把三层 `grid_points` 拍平成 `[[x, y], ...]`

结果：

- 无真实 tokenizer 时可以继续做本地/云端测试
- 修复了 loss 输入 shape 不匹配的问题

### 3.6 `src/models/cross_attention.py`

主要修改：

- 增加 `torch.nn.functional as F` 导入

结果：

- 解决 `F.softmax` 直接报错问题

### 3.7 `src/training/config.py`

主要修改：

- 默认路径改为基于项目根目录推导
- 支持环境变量：
  - `TRAIN_ADAPTER_DATA_ROOT`
  - `TRAIN_ADAPTER_QWEN_PATH`
  - `TRAIN_ADAPTER_SAVE_DIR`

结果：

- 便于在本地与云端切换路径
- 避免继续硬编码单一环境


## 4. 云端排查过程

云端机器：

- Ubuntu 22.04
- GPU: RTX 4090
- Python 3.10.8
- Torch 2.1.2+cu118

### 4.1 环境问题

初始问题：

- 无 `transformers`
- 安装依赖后为 `transformers 5.4.0`
- 与 `torch 2.1.2` 不兼容

处理结果：

- 降级到 `transformers==4.51.3`
- 验证通过：
  - `AutoTokenizer.from_pretrained('/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct', trust_remote_code=True, local_files_only=True)`

### 4.2 模型路径问题

云端实际模型目录：

- `/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct`

README/原代码默认目录：

- `/root/autodl-tmp/Qwen2.5-VL-7B-Instruct`

处理方式：

- 创建软链接进行对齐

### 4.3 数据准备问题

按“原仓库路线”处理：

1. 克隆 `gRefCOCO`
2. 从 Hugging Face 下载：
   - `grefs(unc).json`
   - `instances.json`
3. 从 COCO `train2014` 下载少量测试图

当前云端已存在：

- `/root/autodl-tmp/Data/grefs(unc).json`
- `/root/autodl-tmp/gRefCOCO/data/gRefCOCO/instances.json`


## 5. 新机器上的 100 张测试子集重建

在新机器 `ssh -p 28515 root@connect.nmb1.seetacloud.com` 上，重新按“原仓库路线”构建了 `gRefCOCO` 的 100 张测试子集。

### 5.1 标注文件

实际落盘路径：

- `/root/autodl-tmp/Data/grefs(unc).json`
- `/root/autodl-tmp/Data/grefs_unc_full.json`
- `/root/autodl-tmp/Data/grefs_unc_subset100.json`
- `/root/autodl-tmp/gRefCOCO/data/gRefCOCO/instances.json`

处理方式：

1. 先下载完整 `grefs(unc).json`
2. 备份为 `grefs_unc_full.json`
3. 从完整标注中截取前 100 张唯一图片对应的子集
4. 用 `grefs_unc_subset100.json` 覆盖当前活跃的 `/root/autodl-tmp/Data/grefs(unc).json`，供预处理脚本直接使用

子集统计：

- 子集图片数：100
- 子集标注条数：445

### 5.2 COCO 图片下载

最初通过 `https://images.cocodataset.org` 下载时，大量出现代理侧 `503` 和 TLS 超时。

最终确认：

- `https://images.cocodataset.org/...` 在当前代理环境下不稳定
- `http://images.cocodataset.org/...` 可稳定下载

因此最终改用 `http` 下载 100 张 `train2014` 图片到：

- `/root/autodl-tmp/gRefCOCO/Data/coco/train2014`

实际结果：

- 成功下载图片：100 / 100
- 其中 `COCO_train2014_000000000263.jpg` 首次为截断文件，后续已重新下载修复


## 6. 数据预处理验证结果

已在云端实际执行：

1. `python process_images.py`
2. `python add_grid.py`
3. `python process_masks_final_v3.py`

结果：

- `process_images.py` 最终成功处理 100 张
- `add_grid.py` 成功生成带网格图
- `process_masks_final_v3.py` 成功生成 `/root/autodl-tmp/Data/grefs_with_grids.json`

补充统计：

- `grefs_with_grids.json` 总记录数：445
- 覆盖唯一图片数：100
- 训练实际加载到 `train` split 的样本数：370


## 7. 训练验证结果

### 7.1 使用 fallback backbone 验证训练闭环

为避免 Qwen 视觉接口未适配阻塞整体验证，使用：

- `TRAIN_ADAPTER_QWEN_PATH=/root/autodl-tmp/does_not_exist`

强制走 fallback backbone。

执行命令：

```bash
export TRAIN_ADAPTER_QWEN_PATH=/root/autodl-tmp/does_not_exist
export TRAIN_ADAPTER_DATA_ROOT=/root/autodl-tmp/Data
cd /root/autodl-tmp/TrainAdapter/src
python train.py --preset default --batch_size 2 --num_epochs 1 --device cpu
```

实际结果：

- 训练成功完成 1 个 epoch
- 输出有效 loss
- 成功保存 checkpoint

训练日志中观测到：

- `Step 10, Loss: 453.8321, Avg Loss: 453.6953`
- `Epoch 1 completed, Avg Loss: 453.3332`
- `Saved checkpoint at step 13`

说明：

- 数据加载正常
- 前向传播正常
- 反向传播正常
- 优化器更新正常
- checkpoint 保存正常

结论：

最小训练闭环已经跑通。

### 7.2 新机器上真实 Qwen2.5-VL 训练验证

在新机器上，真实 Qwen 模型路径为：

- `/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct`

训练命令：

```bash
export TRAIN_ADAPTER_DATA_ROOT=/root/autodl-tmp/Data
export TRAIN_ADAPTER_QWEN_PATH=/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct
cd /root/autodl-tmp/TrainAdapter/src
python train.py --preset default --batch_size 1 --num_epochs 1 --device cuda
```

第一次真实训练暴露出新的 GPU 精度问题：

```text
mat1 and mat2 must have the same dtype, but got Half and Float
```

原因：

- Qwen 视觉/文本特征在 CUDA 下为 `float16`
- Adapter 参数默认是 `float32`
- 在交叉注意力和线性层矩阵乘法时触发 dtype mismatch

最终修复：

- 在 [trainer.py](/Users/unf01d/teacher/TrainAdapter/src/training/trainer.py) 的 `forward_batch` 中，把
  - `visual_features`
  - `text_embeddings`
  显式转换到 `next(self.adapter.parameters()).dtype`

修复后真实训练结果：

- 成功加载真实 `Qwen2.5-VL-7B-Instruct`
- 成功加载训练集：370 条 train 样本
- 在 RTX 4090 上稳定训练到 `step 316`
- 训练过程中 loss 持续下降，没有再出现 dtype 报错

训练日志中的关键观测值：

- `Step 10, Avg Loss: 355.3780`
- `Step 50, Avg Loss: 135.3199`
- `Step 100, Avg Loss: 70.5627`
- `Step 200, Avg Loss: 38.0204`
- `Step 300, Avg Loss: 27.6979`

说明：

- 真实 Qwen 视觉特征提取已可用
- Adapter 前向和反向传播已可用
- 100 张子集数据能够被真实模型稳定消费


## 8. 当前结论

本次工作已经完成以下目标：

1. 明确了原始训练失败并非单点问题，而是环境、路径、数据、代码多方面叠加
2. 修复了训练代码中的关键阻断项
3. 按原仓库路线在新机器上成功重建了 100 张 `gRefCOCO` 测试子集
4. 在新机器上用真实 `Qwen2.5-VL-7B-Instruct` 跑通了 GPU 训练
5. 修复了真实训练中的 CUDA 半精度 / 单精度 dtype 冲突

当前状态可以总结为：

- “数据预处理链路” 已跑通
- “最小训练闭环” 已跑通
- “真实 Qwen2.5-VL 接口适配” 已完成
- “真实 Qwen + 100 张子集训练” 已验证可用


## 9. 后续建议

建议后续按以下顺序继续：

1. 为当前 100 张子集单独准备一个干净的输出目录，避免混入早期 smoke test 产物
2. 增加 checkpoint 保存频率或手动保存节点，便于中途停止时保留权重
3. 如果要正式训练，再把子集扩展到完整 `gRefCOCO` / `RefCOCO` 数据
4. 根据 4090 显存情况尝试更大的 `batch_size` 或开启混合精度训练策略


## 10. 附录

本报告对应的主要代码修改文件：

- [train.py](/Users/unf01d/teacher/TrainAdapter/src/train.py)
- [trainer.py](/Users/unf01d/teacher/TrainAdapter/src/training/trainer.py)
- [config.py](/Users/unf01d/teacher/TrainAdapter/src/training/config.py)
- [dataset.py](/Users/unf01d/teacher/TrainAdapter/src/data/dataset.py)
- [adapter.py](/Users/unf01d/teacher/TrainAdapter/src/models/adapter.py)
- [cross_attention.py](/Users/unf01d/teacher/TrainAdapter/src/models/cross_attention.py)
- [hungarian_loss.py](/Users/unf01d/teacher/TrainAdapter/src/loss/hungarian_loss.py)
