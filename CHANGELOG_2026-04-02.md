# 2026-04-02 改动总结：颜色/方位/顺序查询准确率优化

## 诊断结果

| 问题 | 根因 | 严重度 |
|------|------|--------|
| 顺序查询不会做（"左边第二个"） | gRefCOCO 标注错误——"第二个"标成了第一个，监督信号本身就是错的 | 致命，数据层无法修 |
| 颜色识别不准（"红色的车"） | 文本信号进入太晚（CrossAttention 不含文本）+ 文本被池化成单向量后广播到所有位置，丢失了细粒度语义 | 严重 |
| 方位判断不准（"左边的人"） | 同上 + grid logits 没有位置编码，模型不知道哪个格子对应"左"哪个对应"右" | 严重 |
| 中文查询不被识别 | 关键词列表只有英文，中文查询全部归为 easy_salient，得不到过采样 | 中等 |

---

## 改动清单

### 1. 过滤顺序查询（止血）
- **文件**: `src/data/dataset.py`, `src/training/config.py`, `src/train.py`
- **内容**: 新增 `filter_ordinal_queries` 配置项（默认 `True`），预处理阶段跳过 `difficulty_tag == 'ordinal'` 的样本
- **原因**: 标注本身是错的，喂给模型只产生噪声
- **恢复**: 设 `filter_ordinal_queries: false`（不建议，除非标注修正了）

### 2. 中文关键词支持
- **文件**: `src/data/dataset.py`, `src/training/config.py`
- **内容**: 在四类关键词列表中补充中文
  - 方位: 左/右/上/下/前/后/之间/旁边/附近/中间/中央/靠近/远离
  - 顺序: 第一/第二/第三/第四/第五/最后/最左/最右/最远/最近
  - 颜色: 红/蓝/绿/黄/黑/白/棕/橙/紫/粉/灰
  - 多实体: 和/与/之间/旁边/边上

### 3. TextGuidedCrossAttention（核心架构改动）
- **文件**: `src/models/cross_attention.py`, `src/models/adapter.py`
- **内容**: 原来的 CrossAttention 只做 visual × grid，文本完全缺席。新增 `TextGuidedCrossAttention`：
  - 并行运行 visual×grid 和 visual×text 两个 cross-attention 分支
  - 门控融合：每个视觉 token 自适应决定更信任网格还是文本增强
- **效果**: 空间特征融合阶段就能感知"红色""左边"等语义，不用等到最后一步才引入文本

### 4. 移除文本广播，强化 token-level 交互
- **文件**: `src/models/adapter.py`（`predict_grid_logits` 方法）
- **内容**: 旧逻辑将 text_summary 池化成 [B,768] 后 expand 广播到所有 196 个视觉 token——"左边"和"右边"给每个位置的信号完全一样。改为全部使用 token-level 的 `text_condition`，每个视觉 token 独立从文本 token 中提取相关信息
- **效果**: "左边"的信号可以选择性地增强左侧区域的 token

### 5. Grid 位置编码
- **文件**: `src/models/adapter.py`
- **内容**: 新增 `grid_position_embedding` [1, 121, D]，通过 text_summary × position_embedding 的点积产生空间偏置加到 grid_logits 上
- **效果**: 模型能学到"左 = 小 x 坐标""右 = 大 x 坐标"的对应关系

### 6. Ordinal 评估指标
- **文件**: `src/data/dataset.py`（collate 函数）, `src/training/trainer.py`
- **内容**: 新增 `ordinal_acc_top4` 指标，验证日志中会显示 `Ordinal Acc@Top4: xx%`

### 7. 旧 checkpoint 兼容
- **文件**: `src/training/trainer.py`（`_migrate_checkpoint_keys` + `load_checkpoint`）
- **内容**: 加载旧 checkpoint 时自动将 `cross_attention.*` 映射为 `cross_attention.grid_cross_attn.*`，让旧的网格注意力权重被复用

---

## 调用顺序变化

```
旧:  encode_image → adapter(visual) → encode_text → predict_grid_logits(text)
新:  encode_image + encode_text → adapter(visual, text) → predict_grid_logits(text)
```

文本现在在 adapter.forward() 阶段就参与，不再是最后一步才进来。

---

## 使用方式

```bash
# 基于旧 checkpoint 继续训练（推荐）
cd src && python train.py --preset default \
  --resume /path/to/old_best_model.pth \
  --resume_as_init \
  --device cuda

# 从零开始训练
cd src && python train.py --preset default --device cuda
```

### --resume_as_init 做了什么
- `strict=False` 加载，忽略 key 不匹配
- 自动迁移旧 cross_attention 权重（日志打印迁移数量）
- 新增层（text_cross_attn, merge_gate, grid_position_embedding）随机初始化
- optimizer 和 scheduler 重置
- **~85% 旧权重被复用**，比从零训练收敛快

---

## 影响的文件

| 文件 | 改动 |
|------|------|
| `src/data/dataset.py` | 中文关键词、ordinal 过滤、collate 新增 is_ordinal_query |
| `src/models/adapter.py` | 位置编码、移除文本广播、forward 接受 text_features |
| `src/models/cross_attention.py` | 新增 TextGuidedCrossAttention 类 |
| `src/training/config.py` | 中文关键词、filter_ordinal_queries 配置 |
| `src/training/trainer.py` | checkpoint 迁移、ordinal 指标、调用顺序调整 |
| `src/train.py` | 传递 filter_ordinal_queries 到 Dataset |
| `src/inference.py` | 调用顺序调整（先 encode_text 再 adapter forward） |
| `src/visualize_training_results.py` | 同上 |

---

## 后续可做的事

1. **修正 gRefCOCO 顺序标注**: 如果能重新标注或用规则修正 ordinal 查询的 grid_points，可以关闭过滤，让模型学习顺序
2. **按查询类型加权 loss**: 对 relation/color 查询施加更高的 loss 权重（如 1.5x-2x），目前 loss 对所有查询类型一视同仁
3. **Hard example mining**: 训练中根据预测错误率动态调整采样权重，让模型多练习困难样本
