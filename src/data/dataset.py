"""
数据集类：加载grefs_with_grids.json数据，构建训练样本
"""
import json
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoTokenizer


GRID_DIVISIONS = 10.0
DEFAULT_RELATION_KEYWORDS = (
    'left', 'right', 'top', 'bottom', 'front', 'behind', 'between', 'with', 'and',
    'center', 'middle', 'near', 'nearest', 'closest', 'far', 'furthest',
    'first', 'second', 'third', 'fourth', 'last'
)
DEFAULT_ORDINAL_KEYWORDS = (
    'first', 'second', 'third', 'fourth', 'fifth', 'last',
    'leftmost', 'rightmost', 'furthest', 'nearest'
)
DEFAULT_MULTI_ENTITY_KEYWORDS = (
    ' and ', ' with ', ' between ', ' beside ', ' next to '
)


def flatten_grid_points(grid_points):
    """将任意层级的 grid_points 拍平成 [[x, y], ...]。"""
    flattened = []

    def visit(node):
        if not isinstance(node, list):
            return
        if len(node) == 2 and all(isinstance(v, (int, float)) for v in node):
            flattened.append([float(node[0]), float(node[1])])
            return
        for child in node:
            visit(child)

    visit(grid_points)
    return flattened


def normalize_grid_points(grid_points, grid_divisions=GRID_DIVISIONS):
    """将 0..grid_divisions 的网格坐标转换为 0..1 的归一化坐标。"""
    normalized = []
    for x, y in flatten_grid_points(grid_points):
        normalized.append([float(x) / float(grid_divisions), float(y) / float(grid_divisions)])
    return normalized


def select_primary_grid_point(grid_points, grid_size=11):
    """
    为稠密 grid_points 选择一个主监督点。
    规则：取去重后的离散点集合，选择距离几何中心最近的点；
    如有并列，优先更靠近图像中心的点，再按坐标稳定排序。
    """
    discrete_points = []
    seen = set()
    for x, y in flatten_grid_points(grid_points):
        ix = int(round(x))
        iy = int(round(y))
        if not (0 <= ix < grid_size and 0 <= iy < grid_size):
            continue
        key = (ix, iy)
        if key in seen:
            continue
        seen.add(key)
        discrete_points.append([ix, iy])

    if not discrete_points:
        return None

    points_array = np.asarray(discrete_points, dtype=np.float32)
    centroid = points_array.mean(axis=0)
    center = np.asarray([(grid_size - 1) / 2.0, (grid_size - 1) / 2.0], dtype=np.float32)

    best_point = None
    best_key = None
    for point in discrete_points:
        point_arr = np.asarray(point, dtype=np.float32)
        centroid_distance = float(np.linalg.norm(point_arr - centroid))
        center_distance = float(np.linalg.norm(point_arr - center))
        sort_key = (round(centroid_distance, 6), round(center_distance, 6), point[1], point[0])
        if best_key is None or sort_key < best_key:
            best_key = sort_key
            best_point = point

    return best_point


def is_relation_query(query, relation_keywords=None):
    keywords = relation_keywords or DEFAULT_RELATION_KEYWORDS
    lowered = str(query).lower()
    return any(keyword in lowered for keyword in keywords)


def is_ordinal_query(query, ordinal_keywords=None):
    keywords = ordinal_keywords or DEFAULT_ORDINAL_KEYWORDS
    lowered = f" {str(query).lower()} "
    return any(keyword in lowered for keyword in keywords)


def is_multi_entity_query(query, multi_entity_keywords=None):
    keywords = multi_entity_keywords or DEFAULT_MULTI_ENTITY_KEYWORDS
    lowered = f" {str(query).lower()} "
    return any(keyword in lowered for keyword in keywords)


def classify_query_difficulty(
    sample,
    relation_keywords=None,
    ordinal_keywords=None,
    multi_entity_keywords=None
):
    query = str(sample.get('query', '')).strip()
    image_sample_count = int(sample.get('image_sample_count', 1))

    if is_ordinal_query(query, ordinal_keywords=ordinal_keywords):
        return 'ordinal'
    if is_multi_entity_query(query, multi_entity_keywords=multi_entity_keywords):
        return 'multi_entity'
    if is_relation_query(query, relation_keywords=relation_keywords):
        return 'spatial_relation'
    if image_sample_count > 1:
        return 'multi_entity'
    return 'easy_salient'


def build_grid_target(
    grid_points,
    grid_size=11,
    neighbor_soft_label_weight=0.3,
    use_primary_point_only=False
):
    """
    将离散网格点转换成 [grid_size * grid_size] 的 soft multi-hot 监督。
    真值点为 1.0，8 邻域平滑为 neighbor_soft_label_weight。
    """
    target = torch.zeros(grid_size * grid_size, dtype=torch.float32)
    if use_primary_point_only:
        primary_point = select_primary_grid_point(grid_points, grid_size=grid_size)
        discrete_points = [primary_point] if primary_point is not None else []
    else:
        discrete_points = flatten_grid_points(grid_points)

    for point in discrete_points:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            continue

        center_index = y * grid_size + x
        target[center_index] = 1.0

        if neighbor_soft_label_weight <= 0:
            continue

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    neighbor_index = ny * grid_size + nx
                    target[neighbor_index] = max(
                        target[neighbor_index].item(),
                        float(neighbor_soft_label_weight)
                    )

    return target


def split_indices_by_image(samples, val_ratio=0.2, seed=42):
    """
    按图片级切分，避免同图不同 query 同时落到 train/val。
    """
    image_to_indices = {}
    for idx, sample in enumerate(samples):
        image_to_indices.setdefault(sample['image_id'], []).append(idx)

    image_ids = sorted(image_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    if len(image_ids) <= 1:
        return list(range(len(samples))), []

    val_count = max(1, int(round(len(image_ids) * float(val_ratio))))
    val_count = min(val_count, len(image_ids) - 1)
    val_images = set(image_ids[:val_count])

    train_indices = []
    val_indices = []
    for image_id, indices in image_to_indices.items():
        if image_id in val_images:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return sorted(train_indices), sorted(val_indices)


def sample_farthest_points(points, max_points):
    """
    用 farthest point sampling 从区域点中选出代表点。
    第一个点取离几何中心最近的点，后续点依次取离已选集合最远的点。
    """
    if max_points <= 0 or not points:
        return []

    unique_points = []
    seen = set()
    for x, y in points:
        key = (round(float(x), 6), round(float(y), 6))
        if key in seen:
            continue
        seen.add(key)
        unique_points.append([float(x), float(y)])

    if len(unique_points) <= max_points:
        return unique_points

    points_array = np.asarray(unique_points, dtype=np.float32)
    centroid = points_array.mean(axis=0, keepdims=True)
    first_idx = int(np.argmin(np.linalg.norm(points_array - centroid, axis=1)))

    selected_indices = [first_idx]
    remaining_indices = set(range(len(unique_points))) - {first_idx}

    while remaining_indices and len(selected_indices) < max_points:
        remaining_list = sorted(remaining_indices)
        remaining_points = points_array[remaining_list]
        selected_points = points_array[selected_indices]
        # 为了覆盖区域，每轮选离当前已选集合最远的点。
        min_distances = np.linalg.norm(
            remaining_points[:, None, :] - selected_points[None, :, :],
            axis=-1
        ).min(axis=1)
        next_pos = int(np.argmax(min_distances))
        next_idx = remaining_list[next_pos]
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)

    return [unique_points[idx] for idx in selected_indices]


class SimpleTokenizer:
    """当本地没有可用 tokenizer 时的轻量级回退实现。"""
    def __init__(self, pad_token_id=0, unk_token_id=1, vocab_size=2048):
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.vocab_size = vocab_size

    def __call__(self, text, padding='max_length', truncation=True, max_length=512, return_tensors='pt'):
        tokens = text.split()
        token_ids = [self._token_to_id(tok) for tok in tokens][:max_length]
        attention_mask = [1] * len(token_ids)

        if padding == 'max_length' and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }

    def _token_to_id(self, token):
        return abs(hash(token)) % (self.vocab_size - 2) + 2


def collate_fn_pad_batch(batch):
    """
    自定义collate_fn: 处理不同尺寸的图像，将batch中所有图像在右下角padding至当前batch最大宽高。
    这样保持了图像左上角(0,0)坐标原点不变，真实标签(gt_points)无需任何调整。
    """
    # 提取基本字段
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    gt_points = [item['gt_points'] for item in batch]
    grid_targets = torch.stack([item['grid_target'] for item in batch])
    image_sizes = [item['image_size'] for item in batch]
    queries = [item['query'] for item in batch]
    instructions = [item['instruction'] for item in batch]
    relation_flags = torch.tensor(
        [1 if item.get('is_relation_query', False) else 0 for item in batch],
        dtype=torch.bool
    )
    image_ids = [item.get('image_id', '') for item in batch]
    
    images = [item['image'] for item in batch]
    grid_images = [item['grid_image'] for item in batch]
    
    # 两路图像都可能存在不同尺寸，padding 时取联合最大宽高。
    max_h = max(max(img.shape[1] for img in images), max(img.shape[1] for img in grid_images))
    max_w = max(max(img.shape[2] for img in images), max(img.shape[2] for img in grid_images))
    
    padded_images = []
    padded_grid_images = []
    
    for img, g_img in zip(images, grid_images):
        # 两路图像可能原始尺寸不同，需要分别计算 pad。
        img_pad_w = max_w - img.shape[2]
        img_pad_h = max_h - img.shape[1]
        grid_pad_w = max_w - g_img.shape[2]
        grid_pad_h = max_h - g_img.shape[1]

        if img_pad_w > 0 or img_pad_h > 0:
            img_padded = torch.nn.functional.pad(img, (0, img_pad_w, 0, img_pad_h), value=0)
        else:
            img_padded = img

        if grid_pad_w > 0 or grid_pad_h > 0:
            g_img_padded = torch.nn.functional.pad(g_img, (0, grid_pad_w, 0, grid_pad_h), value=0)
        else:
            g_img_padded = g_img
            
        padded_images.append(img_padded)
        padded_grid_images.append(g_img_padded)
        
    return {
        'image': torch.stack(padded_images),
        'grid_image': torch.stack(padded_grid_images),
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'gt_points': gt_points,
        'grid_target': grid_targets,
        'image_size': image_sizes,
        'query': queries,
        'instruction': instructions,
        'is_relation_query': relation_flags,
        'image_id': image_ids
    }


class CoordinateDataset(Dataset):
    """
    坐标数据集：加载grefs数据，构建训练样本
    """
    def __init__(self, 
                 data_root,
                 annotation_file,
                 image_dir='images',
                 grid_image_dir='grid_images',
                 tokenizer_path='/root/autodl-tmp/Qwen2.5-VL-7B-Instruct',
                 image_size=(448, 448),
                 max_length=512,
                 transform=None,
                 num_output_points=4,
                 target_point_strategy='fps',
                 target_coordinate_mode='normalized_grid',
                 output_mode='grid_logits',
                 grid_size=11,
                 neighbor_soft_label_weight=0.3,
                 use_primary_grid_target=False,
                 relation_keywords=None,
                 ordinal_keywords=None,
                 multi_entity_keywords=None):
        """
        Args:
            data_root: 数据根目录
            annotation_file: 标注文件路径（相对于data_root）
            image_dir: 原始图像文件夹名称
            grid_image_dir: 网格图像文件夹名称
            tokenizer_path: 分词器路径
            image_size: 图像尺寸
            max_length: 文本最大长度
            transform: 图像变换
        """
        self.data_root = data_root
        self.image_dir = image_dir
        self.grid_image_dir = grid_image_dir
        self.image_size = image_size
        self.max_length = max_length
        self.transform = transform
        self.num_output_points = num_output_points
        self.target_point_strategy = target_point_strategy
        self.target_coordinate_mode = target_coordinate_mode
        self.output_mode = output_mode
        self.grid_size = grid_size
        self.neighbor_soft_label_weight = neighbor_soft_label_weight
        self.use_primary_grid_target = use_primary_grid_target
        self.relation_keywords = tuple(relation_keywords or DEFAULT_RELATION_KEYWORDS)
        self.ordinal_keywords = tuple(ordinal_keywords or DEFAULT_ORDINAL_KEYWORDS)
        self.multi_entity_keywords = tuple(multi_entity_keywords or DEFAULT_MULTI_ENTITY_KEYWORDS)
        
        # 加载分词器
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        
        # 加载标注数据
        annotation_path = os.path.join(data_root, annotation_file)
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 预处理数据
        self.samples = self._preprocess_annotations()
        
        print(f"Loaded {len(self.samples)} samples from {annotation_path}")

    def _load_tokenizer(self, tokenizer_path):
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception as e:
            print(f"Warning: failed to load tokenizer from {tokenizer_path}: {e}")
            print("Falling back to SimpleTokenizer for local smoke tests")
            return SimpleTokenizer()
    
    def _preprocess_annotations(self):
        """
        预处理标注数据，构建样本列表
        """
        samples = []
        
        image_counts = {}
        normalized_annotations = []

        for idx, ann in enumerate(self.annotations):
            img_id = ann.get('file_name') or ann.get('img_id')
            # 确保 img_id 有效且为字符串，如果是数字，转换为旧的COCO格式
            if isinstance(img_id, int):
                img_id = f"COCO_train2014_{img_id:012d}.jpg"
                
            sentences = ann.get('sentences', [])
            grid_points = ann.get('grid_points', [])
            
            # 很多时候JSON中没有grid_image_path，需要自己推导
            grid_image_path = ann.get('grid_image_path', str(img_id))
            
            if not img_id or not sentences or not grid_points:
                continue

            normalized_annotations.append((idx, ann, str(img_id)))
            image_counts[str(img_id)] = image_counts.get(str(img_id), 0) + 1

        for idx, ann, img_id in normalized_annotations:
            sentences = ann.get('sentences', [])
            grid_points = ann.get('grid_points', [])
            
            # 使用句子的 'sent' 或第一个元素的文本
            query = sentences[0].get('sent', sentences[0]) if isinstance(sentences[0], dict) else sentences[0]
            
            # 构建样本
            sample = {
                'image_id': str(img_id),
                'query': query,  # 使用第一个句子作为查询
                'grid_points': grid_points,  # 真值坐标点
                'grid_image_path': grid_image_path,
                'index': idx,
                'image_sample_count': image_counts.get(str(img_id), 1)
            }
            sample['difficulty_tag'] = classify_query_difficulty(
                sample,
                relation_keywords=self.relation_keywords,
                ordinal_keywords=self.ordinal_keywords,
                multi_entity_keywords=self.multi_entity_keywords
            )
            
            # 过滤掉本地实际不存在的图像，以防止DataLoader由于文件不存在而崩溃
            image_path = os.path.join(self.data_root, self.image_dir, sample['image_id'])
            if not os.path.exists(image_path):
                continue
                
            samples.append(sample)
        
        return samples
    
    def _build_instruction(self, query):
        """
        构建文本指令
        
        Args:
            query: 原始查询文本
            
        Returns:
            完整的指令文本
        """
        instruction_templates = [
            f"Given the grid coordinate system, locate the referent described as '{query}' in the image and predict its target coordinates.",
            f"Use the grid as spatial guidance to find '{query}' in the image and return the most likely target points.",
            f"Locate '{query}' with the help of the image grid and predict the corresponding target coordinates.",
            f"Identify where '{query}' is in the image according to the grid and output the most likely target points."
        ]
        
        template = random.choice(instruction_templates)
        
        return template

    def _normalize_grid_points(self, grid_points):
        """
        将区域网格点转换为训练监督目标。
        默认输出为归一化后的最多 num_output_points 个代表点。
        """
        flattened_points = flatten_grid_points(grid_points)
        if self.target_coordinate_mode == 'normalized_grid':
            converted_points = normalize_grid_points(flattened_points)
        else:
            converted_points = flattened_points

        if self.target_point_strategy == 'fps':
            return sample_farthest_points(converted_points, self.num_output_points)
        if self.target_point_strategy == 'all':
            return converted_points[:self.num_output_points]
        raise ValueError(f"Unknown target point strategy: {self.target_point_strategy}")
    
    def _load_image(self, image_path):
        """
        加载并预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像张量
        """
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # 默认预处理：不再强制resize，保留原图分辨率
            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取样本
        
        Returns:
            sample: 包含以下字段的字典
                - image: 原始图像张量
                - grid_image: 网格图像张量
                - input_ids: 文本输入的token ids
                - attention_mask: 注意力掩码
                - gt_points: 真值坐标点 [[x1,y1], [x2,y2], ...]
                - image_size: 图像尺寸 (width, height)
                - query: 原始查询文本
        """
        sample = self.samples[idx]
        
        # 1. 加载原始图像
        image_path = os.path.join(self.data_root, self.image_dir, sample['image_id'])
        image = self._load_image(image_path)
        
        # 2. 加载网格图像
        grid_image_path = os.path.join(self.data_root, self.grid_image_dir, 
                                       os.path.basename(sample['grid_image_path']))
        if os.path.exists(grid_image_path):
            grid_image = self._load_image(grid_image_path)
        else:
            # 如果网格图像不存在，使用原始图像（后期会添加网格）
            grid_image = image.clone()
        
        # 3. 构建文本指令
        instruction = self._build_instruction(sample['query'])
        
        # 4. 编码文本
        encoding = self.tokenizer(
            instruction,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)  # [max_length]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_length]
        
        # 5. 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # 6. 构建输出
        normalized_all_points = normalize_grid_points(sample['grid_points'])
        if self.output_mode == 'grid_logits':
            gt_points = normalized_all_points
        else:
            gt_points = self._normalize_grid_points(sample['grid_points'])

        data = {
            'image': image,
            'grid_image': grid_image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'gt_points': gt_points,
            'grid_target': build_grid_target(
                sample['grid_points'],
                grid_size=self.grid_size,
                neighbor_soft_label_weight=self.neighbor_soft_label_weight,
                use_primary_point_only=self.use_primary_grid_target
            ),
            'image_size': (image_width, image_height),
            'query': sample['query'],
            'instruction': instruction,
            'target_coordinate_mode': self.target_coordinate_mode,
            'is_relation_query': is_relation_query(sample['query'], self.relation_keywords),
            'is_ordinal_query': is_ordinal_query(sample['query'], self.ordinal_keywords),
            'is_multi_entity_query': is_multi_entity_query(sample['query'], self.multi_entity_keywords),
            'difficulty_tag': sample['difficulty_tag'],
            'image_sample_count': sample.get('image_sample_count', 1),
            'image_id': sample['image_id']
        }
        
        return data


class CoordinateDatasetV2(Dataset):
    """
    改进版数据集：支持更多数据增强和负样本
    """
    def __init__(self, 
                 data_root,
                 annotation_file,
                 image_dir='images',
                 grid_image_dir='grid_images',
                 tokenizer_path='Qwen2.5-VL-7B-Instruct',
                 image_size=(448, 448),
                 max_length=512,
                 transform=None,
                 num_output_points=4,
                 target_point_strategy='fps',
                 target_coordinate_mode='normalized_grid',
                 output_mode='grid_logits',
                 grid_size=11,
                 neighbor_soft_label_weight=0.3,
                 use_primary_grid_target=False,
                 relation_keywords=None,
                 use_negative_samples=True,
                 negative_sample_ratio=0.2):
        """
        Args:
            use_negative_samples: 是否使用负样本（没有目标物体的样本）
            negative_sample_ratio: 负样本比例
        """
        self.data_root = data_root
        self.image_dir = image_dir
        self.grid_image_dir = grid_image_dir
        self.image_size = image_size
        self.max_length = max_length
        self.transform = transform
        self.num_output_points = num_output_points
        self.target_point_strategy = target_point_strategy
        self.target_coordinate_mode = target_coordinate_mode
        self.output_mode = output_mode
        self.grid_size = grid_size
        self.neighbor_soft_label_weight = neighbor_soft_label_weight
        self.use_primary_grid_target = use_primary_grid_target
        self.relation_keywords = tuple(relation_keywords or DEFAULT_RELATION_KEYWORDS)
        self.use_negative_samples = use_negative_samples
        self.negative_sample_ratio = negative_sample_ratio
        
        # 加载分词器
        self.tokenizer = CoordinateDataset._load_tokenizer(self, tokenizer_path)
        
        # 加载标注数据
        annotation_path = os.path.join(data_root, annotation_file)
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 预处理数据
        self.samples = self._preprocess_annotations()
        
        # 添加负样本
        if use_negative_samples:
            self._add_negative_samples()
        
        print(f"Loaded {len(self.samples)} samples from {annotation_path}")
    
    def _preprocess_annotations(self):
        """预处理标注数据"""
        samples = []
        
        for idx, ann in enumerate(self.annotations):
            img_id = ann.get('img_id')
            sentences = ann.get('sentences', [])
            grid_points = ann.get('grid_points', [])
            grid_image_path = ann.get('grid_image_path', '')
            
            if not img_id or not sentences:
                continue
            
            # 构建样本
            sample = {
                'image_id': img_id,
                'query': sentences[0],
                'grid_points': grid_points if grid_points else [],
                'grid_image_path': grid_image_path,
                'has_target': len(grid_points) > 0,
                'index': idx
            }
            samples.append(sample)
        
        return samples
    
    def _add_negative_samples(self):
        """添加负样本"""
        positive_samples = [s for s in self.samples if s['has_target']]
        num_negative = int(len(positive_samples) * self.negative_sample_ratio)
        
        # 从正样本中随机选择一些作为负样本模板
        import random
        negative_candidates = random.sample(positive_samples, min(num_negative, len(positive_samples)))
        
        for sample in negative_candidates:
            # 创建负样本（相同图像，但查询不相关的物体）
            negative_sample = sample.copy()
            negative_sample['query'] = self._get_negative_query(sample['query'])
            negative_sample['grid_points'] = []  # 负样本没有真值点
            negative_sample['has_target'] = False
            self.samples.append(negative_sample)
    
    def _get_negative_query(self, original_query):
        """获取负样本查询（随机选择一个不相关的查询）"""
        negative_queries = [
            "一个不存在的物体",
            "背景区域",
            "空白处",
            "随机位置",
            "无目标"
        ]
        import random
        return random.choice(negative_queries)
    
    def _build_instruction(self, query, has_target=True):
        """构建文本指令"""
        if has_target:
            return CoordinateDataset._build_instruction(self, query)
        else:
            # 负样本指令
            return f"Locate '{query}' in the image. If it does not exist, answer 'not found'."
    
    def __getitem__(self, idx):
        """获取样本"""
        sample = self.samples[idx]
        
        # 1. 加载原始图像
        image_path = os.path.join(self.data_root, self.image_dir, sample['image_id'])
        image = CoordinateDataset._load_image(self, image_path)
        
        # 2. 加载网格图像
        grid_image_path = os.path.join(self.data_root, self.grid_image_dir, 
                                       os.path.basename(sample['grid_image_path']))
        if os.path.exists(grid_image_path):
            grid_image = CoordinateDataset._load_image(self, grid_image_path)
        else:
            grid_image = image.clone()
        
        # 3. 构建文本指令
        instruction = self._build_instruction(sample['query'], sample['has_target'])
        
        # 4. 编码文本
        encoding = self.tokenizer(
            instruction,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 5. 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # 6. 构建输出
        data = {
            'image': image,
            'grid_image': grid_image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'gt_points': CoordinateDataset._normalize_grid_points(self, sample['grid_points']),
            'grid_target': build_grid_target(
                sample['grid_points'],
                grid_size=self.grid_size,
                neighbor_soft_label_weight=self.neighbor_soft_label_weight,
                use_primary_point_only=self.use_primary_grid_target
            ),
            'image_size': (image_width, image_height),
            'query': sample['query'],
            'instruction': instruction,
            'has_target': sample['has_target'],
            'target_coordinate_mode': self.target_coordinate_mode,
            'is_relation_query': is_relation_query(sample['query'], self.relation_keywords),
            'image_id': sample['image_id']
        }
        
        return data


if __name__ == "__main__":
    # 测试代码
    print("=== 测试CoordinateDataset ===")
    
    data_root = "d:/VSCode_MyCode/Adapter/Data"
    annotation_file = "grefs_with_grids.json"
    
    # 检查文件是否存在
    import os
    annotation_path = os.path.join(data_root, annotation_file)
    if not os.path.exists(annotation_path):
        print(f"警告: 标注文件不存在 {annotation_path}")
        print("创建模拟数据进行测试")
        
        # 创建模拟数据
        os.makedirs(data_root, exist_ok=True)
        mock_data = [
            {
                "img_id": "test_image_1.jpg",
                "sentences": ["查找红色汽车"],
                "grid_points": [[125, 240], [300, 410]],
                "grid_image_path": "grid_images/test_image_1_grid.jpg"
            },
            {
                "img_id": "test_image_2.jpg",
                "sentences": ["定位行人"],
                "grid_points": [[200, 350]],
                "grid_image_path": "grid_images/test_image_2_grid.jpg"
            }
        ]
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, ensure_ascii=False, indent=2)
    
    # 创建数据集
    dataset = CoordinateDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        tokenizer_path="Qwen2.5-VL-7B-Instruct"
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"样本键: {list(sample.keys())}")
    print(f"图像形状: {sample['image'].shape}")
    print(f"网格图像形状: {sample['grid_image'].shape}")
    print(f"输入ID形状: {sample['input_ids'].shape}")
    print(f"真值点: {sample['gt_points']}")
    print(f"图像尺寸: {sample['image_size']}")
    print(f"查询: {sample['query']}")
    print(f"指令: {sample['instruction']}")
