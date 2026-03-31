"""
推理脚本：使用训练好的Coordinate Adapter进行推理
"""
import os
import sys
import argparse
import json

import torch
from PIL import Image

# 添加src到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SimpleTokenizer
from training.config import Config, get_config
from train import create_adapter, load_qwen_model, setup_transforms


class CoordinateAdapterInference:
    """
    Coordinate Adapter推理类
    """

    def __init__(
        self,
        adapter_path,
        qwen_model_path='Qwen2.5-VL-7B-Instruct',
        adapter_type='standard',
        device='cuda',
        config=None,
        num_output_points=None,
        grid_size=None
    ):
        """
        Args:
            adapter_path: Adapter模型路径
            qwen_model_path: Qwen2.5-VL模型路径
            adapter_type: Adapter类型
            device: 设备
            config: 可选配置对象
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config or get_config('default')
        self.config.model.qwen_model_path = qwen_model_path
        self.config.model.adapter_type = adapter_type
        if num_output_points is not None:
            self.config.model.num_output_points = num_output_points
        if grid_size is not None:
            self.config.model.grid_size = grid_size

        self.qwen_model, self.tokenizer = load_qwen_model(self.config.model.qwen_model_path, self.device)
        if hasattr(self.qwen_model, 'visual_dim'):
            self.config.model.visual_dim = self.qwen_model.visual_dim

        self.adapter = create_adapter(self.config)
        self._load_adapter_weights(adapter_path)

        if self.tokenizer is None:
            self.tokenizer = SimpleTokenizer()

        _, self.transform = setup_transforms(self.config)
        print(f"Model loaded successfully. Using device: {self.device}")

    def _load_adapter_weights(self, adapter_path):
        """加载Adapter权重。"""
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter checkpoint not found at {adapter_path}")
            print("Using randomly initialized adapter")
            self.adapter.to(self.device)
            self.adapter.eval()
            return

        checkpoint = torch.load(adapter_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        missing_keys, unexpected_keys = self.adapter.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading adapter: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading adapter: {unexpected_keys}")

        self.adapter.to(self.device)
        self.adapter.eval()
        print(f"Loaded adapter from {adapter_path}")

    def preprocess_image(self, image_path, grid_image_path=None):
        """
        预处理图像

        Args:
            image_path: 原始图像路径
            grid_image_path: 可选网格图像路径

        Returns:
            image_tensor, grid_image_tensor, original_size
        """
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        if grid_image_path and os.path.exists(grid_image_path):
            grid_image = Image.open(grid_image_path).convert('RGB')
            grid_image_tensor = self.transform(grid_image).unsqueeze(0).to(self.device)
        else:
            grid_image_tensor = image_tensor.clone()

        return image_tensor, grid_image_tensor, original_size

    def _build_instruction(self, query):
        return f"请根据网格坐标系，在图像中定位'{query}'的位置，输出坐标点[x,y]格式。"

    def _encode_text(self, instruction):
        encoding = self.tokenizer(
            instruction,
            padding='max_length',
            truncation=True,
            max_length=self.config.data.max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)

    def _to_absolute_points(self, normalized_points, image_size):
        width, height = image_size
        return [
            [round(float(x) * width, 2), round(float(y) * height, 2)]
            for x, y in normalized_points
        ]

    def predict(self, image_path, query, return_text=False, grid_image_path=None, return_normalized=False):
        """
        预测坐标

        Args:
            image_path: 图像路径
            query: 查询文本
            return_text: 是否返回详细推理摘要
            grid_image_path: 可选网格图像路径
            return_normalized: 是否直接返回归一化预测详情

        Returns:
            绝对像素坐标列表，或详细预测摘要
        """
        image, grid_image, original_size = self.preprocess_image(
            image_path,
            grid_image_path=grid_image_path
        )
        instruction = self._build_instruction(query)
        input_ids, attention_mask = self._encode_text(instruction)
        adapter_dtype = next(self.adapter.parameters()).dtype

        with torch.no_grad():
            visual_features = self.qwen_model.encode_image(image)
            if visual_features.dtype != adapter_dtype:
                visual_features = visual_features.to(dtype=adapter_dtype)

            enhanced_features = self.adapter(image, grid_image, visual_features)

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
                    top_k=self.adapter.num_output_points
                )
            else:
                pred_points, pred_logits = self.adapter.predict_point_regression(
                    enhanced_features,
                    text_features=text_embeddings,
                    attention_mask=attention_mask
                )

        normalized_points = pred_points[0].detach().cpu().tolist()
        scores = torch.sigmoid(pred_logits[0]).detach().cpu().tolist()
        absolute_points = self._to_absolute_points(normalized_points, original_size)
        summary = {
            'query': query,
            'instruction': instruction,
            'output_mode': getattr(self.adapter, 'output_mode', 'point_regression'),
            'normalized_points': [[round(float(x), 4), round(float(y), 4)] for x, y in normalized_points],
            'absolute_points': absolute_points,
            'scores': [round(float(score), 4) for score in scores],
            'image_size': list(original_size)
        }

        if return_normalized:
            return summary
        if return_text:
            return absolute_points, json.dumps(summary, ensure_ascii=False, indent=2)
        return absolute_points

    def batch_predict(self, image_paths, queries):
        """
        批量预测

        Args:
            image_paths: 图像路径列表
            queries: 查询列表

        Returns:
            results: 结果列表
        """
        results = []

        for image_path, query in zip(image_paths, queries):
            try:
                points = self.predict(image_path, query)
                results.append({
                    'image_path': image_path,
                    'query': query,
                    'points': points,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'query': query,
                    'points': [],
                    'status': 'error',
                    'error_message': str(e)
                })

        return results

    def save_prediction(self, image_path, query, points, save_dir='predictions'):
        """
        保存预测结果（包括可视化）

        Args:
            image_path: 图像路径
            query: 查询文本
            points: 预测的像素坐标点
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        image_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, image_name)

        image = Image.open(image_path).convert('RGB')

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)

        for idx, point in enumerate(points):
            x, y = point
            circle = patches.Circle((x, y), radius=10, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.plot(x, y, 'ro', markersize=5)
            ax.text(x + 15, y - 15, f'{idx + 1}', color='red', fontsize=12, weight='bold')

        ax.set_title(f'Query: {query}', fontsize=14)
        ax.axis('off')

        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        result = {
            'image_path': image_path,
            'query': query,
            'points': points,
            'image_size': image.size
        }

        json_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Prediction saved to {save_path} and {json_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Coordinate Adapter Inference')

    parser.add_argument('--adapter_path', type=str, required=True, help='Adapter模型路径')
    parser.add_argument('--config', type=str, default=None, help='训练时保存的config.json路径')
    parser.add_argument('--qwen_model_path', type=str, default='Qwen2.5-VL-7B-Instruct', help='Qwen模型路径')
    parser.add_argument('--adapter_type', type=str, default='standard', choices=['standard', 'lightweight'], help='Adapter类型')
    parser.add_argument('--image_path', type=str, required=True, help='图像路径')
    parser.add_argument('--grid_image_path', type=str, default=None, help='可选的网格图像路径')
    parser.add_argument('--query', type=str, required=True, help='查询文本')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--num_output_points', type=int, default=None, help='覆盖config中的top-k输出数量')
    parser.add_argument('--grid_size', type=int, default=None, help='覆盖config中的网格边长')
    parser.add_argument('--save_pred', action='store_true', help='保存预测结果')
    parser.add_argument('--save_dir', type=str, default='predictions', help='保存目录')
    parser.add_argument('--return_text', action='store_true', help='返回详细预测摘要')
    parser.add_argument('--return_normalized', action='store_true', help='输出归一化坐标和分数')

    args = parser.parse_args()

    config = Config.load(args.config) if args.config and os.path.exists(args.config) else get_config('default')
    if args.qwen_model_path:
        config.model.qwen_model_path = args.qwen_model_path
    if args.adapter_type:
        config.model.adapter_type = args.adapter_type
    if args.num_output_points is not None:
        config.model.num_output_points = args.num_output_points
    if args.grid_size is not None:
        config.model.grid_size = args.grid_size

    inferencer = CoordinateAdapterInference(
        adapter_path=args.adapter_path,
        qwen_model_path=config.model.qwen_model_path,
        adapter_type=config.model.adapter_type,
        device=args.device,
        config=config,
        num_output_points=args.num_output_points,
        grid_size=args.grid_size
    )

    if args.return_normalized:
        prediction_payload = inferencer.predict(
            args.image_path,
            args.query,
            grid_image_path=args.grid_image_path,
            return_normalized=True
        )
        points = prediction_payload['absolute_points']
        print(json.dumps(prediction_payload, ensure_ascii=False, indent=2))
    elif args.return_text:
        points, text = inferencer.predict(
            args.image_path,
            args.query,
            return_text=True,
            grid_image_path=args.grid_image_path
        )
        print(f"Prediction summary: {text}")
    else:
        points = inferencer.predict(
            args.image_path,
            args.query,
            grid_image_path=args.grid_image_path
        )

    print(f"Query: {args.query}")
    print(f"Predicted points: {points}")

    if args.save_pred:
        inferencer.save_prediction(
            args.image_path,
            args.query,
            points,
            args.save_dir
        )


if __name__ == '__main__':
    main()
