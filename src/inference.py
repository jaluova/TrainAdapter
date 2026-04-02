"""
推理脚本：使用训练好的Coordinate Adapter进行推理
"""
import os
import sys
import argparse
import json

import torch
from PIL import Image, ImageDraw, ImageFont

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

    @staticmethod
    def _load_font(size, bold=False):
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

    def preprocess_pil_image(self, image, grid_image=None):
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        image = image.convert('RGB')
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        if grid_image is not None:
            grid_image = grid_image.convert('RGB')
            grid_image_tensor = self.transform(grid_image).unsqueeze(0).to(self.device)
        else:
            grid_image_tensor = image_tensor.clone()

        return image_tensor, grid_image_tensor, original_size

    def _build_instruction(self, query):
        return (
            f"Given the grid coordinate system, locate the referent described as "
            f"'{query}' in the image and predict the most likely target points."
        )

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

    def _run_model(self, image_tensor, grid_image_tensor, query, use_dynamic_topk=False, dynamic_topk_params=None):
        instruction = self._build_instruction(query)
        input_ids, attention_mask = self._encode_text(instruction)
        adapter_dtype = next(self.adapter.parameters()).dtype
        dynamic_topk_params = dynamic_topk_params or {}

        with torch.no_grad():
            visual_features = self.qwen_model.encode_image(image_tensor)
            if visual_features.dtype != adapter_dtype:
                visual_features = visual_features.to(dtype=adapter_dtype)

            enhanced_features = self.adapter(image_tensor, grid_image_tensor, visual_features)

            text_embeddings = self.qwen_model.encode_text(input_ids, attention_mask)
            if text_embeddings.dtype != adapter_dtype:
                text_embeddings = text_embeddings.to(dtype=adapter_dtype)

            if getattr(self.adapter, 'output_mode', 'point_regression') == 'grid_logits':
                pred_grid_logits = self.adapter.predict_grid_logits(
                    enhanced_features,
                    text_features=text_embeddings,
                    attention_mask=attention_mask
                )
                if use_dynamic_topk:
                    selected = self.adapter.decode_grid_logits_dynamic(
                        pred_grid_logits,
                        abs_threshold=dynamic_topk_params.get('abs_threshold', 0.35),
                        rel_ratio=dynamic_topk_params.get('rel_ratio', 0.75),
                        min_k=dynamic_topk_params.get('min_k', 1),
                        max_k=dynamic_topk_params.get('max_k', 6)
                    )
                    pred_points = selected['selected_points'][0].detach().cpu()
                    pred_logits = selected['selected_logits'][0].detach().cpu()
                    selection_mode = 'dynamic_topk'
                    selected_k = int(selected['selected_ks'][0])
                    dynamic_meta = {
                        'abs_threshold': float(dynamic_topk_params.get('abs_threshold', 0.35)),
                        'rel_ratio': float(dynamic_topk_params.get('rel_ratio', 0.75)),
                        'min_k': int(dynamic_topk_params.get('min_k', 1)),
                        'max_k': int(dynamic_topk_params.get('max_k', 6)),
                    }
                else:
                    pred_points, pred_logits = self.adapter.decode_grid_logits(
                        pred_grid_logits,
                        top_k=self.adapter.num_output_points
                    )
                    pred_points = pred_points[0].detach().cpu()
                    pred_logits = pred_logits[0].detach().cpu()
                    selection_mode = 'fixed_topk'
                    selected_k = len(pred_points)
                    dynamic_meta = None
            else:
                pred_points, pred_logits = self.adapter.predict_point_regression(
                    enhanced_features,
                    text_features=text_embeddings,
                    attention_mask=attention_mask
                )
                pred_points = pred_points[0].detach().cpu()
                pred_logits = pred_logits[0].detach().cpu()
                selection_mode = 'fixed_topk'
                selected_k = len(pred_points)
                dynamic_meta = None

        return {
            'instruction': instruction,
            'pred_points': pred_points,
            'pred_logits': pred_logits,
            'selection_mode': selection_mode,
            'selected_k': selected_k,
            'dynamic_topk_params': dynamic_meta,
        }

    def render_prediction_overlay(self, image, absolute_points, scores, query):
        canvas = image.convert('RGB').copy()
        draw = ImageDraw.Draw(canvas)
        title_font = self._load_font(20, bold=True)
        score_font = self._load_font(16)

        for idx, ((x, y), score) in enumerate(zip(absolute_points, scores), start=1):
            radius = 10
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, outline="#dc2626", width=3)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill="#dc2626")
            draw.text((x + 14, y - 12), f"P{idx}", fill="#dc2626", font=title_font)
            draw.text((x + 14, y + 10), f"{float(score):.2f}", fill="#991b1b", font=score_font)

        draw.rectangle([0, 0, canvas.width, 34], fill=(245, 247, 251))
        draw.text((12, 7), f"Query: {query}", fill="#111827", font=score_font)
        return canvas

    def predict_from_pil(
        self,
        image,
        query,
        grid_image=None,
        use_dynamic_topk=False,
        dynamic_topk_params=None,
        include_annotated_image=True
    ):
        image_tensor, grid_image_tensor, original_size = self.preprocess_pil_image(
            image,
            grid_image=grid_image
        )
        outputs = self._run_model(
            image_tensor,
            grid_image_tensor,
            query,
            use_dynamic_topk=use_dynamic_topk,
            dynamic_topk_params=dynamic_topk_params
        )

        normalized_points = outputs['pred_points'].tolist()
        scores = torch.sigmoid(outputs['pred_logits']).tolist()
        absolute_points = self._to_absolute_points(normalized_points, original_size)
        summary = {
            'query': query,
            'instruction': outputs['instruction'],
            'output_mode': getattr(self.adapter, 'output_mode', 'point_regression'),
            'normalized_points': [[round(float(x), 4), round(float(y), 4)] for x, y in normalized_points],
            'absolute_points': absolute_points,
            'scores': [round(float(score), 4) for score in scores],
            'image_size': list(original_size),
            'selection_mode': outputs['selection_mode'],
            'selected_k': int(outputs['selected_k']),
            'dynamic_topk_params': outputs['dynamic_topk_params'],
        }

        if include_annotated_image:
            summary['annotated_image'] = self.render_prediction_overlay(
                image,
                absolute_points,
                summary['scores'],
                query
            )

        return summary

    def predict(
        self,
        image_path,
        query,
        return_text=False,
        grid_image_path=None,
        return_normalized=False,
        use_dynamic_topk=False,
        dynamic_topk_params=None
    ):
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
        image = Image.open(image_path).convert('RGB')
        grid_image = Image.open(grid_image_path).convert('RGB') if grid_image_path and os.path.exists(grid_image_path) else None
        summary = self.predict_from_pil(
            image,
            query,
            grid_image=grid_image,
            use_dynamic_topk=use_dynamic_topk,
            dynamic_topk_params=dynamic_topk_params,
            include_annotated_image=False
        )
        absolute_points = summary['absolute_points']

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
        annotated = self.render_prediction_overlay(
            image,
            points,
            [1.0] * len(points),
            query
        )
        annotated.save(save_path)

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
    parser.add_argument('--dynamic_topk', action='store_true', help='启用动态 top-k 解码')
    parser.add_argument('--dynamic_abs_threshold', type=float, default=0.35, help='动态 top-k 绝对阈值')
    parser.add_argument('--dynamic_rel_ratio', type=float, default=0.75, help='动态 top-k 相对 top1 比例')
    parser.add_argument('--dynamic_min_k', type=int, default=1, help='动态 top-k 最小点数')
    parser.add_argument('--dynamic_max_k', type=int, default=6, help='动态 top-k 最大点数')

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
            return_normalized=True,
            use_dynamic_topk=args.dynamic_topk,
            dynamic_topk_params={
                'abs_threshold': args.dynamic_abs_threshold,
                'rel_ratio': args.dynamic_rel_ratio,
                'min_k': args.dynamic_min_k,
                'max_k': args.dynamic_max_k,
            }
        )
        points = prediction_payload['absolute_points']
        print(json.dumps(prediction_payload, ensure_ascii=False, indent=2))
    elif args.return_text:
        points, text = inferencer.predict(
            args.image_path,
            args.query,
            return_text=True,
            grid_image_path=args.grid_image_path,
            use_dynamic_topk=args.dynamic_topk,
            dynamic_topk_params={
                'abs_threshold': args.dynamic_abs_threshold,
                'rel_ratio': args.dynamic_rel_ratio,
                'min_k': args.dynamic_min_k,
                'max_k': args.dynamic_max_k,
            }
        )
        print(f"Prediction summary: {text}")
    else:
        points = inferencer.predict(
            args.image_path,
            args.query,
            grid_image_path=args.grid_image_path,
            use_dynamic_topk=args.dynamic_topk,
            dynamic_topk_params={
                'abs_threshold': args.dynamic_abs_threshold,
                'rel_ratio': args.dynamic_rel_ratio,
                'min_k': args.dynamic_min_k,
                'max_k': args.dynamic_max_k,
            }
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
