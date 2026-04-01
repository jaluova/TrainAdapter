"""
轻量推理服务：在训练机常驻加载模型，对外提供本机 HTTP 推理接口。
"""
import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from inference import CoordinateAdapterInference
from training.config import Config, get_config


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_from_base64(encoded):
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def load_runtime_config(args):
    config = Config.load(args.config) if args.config and os.path.exists(args.config) else get_config('default')
    if args.qwen_model_path:
        config.model.qwen_model_path = args.qwen_model_path
    if args.adapter_type:
        config.model.adapter_type = args.adapter_type
    if args.num_output_points is not None:
        config.model.num_output_points = args.num_output_points
    if args.grid_size is not None:
        config.model.grid_size = args.grid_size
    return config


def build_handler(inferencer):
    class InferenceHandler(BaseHTTPRequestHandler):
        server_version = "GridGroundInference/0.1"

        def _write_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._write_json({
                    "ok": True,
                    "device": str(inferencer.device),
                    "output_mode": getattr(inferencer.adapter, "output_mode", "point_regression"),
                    "adapter_type": inferencer.config.model.adapter_type,
                })
                return
            self._write_json({"ok": False, "error": "Not found"}, status=404)

        def do_POST(self):
            if self.path.rstrip("/") != "/predict":
                self._write_json({"ok": False, "error": "Not found"}, status=404)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                request_body = self.rfile.read(content_length)
                payload = json.loads(request_body.decode("utf-8"))

                image = image_from_base64(payload["image_base64"])
                query = payload["query"]
                dynamic_params = {
                    "abs_threshold": payload.get("abs_threshold", 0.35),
                    "rel_ratio": payload.get("rel_ratio", 0.75),
                    "min_k": payload.get("min_k", 1),
                    "max_k": payload.get("max_k", 6),
                }
                result = inferencer.predict_from_pil(
                    image,
                    query,
                    use_dynamic_topk=bool(payload.get("use_dynamic_topk", False)),
                    dynamic_topk_params=dynamic_params,
                    include_annotated_image=True
                )

                annotated_image = result.pop("annotated_image")
                result["annotated_image_base64"] = image_to_base64(annotated_image)
                self._write_json({"ok": True, "result": result})
            except Exception as exc:
                self._write_json({"ok": False, "error": str(exc)}, status=500)

        def log_message(self, fmt, *args):
            return

    return InferenceHandler


def parse_args():
    parser = argparse.ArgumentParser(description="TrainAdapter local inference service")
    parser.add_argument("--adapter_path", type=str, required=True, help="Adapter checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Training config.json path")
    parser.add_argument("--qwen_model_path", type=str, default=None, help="Override Qwen model path")
    parser.add_argument("--adapter_type", type=str, default=None, choices=["standard", "lightweight"], help="Override adapter type")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--num_output_points", type=int, default=None, help="Override number of output points")
    parser.add_argument("--grid_size", type=int, default=None, help="Override grid size")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_runtime_config(args)
    inferencer = CoordinateAdapterInference(
        adapter_path=args.adapter_path,
        qwen_model_path=config.model.qwen_model_path,
        adapter_type=config.model.adapter_type,
        device=args.device,
        config=config,
        num_output_points=args.num_output_points,
        grid_size=args.grid_size
    )

    server = ThreadingHTTPServer((args.host, args.port), build_handler(inferencer))
    print(f"Inference service listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
