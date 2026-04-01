"""
公网机轻量网页 Demo：上传图片和 query，代理到训练机推理服务。
"""
import argparse
import base64
import io

import gradio as gr
import requests
from PIL import Image


def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(encoded):
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


class RemoteInferenceClient:
    def __init__(self, inference_url, timeout=180):
        self.inference_url = inference_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def predict(self, image, query, use_dynamic_topk, abs_threshold, rel_ratio, min_k, max_k):
        payload = {
            "image_base64": pil_to_base64(image),
            "query": query,
            "use_dynamic_topk": bool(use_dynamic_topk),
            "abs_threshold": float(abs_threshold),
            "rel_ratio": float(rel_ratio),
            "min_k": int(min_k),
            "max_k": int(max_k),
        }
        response = self.session.post(
            f"{self.inference_url}/predict",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(body.get("error", "Unknown inference error"))
        return body["result"]


def build_summary(result):
    scores = result.get("scores", [])
    top_score = max(scores) if scores else 0.0
    return (
        f"Selection mode: `{result.get('selection_mode', 'fixed_topk')}`\n\n"
        f"Selected points: `{result.get('selected_k', len(result.get('absolute_points', [])))}`\n\n"
        f"Top score: `{top_score:.4f}`\n\n"
        f"Image size: `{result.get('image_size')}`"
    )


def build_point_rows(result):
    rows = []
    for idx, (norm_pt, abs_pt, score) in enumerate(
        zip(
            result.get("normalized_points", []),
            result.get("absolute_points", []),
            result.get("scores", []),
        ),
        start=1
    ):
        rows.append([
            idx,
            round(float(norm_pt[0]), 4),
            round(float(norm_pt[1]), 4),
            round(float(abs_pt[0]), 2),
            round(float(abs_pt[1]), 2),
            round(float(score), 4),
        ])
    return rows


def launch_app(args):
    client = RemoteInferenceClient(args.inference_url, timeout=args.timeout)

    def run_demo(image, query, use_dynamic_topk, abs_threshold, rel_ratio, min_k, max_k):
        if image is None:
            raise gr.Error("Please upload an image first.")
        query = (query or "").strip()
        if not query:
            raise gr.Error("Please enter a query.")

        result = client.predict(
            image,
            query,
            use_dynamic_topk,
            abs_threshold,
            rel_ratio,
            min_k,
            max_k
        )
        annotated = base64_to_pil(result["annotated_image_base64"])
        return annotated, build_point_rows(result), result, build_summary(result)

    with gr.Blocks(title=args.title) as demo:
        gr.Markdown(f"# {args.title}")
        gr.Markdown("Upload one image, type a referring-expression query, and inspect the predicted points plus raw JSON.")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Image")
            with gr.Column():
                query_input = gr.Textbox(label="Query", placeholder="e.g. the man on the left")
                use_dynamic_topk = gr.Checkbox(label="Use dynamic top-k", value=False)
                with gr.Accordion("Dynamic top-k settings", open=False):
                    abs_threshold = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Absolute threshold")
                    rel_ratio = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Relative ratio")
                    min_k = gr.Slider(1, 10, value=1, step=1, label="Min k")
                    max_k = gr.Slider(1, 10, value=6, step=1, label="Max k")
                run_button = gr.Button("Analyze", variant="primary")

        with gr.Row():
            annotated_output = gr.Image(type="pil", label="Annotated result")
            with gr.Column():
                summary_output = gr.Markdown(label="Summary")
                table_output = gr.Dataframe(
                    headers=["Rank", "Norm X", "Norm Y", "Abs X", "Abs Y", "Score"],
                    datatype=["number", "number", "number", "number", "number", "number"],
                    label="Predicted points",
                    interactive=False
                )
        json_output = gr.JSON(label="Raw JSON")

        run_button.click(
            fn=run_demo,
            inputs=[image_input, query_input, use_dynamic_topk, abs_threshold, rel_ratio, min_k, max_k],
            outputs=[annotated_output, table_output, json_output, summary_output]
        )

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        quiet=False
    )


def parse_args():
    parser = argparse.ArgumentParser(description="GridGround upload-and-query web demo")
    parser.add_argument("--inference_url", type=str, default="http://127.0.0.1:8765", help="Remote inference service base URL")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gradio bind host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio bind port")
    parser.add_argument("--timeout", type=int, default=180, help="Inference request timeout in seconds")
    parser.add_argument("--title", type=str, default="GridGround Demo", help="Page title")
    return parser.parse_args()


if __name__ == "__main__":
    launch_app(parse_args())
