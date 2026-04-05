from pathlib import Path
import os

from modelscope.hub.api import HubApi


def main():
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "alpharho/GridGround-TextGuided")
    token_path = Path.home() / ".modelscope" / "credentials" / "session"
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token and token_path.exists():
        token = token_path.read_text().strip()
    if not token:
        raise RuntimeError("No ModelScope token found in MODELSCOPE_API_TOKEN or ~/.modelscope/credentials/session")

    model_dir = Path(__file__).resolve().parent
    api = HubApi()
    api.login(token)
    api.push_model(model_id=model_id, model_dir=str(model_dir))
    print(f"Uploaded to {model_id}")


if __name__ == "__main__":
    main()
