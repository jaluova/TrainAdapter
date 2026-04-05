# GridGround Text-Guided Adapter

GridGround is a lightweight text-guided spatial adapter trained on top of a frozen `Qwen2.5-VL-7B-Instruct` backbone for referring-expression localization.

This release contains a continuation checkpoint from the `textguided_localsync_20260405_4epoch` experiment. The model predicts `11 x 11` grid logits and is intended to be used together with the original `TrainAdapter` codebase.

## Included Files

- `best_model.pth`: best checkpoint from this release round
- `config.json`: training and model configuration used by the checkpoint

## Backbone Dependency

This release does **not** bundle the full Qwen base model. You still need:

- `Qwen/Qwen2.5-VL-7B-Instruct`

Update `model.qwen_model_path` in `config.json` or override it at runtime to point to your local Qwen2.5-VL directory.

## Current Snapshot

- Backbone: frozen `Qwen2.5-VL-7B-Instruct`
- Adapter: `lightweight`
- Output mode: `grid_logits`
- Grid size: `11 x 11`
- Training subset: `fast10000`
- Resume mode: `resume_as_init`

Observed validation snapshot during this run:

- `Acc@1Grid`: `55.21%`
- `Acc@Top4`: `77.55%`
- `Relation Acc@Top4`: `82.19%`
- `Color Acc@Top4`: `73.10%`
- `GT Coverage@Top4`: `22.31%`

## Intended Usage

This checkpoint is designed for the `TrainAdapter` / `GridGround` project codebase, not as a standalone Hugging Face-style end-to-end package.

Typical workflow:

1. Prepare the `TrainAdapter` codebase.
2. Download `Qwen2.5-VL-7B-Instruct`.
3. Point `config.json` to the local Qwen path.
4. Load `best_model.pth` with the project inference or visualization scripts.

## Example Inference Command

```bash
python src/inference.py \
  --adapter_path best_model.pth \
  --config config.json \
  --image_path /path/to/image.jpg \
  --query "the red object on the left"
```

## Notes

- This release is primarily for experiment sharing and reproduction.
- The checkpoint may not be directly compatible with newer default configs if architecture defaults change.
- For continued training, prefer loading from `config.json` and using resume/init mode in the project training script.
