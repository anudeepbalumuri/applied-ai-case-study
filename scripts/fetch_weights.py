"""Fetch every pretrained model the pipeline uses into ./weights (no weights live in the repo).

Disclosure (all public, used as-is, no fine-tuning):
  depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf   metric monocular depth (video, photo tiers)
  IDEA-Research/grounding-dino-base                          open-vocabulary damage proposals
  facebook/sam-vit-base                                      damage masks
  openai/clip-vit-large-patch14                              damage verification, room typing
  kornia DISK ("depth") + LightGlue ("disk")                 keypoints / matching (video, photo tiers)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "weights", "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "weights", "torch"))

from huggingface_hub import snapshot_download  # noqa: E402

HF = ["depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "IDEA-Research/grounding-dino-base",
      "facebook/sam-vit-base", "openai/clip-vit-large-patch14"]
for repo in HF:
    print("fetching", repo, flush=True)
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab*"])
print("fetching DISK + LightGlue (kornia)", flush=True)
import kornia.feature as KF  # noqa: E402
KF.DISK.from_pretrained("depth")
KF.LightGlue("disk")
print("all weights in", os.path.join(ROOT, "weights"))
