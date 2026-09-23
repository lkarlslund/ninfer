"""Flash-Next Vision checkpoint names and exact reshape."""
import re
from .inventory import VISION_TENSOR_SPECS

def source_name(name):
    if name == "vision/patch_embedding": return "model.visual.patch_embed.proj.weight"
    if name == "vision/patch_embedding_bias": return "model.visual.patch_embed.proj.bias"
    if name == "vision/position_embedding": return "model.visual.pos_embed.weight"
    suffixes = {"attention/qkv": "attn.qkv.weight", "attention/qkv_bias": "attn.qkv.bias",
                "attention/output": "attn.proj.weight", "attention/output_bias": "attn.proj.bias",
                "mlp/fc1": "mlp.linear_fc1.weight", "mlp/fc1_bias": "mlp.linear_fc1.bias",
                "mlp/fc2": "mlp.linear_fc2.weight", "mlp/fc2_bias": "mlp.linear_fc2.bias"}
    match = re.fullmatch(r"vision/layers/(\d+)/(.*)", name)
    if match:
        layer, suffix = match.groups()
        return f"model.visual.blocks.{layer}." + suffixes.get(suffix, suffix.replace("/", "."))
    suffix = name.removeprefix("vision/merger/")
    names = {"fc1":"linear_fc1.weight", "fc1_bias":"linear_fc1.bias", "fc2":"linear_fc2.weight",
             "fc2_bias":"linear_fc2.bias", "norm/weight":"norm.weight", "norm/bias":"norm.bias"}
    return "model.visual.merger." + names[suffix]

SOURCES = {s.id: source_name(s.id) for s in VISION_TENSOR_SPECS}
def signatures():
    return {SOURCES[s.id]: ((1152,3,2,16,16) if s.id == "vision/patch_embedding" else s.shape, "BF16")
            for s in VISION_TENSOR_SPECS}
