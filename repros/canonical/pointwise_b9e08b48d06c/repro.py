"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: b9e08b48d06c
Shape hash: 9037f916
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8, 512, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "bf16[512, 8, 1024]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2]);  arg0_1 = None
        view: "bf16[1, 512, 8, 1024]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        clone: "bf16[1, 512, 8, 1024]" = torch.ops.aten.clone.default(view, memory_format = torch.contiguous_format);  view = None
        view_1: "bf16[1, 512, 8192]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[512, 8192]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        return view_2



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
