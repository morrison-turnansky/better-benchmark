"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 3f1f2ad3b0eb
Shape hash: dd66a9e5
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
    def forward(self, arg0_1: "bf16[3072, 4096]"):
        # No stacktrace found for following nodes
        split = torch.ops.aten.split.Tensor(arg0_1, 2048, -1);  arg0_1 = None
        getitem: "bf16[3072, 2048]" = split[0]
        getitem_1: "bf16[3072, 2048]" = split[1];  split = None
        convert_element_type: "f32[3072, 2048]" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        clamp_max: "f32[3072, 2048]" = torch.ops.aten.clamp_max.default(convert_element_type, 10.0);  convert_element_type = None
        convert_element_type_1: "bf16[3072, 2048]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.bfloat16);  clamp_max = None
        convert_element_type_2: "f32[3072, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        neg: "f32[3072, 2048]" = torch.ops.aten.neg.default(convert_element_type_2)
        exp: "f32[3072, 2048]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[3072, 2048]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[3072, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_2, add);  convert_element_type_2 = add = None
        convert_element_type_3: "bf16[3072, 2048]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convert_element_type_4: "f32[3072, 2048]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        clamp_min: "f32[3072, 2048]" = torch.ops.aten.clamp_min.default(convert_element_type_4, -10.0);  convert_element_type_4 = None
        clamp_max_1: "f32[3072, 2048]" = torch.ops.aten.clamp_max.default(clamp_min, 10.0);  clamp_min = None
        convert_element_type_5: "bf16[3072, 2048]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.bfloat16);  clamp_max_1 = None
        mul: "bf16[3072, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, convert_element_type_5);  convert_element_type_3 = convert_element_type_5 = None
        return mul



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
