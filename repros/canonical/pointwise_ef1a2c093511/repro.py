"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: ef1a2c093511
Shape hash: e1913a9f
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
    def forward(self, arg0_1: "bf16[512, 2048]", arg1_1: "bf16[512, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[1, 512, 2048]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        clamp_max: "f32[1, 512, 2048]" = torch.ops.aten.clamp_max.default(convert_element_type, 10.0);  convert_element_type = None
        convert_element_type_1: "bf16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.bfloat16);  clamp_max = None
        convert_element_type_2: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        neg: "f32[1, 512, 2048]" = torch.ops.aten.neg.default(convert_element_type_2)
        exp: "f32[1, 512, 2048]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 512, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_2, add);  convert_element_type_2 = add = None
        convert_element_type_3: "bf16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_1: "bf16[1, 512, 2048]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        convert_element_type_4: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        clamp_min: "f32[1, 512, 2048]" = torch.ops.aten.clamp_min.default(convert_element_type_4, -10.0);  convert_element_type_4 = None
        clamp_max_1: "f32[1, 512, 2048]" = torch.ops.aten.clamp_max.default(clamp_min, 10.0);  clamp_min = None
        convert_element_type_5: "bf16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.bfloat16);  clamp_max_1 = None
        mul: "bf16[1, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, convert_element_type_5);  convert_element_type_3 = convert_element_type_5 = None
        view_2: "bf16[512, 2048]" = torch.ops.aten.view.default(mul, _shape_param_2);  mul = _shape_param_2 = None
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
