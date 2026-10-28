"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 44ed9027e3e9
Shape hash: 603230fc
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
    def forward(self, arg0_1: "f32[512, 4]", arg1_1: "f32[1]", arg2_1: "f32[4]", arg3_1: "bf16[1, 512, 4, 4096]", arg4_1: "f32[4096]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "f32[1, 512, 4]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "f32[1, 512, 4]" = torch.ops.aten.mul.Tensor(view, arg1_1);  view = arg1_1 = None
        add: "f32[1, 512, 4]" = torch.ops.aten.add.Tensor(mul, arg2_1);  mul = arg2_1 = None
        sigmoid: "f32[1, 512, 4]" = torch.ops.aten.sigmoid.default(add);  add = None
        add_1: "f32[1, 512, 4]" = torch.ops.aten.add.Tensor(sigmoid, 1e-06);  sigmoid = None
        unsqueeze: "f32[1, 512, 4, 1]" = torch.ops.aten.unsqueeze.default(add_1, -1);  add_1 = None
        mul_1: "f32[1, 512, 4, 4096]" = torch.ops.aten.mul.Tensor(unsqueeze, arg3_1);  unsqueeze = arg3_1 = None
        sum_1: "f32[1, 512, 4096]" = torch.ops.aten.sum.dim_IntList(mul_1, [2]);  mul_1 = None
        convert_element_type: "bf16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        convert_element_type_1: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        pow_1: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2)
        mean: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_1, rsqrt);  convert_element_type_1 = rsqrt = None
        mul_3: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg4_1, mul_2);  arg4_1 = mul_2 = None
        convert_element_type_2: "bf16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view_1: "bf16[512, 4096]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        return view_1



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
