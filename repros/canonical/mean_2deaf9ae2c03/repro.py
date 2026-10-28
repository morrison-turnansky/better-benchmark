"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 2deaf9ae2c03
Shape hash: eedff0d6
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
    def forward(self, arg0_1: "f32[1, 512, 4]", arg1_1: "f32[]", arg2_1: "f32[4]", arg3_1: "bf16[512, 4096]", arg4_1: "bf16[512, 4, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mul: "f32[1, 512, 4]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        add: "f32[1, 512, 4]" = torch.ops.aten.add.Tensor(mul, arg2_1);  mul = arg2_1 = None
        sigmoid: "f32[1, 512, 4]" = torch.ops.aten.sigmoid.default(add);  add = None
        mul_1: "f32[1, 512, 4]" = torch.ops.aten.mul.Tensor(sigmoid, 2);  sigmoid = None
        convert_element_type: "bf16[1, 512, 4]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        unsqueeze: "bf16[1, 512, 4, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        view: "bf16[1, 512, 4096]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        unsqueeze_1: "bf16[1, 512, 1, 4096]" = torch.ops.aten.unsqueeze.default(view, -2);  view = None
        mul_2: "bf16[1, 512, 4, 4096]" = torch.ops.aten.mul.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        view_1: "bf16[1, 512, 4, 4096]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        add_1: "bf16[1, 512, 4, 4096]" = torch.ops.aten.add.Tensor(mul_2, view_1);  mul_2 = view_1 = None
        view_2: "bf16[1, 512, 16384]" = torch.ops.aten.view.default(add_1, _shape_param_2);  _shape_param_2 = None
        convert_element_type_1: "f32[1, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        pow_1: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2)
        mean: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_3: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_1, rsqrt);  convert_element_type_1 = rsqrt = None
        view_3: "f32[512, 16384]" = torch.ops.aten.view.default(mul_3, _shape_param_3);  mul_3 = _shape_param_3 = None
        expand: "bf16[1, 512, 4, 4096]" = torch.ops.aten.expand.default(add_1, _shape_param_4);  _shape_param_4 = None
        view_4: "bf16[512, 4, 4096]" = torch.ops.aten.view.default(expand, _shape_param_5);  expand = _shape_param_5 = None
        return (add_1, view_3, view_4)



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
