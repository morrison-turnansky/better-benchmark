"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 7802e173c708
Shape hash: 1dbaf289
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
    def forward(self, arg0_1: "bf16[512, 32768]", arg1_1: "bf16[1, 512, 32]", arg2_1: "bf16[1, 512, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        view: "bf16[1, 512, 32768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[1, 512, 64, 512]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[1, 64, 512, 512]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type: "f32[1, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(permute, torch.float32)
        pow_1: "f32[1, 64, 512, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        mean: "f32[1, 64, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 64, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 64, 512, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        convert_element_type_1: "bf16[1, 64, 512, 1]" = torch.ops.prims.convert_element_type.default(rsqrt, torch.bfloat16);  rsqrt = None
        mul: "bf16[1, 64, 512, 512]" = torch.ops.aten.mul.Tensor(permute, convert_element_type_1);  permute = convert_element_type_1 = None
        slice_1: "bf16[1, 64, 512, 448]" = torch.ops.aten.slice.Tensor(mul, 3, 0, -64)
        slice_2: "bf16[1, 64, 512, 64]" = torch.ops.aten.slice.Tensor(mul, 3, -64, 9223372036854775807);  mul = None
        convert_element_type_2: "f32[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(slice_2, torch.float32)
        unsqueeze: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 3);  arg1_1 = None
        expand: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_2);  unsqueeze = _shape_param_2 = None
        clone: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        unsqueeze_1: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_2, 1);  view_2 = None
        mul_1: "f32[1, 64, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_2, unsqueeze_1);  convert_element_type_2 = unsqueeze_1 = None
        slice_3: "bf16[1, 64, 512, 32]" = torch.ops.aten.slice.Tensor(slice_2, 3, 1, 9223372036854775807, 2)
        neg: "bf16[1, 64, 512, 32]" = torch.ops.aten.neg.default(slice_3);  slice_3 = None
        unsqueeze_2: "bf16[1, 64, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        slice_4: "bf16[1, 64, 512, 32]" = torch.ops.aten.slice.Tensor(slice_2, 3, 0, 9223372036854775807, 2);  slice_2 = None
        unsqueeze_3: "bf16[1, 64, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_4, 4);  slice_4 = None
        cat: "bf16[1, 64, 512, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        view_3: "bf16[1, 64, 512, 64]" = torch.ops.aten.view.default(cat, _shape_param_4);  cat = _shape_param_4 = None
        convert_element_type_3: "f32[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        unsqueeze_4: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 3);  arg2_1 = None
        expand_1: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_4, _shape_param_5);  unsqueeze_4 = _shape_param_5 = None
        clone_1: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_4: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        unsqueeze_5: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_4, 1);  view_4 = None
        mul_2: "f32[1, 64, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_3, unsqueeze_5);  convert_element_type_3 = unsqueeze_5 = None
        add_1: "f32[1, 64, 512, 64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_4: "bf16[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        cat_1: "bf16[1, 64, 512, 512]" = torch.ops.aten.cat.default([slice_1, convert_element_type_4], -1);  slice_1 = convert_element_type_4 = None
        expand_2: "bf16[1, 64, 512, 512]" = torch.ops.aten.expand.default(cat_1, _shape_param_7);  cat_1 = _shape_param_7 = None
        view_5: "bf16[64, 512, 512]" = torch.ops.aten.view.default(expand_2, _shape_param_8);  expand_2 = _shape_param_8 = None
        return view_5



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
