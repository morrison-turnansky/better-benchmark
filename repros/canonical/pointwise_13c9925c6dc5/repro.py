"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 13c9925c6dc5
Shape hash: 7d26a7fe
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
    def forward(self, arg0_1: "bf16[64, 512, 512]", arg1_1: "bf16[1, 512, 32]", arg2_1: "bf16[1, 512, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view: "bf16[1, 64, 512, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[1, 512, 64, 512]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone: "bf16[1, 512, 64, 512]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        permute_1: "bf16[1, 64, 512, 512]" = torch.ops.aten.permute.default(clone, [0, 2, 1, 3]);  clone = None
        slice_1: "bf16[1, 64, 512, 448]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, -64)
        slice_2: "bf16[1, 64, 512, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, -64, 9223372036854775807);  permute_1 = None
        convert_element_type: "f32[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(slice_2, torch.float32)
        unsqueeze: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 3);  arg1_1 = None
        expand: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_1);  unsqueeze = _shape_param_1 = None
        clone_1: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_1: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_2);  clone_1 = _shape_param_2 = None
        unsqueeze_1: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_1, 1);  view_1 = None
        mul: "f32[1, 64, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type, unsqueeze_1);  convert_element_type = unsqueeze_1 = None
        slice_3: "bf16[1, 64, 512, 32]" = torch.ops.aten.slice.Tensor(slice_2, 3, 1, 9223372036854775807, 2)
        neg: "bf16[1, 64, 512, 32]" = torch.ops.aten.neg.default(slice_3);  slice_3 = None
        unsqueeze_2: "bf16[1, 64, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        slice_4: "bf16[1, 64, 512, 32]" = torch.ops.aten.slice.Tensor(slice_2, 3, 0, 9223372036854775807, 2);  slice_2 = None
        unsqueeze_3: "bf16[1, 64, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_4, 4);  slice_4 = None
        cat: "bf16[1, 64, 512, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        view_2: "bf16[1, 64, 512, 64]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        convert_element_type_1: "f32[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        neg_1: "bf16[1, 512, 32]" = torch.ops.aten.neg.default(arg2_1);  arg2_1 = None
        unsqueeze_4: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 3);  neg_1 = None
        expand_1: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_4, _shape_param_4);  unsqueeze_4 = _shape_param_4 = None
        clone_2: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_3: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_5);  clone_2 = _shape_param_5 = None
        unsqueeze_5: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        mul_1: "f32[1, 64, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_1, unsqueeze_5);  convert_element_type_1 = unsqueeze_5 = None
        add: "f32[1, 64, 512, 64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        convert_element_type_2: "bf16[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        cat_1: "bf16[1, 64, 512, 512]" = torch.ops.aten.cat.default([slice_1, convert_element_type_2], -1);  slice_1 = convert_element_type_2 = None
        permute_2: "bf16[1, 512, 64, 512]" = torch.ops.aten.permute.default(cat_1, [0, 2, 1, 3]);  cat_1 = None
        clone_3: "bf16[1, 512, 64, 512]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_4: "bf16[1, 512, 8, 4096]" = torch.ops.aten.view.default(clone_3, _shape_param_6);  clone_3 = _shape_param_6 = None
        view_5: "bf16[512, 8, 4096]" = torch.ops.aten.view.default(view_4, _shape_param_7);  view_4 = _shape_param_7 = None
        permute_3: "bf16[8, 512, 4096]" = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        return permute_3



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
