"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 6c423007effe
Shape hash: 806ad812
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
    def forward(self, arg0_1: "bf16[512, 32768]", arg1_1: "f32[32]", arg2_1: "bf16[512, 512]", arg3_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23):
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
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        unsqueeze_1: "i64[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        convert_element_type_3: "f32[1, 512, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        mul_1: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg1_1);  convert_element_type_3 = arg1_1 = None
        cos: "f32[1, 512, 32]" = torch.ops.aten.cos.default(mul_1)
        convert_element_type_4: "bf16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(cos, torch.bfloat16);  cos = None
        unsqueeze_2: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 3)
        expand: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_2);  unsqueeze_2 = _shape_param_2 = None
        clone: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        unsqueeze_3: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_2, 1);  view_2 = None
        mul_2: "f32[1, 64, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_2, unsqueeze_3);  convert_element_type_2 = unsqueeze_3 = None
        slice_3: "bf16[1, 64, 512, 32]" = torch.ops.aten.slice.Tensor(slice_2, 3, 1, 9223372036854775807, 2)
        neg: "bf16[1, 64, 512, 32]" = torch.ops.aten.neg.default(slice_3);  slice_3 = None
        unsqueeze_4: "bf16[1, 64, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        slice_4: "bf16[1, 64, 512, 32]" = torch.ops.aten.slice.Tensor(slice_2, 3, 0, 9223372036854775807, 2);  slice_2 = None
        unsqueeze_5: "bf16[1, 64, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_4, 4);  slice_4 = None
        cat: "bf16[1, 64, 512, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], -1);  unsqueeze_4 = unsqueeze_5 = None
        view_3: "bf16[1, 64, 512, 64]" = torch.ops.aten.view.default(cat, _shape_param_4);  cat = _shape_param_4 = None
        convert_element_type_5: "f32[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        sin: "f32[1, 512, 32]" = torch.ops.aten.sin.default(mul_1);  mul_1 = None
        convert_element_type_6: "bf16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(sin, torch.bfloat16);  sin = None
        unsqueeze_6: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3)
        expand_1: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_6, _shape_param_5);  unsqueeze_6 = _shape_param_5 = None
        clone_1: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_4: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        unsqueeze_7: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_4, 1);  view_4 = None
        mul_3: "f32[1, 64, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_5, unsqueeze_7);  convert_element_type_5 = unsqueeze_7 = None
        add_1: "f32[1, 64, 512, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        convert_element_type_7: "bf16[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        cat_1: "bf16[1, 64, 512, 512]" = torch.ops.aten.cat.default([slice_1, convert_element_type_7], -1);  slice_1 = convert_element_type_7 = None
        expand_2: "bf16[1, 64, 512, 512]" = torch.ops.aten.expand.default(cat_1, _shape_param_7);  cat_1 = _shape_param_7 = None
        view_5: "bf16[64, 512, 512]" = torch.ops.aten.view.default(expand_2, _shape_param_8);  expand_2 = _shape_param_8 = None
        view_6: "bf16[1, 512, 512]" = torch.ops.aten.view.default(arg2_1, _shape_param_9);  arg2_1 = _shape_param_9 = None
        convert_element_type_8: "f32[1, 512, 512]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        pow_2: "f32[1, 512, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 2)
        mean_1: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4: "f32[1, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, rsqrt_1);  convert_element_type_8 = rsqrt_1 = None
        mul_5: "f32[1, 512, 512]" = torch.ops.aten.mul.Tensor(arg3_1, mul_4);  arg3_1 = mul_4 = None
        convert_element_type_9: "bf16[1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_7: "bf16[1, 512, 1, 512]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_10);  convert_element_type_9 = _shape_param_10 = None
        permute_1: "bf16[1, 1, 512, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        slice_5: "bf16[1, 1, 512, 448]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, -64)
        slice_6: "bf16[1, 1, 512, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, -64, 9223372036854775807);  permute_1 = None
        convert_element_type_10: "f32[1, 1, 512, 64]" = torch.ops.prims.convert_element_type.default(slice_6, torch.float32)
        unsqueeze_8: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 3)
        expand_3: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_8, _shape_param_11);  unsqueeze_8 = _shape_param_11 = None
        clone_2: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_8: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_12);  clone_2 = _shape_param_12 = None
        unsqueeze_9: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_8, 1);  view_8 = None
        mul_6: "f32[1, 1, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_10, unsqueeze_9);  convert_element_type_10 = unsqueeze_9 = None
        slice_7: "bf16[1, 1, 512, 32]" = torch.ops.aten.slice.Tensor(slice_6, 3, 1, 9223372036854775807, 2)
        neg_1: "bf16[1, 1, 512, 32]" = torch.ops.aten.neg.default(slice_7);  slice_7 = None
        unsqueeze_10: "bf16[1, 1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
        slice_8: "bf16[1, 1, 512, 32]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 9223372036854775807, 2);  slice_6 = None
        unsqueeze_11: "bf16[1, 1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_8, 4);  slice_8 = None
        cat_2: "bf16[1, 1, 512, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_10, unsqueeze_11], -1);  unsqueeze_10 = unsqueeze_11 = None
        view_9: "bf16[1, 1, 512, 64]" = torch.ops.aten.view.default(cat_2, _shape_param_13);  cat_2 = _shape_param_13 = None
        convert_element_type_11: "f32[1, 1, 512, 64]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        unsqueeze_12: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3)
        expand_4: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_12, _shape_param_14);  unsqueeze_12 = _shape_param_14 = None
        clone_3: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_10: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_3, _shape_param_15);  clone_3 = _shape_param_15 = None
        unsqueeze_13: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_10, 1);  view_10 = None
        mul_7: "f32[1, 1, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_11, unsqueeze_13);  convert_element_type_11 = unsqueeze_13 = None
        add_3: "f32[1, 1, 512, 64]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        convert_element_type_12: "bf16[1, 1, 512, 64]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        cat_3: "bf16[1, 1, 512, 512]" = torch.ops.aten.cat.default([slice_5, convert_element_type_12], -1);  slice_5 = convert_element_type_12 = None
        unsqueeze_14: "bf16[1, 1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(cat_3, 2)
        expand_5: "bf16[1, 1, 64, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_14, _shape_param_16);  unsqueeze_14 = _shape_param_16 = None
        view_11: "bf16[1, 64, 512, 512]" = torch.ops.aten.view.default(expand_5, _shape_param_17);  expand_5 = _shape_param_17 = None
        permute_2: "bf16[1, 64, 512, 512]" = torch.ops.aten.permute.default(view_11, [0, 1, 3, 2]);  view_11 = None
        expand_6: "bf16[1, 64, 512, 512]" = torch.ops.aten.expand.default(permute_2, _shape_param_18);  permute_2 = _shape_param_18 = None
        view_12: "bf16[64, 512, 512]" = torch.ops.aten.view.default(expand_6, _shape_param_19);  expand_6 = _shape_param_19 = None
        slice_9: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_9, 1);  slice_9 = None
        cat_4: "i64[1, 513]" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = None
        slice_10: "i64[1, 512]" = torch.ops.aten.slice.Tensor(cat_4, -1, 1, 513)
        slice_11: "i64[1, 512]" = torch.ops.aten.slice.Tensor(cat_4, -1, 0, 512);  cat_4 = None
        sub_1: "i64[1, 512]" = torch.ops.aten.sub.Tensor(slice_10, slice_11);  slice_10 = slice_11 = None
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        unsqueeze_15: "bf16[1, 1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(cat_3, 2);  cat_3 = None
        expand_7: "bf16[1, 1, 64, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_15, _shape_param_20);  unsqueeze_15 = _shape_param_20 = None
        view_13: "bf16[1, 64, 512, 512]" = torch.ops.aten.view.default(expand_7, _shape_param_21);  expand_7 = _shape_param_21 = None
        expand_8: "bf16[1, 64, 512, 512]" = torch.ops.aten.expand.default(view_13, _shape_param_22);  view_13 = _shape_param_22 = None
        view_14: "bf16[64, 512, 512]" = torch.ops.aten.view.default(expand_8, _shape_param_23);  expand_8 = _shape_param_23 = None
        return (unsqueeze, convert_element_type_4, convert_element_type_6, view_5, view_12, ne, view_14)



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
