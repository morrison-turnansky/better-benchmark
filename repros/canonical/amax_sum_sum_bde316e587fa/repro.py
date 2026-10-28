"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: bde316e587fa
Shape hash: 820111cc
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
    def forward(self, arg0_1: "bf16[512, 256]", arg1_1: "bf16[512, 256]", arg2_1: "f32[4, 256]", arg3_1: "f32[128]", arg4_1: "f32[32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        full: "bf16[1, 128, 8, 128]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        slice_1: "bf16[1, 128, 4, 128]" = torch.ops.aten.slice.Tensor(full, 2, 4, 9223372036854775807)
        view: "bf16[1, 512, 256]" = torch.ops.aten.view.default(arg0_1, _shape_param_1);  arg0_1 = _shape_param_1 = None
        view_1: "bf16[1, 128, 4, 256]" = torch.ops.aten.view.default(view, _shape_param_2);  view = _shape_param_2 = None
        slice_2: "bf16[1, 128, 4, 128]" = torch.ops.aten.slice.Tensor(view_1, 3, 128, 9223372036854775807)
        copy: "bf16[1, 128, 4, 128]" = torch.ops.aten.copy.default(slice_1, slice_2);  slice_1 = slice_2 = None
        slice_scatter: "bf16[1, 128, 8, 128]" = torch.ops.aten.slice_scatter.default(full, copy, 2, 4, 9223372036854775807);  full = copy = None
        slice_3: "bf16[1, 127, 8, 128]" = torch.ops.aten.slice.Tensor(slice_scatter, 1, 1, 9223372036854775807)
        slice_4: "bf16[1, 127, 8, 128]" = torch.ops.aten.slice.Tensor(slice_scatter, 1, 1, 9223372036854775807)
        slice_5: "bf16[1, 127, 4, 128]" = torch.ops.aten.slice.Tensor(slice_4, 2, 0, 4);  slice_4 = None
        slice_6: "bf16[1, 127, 4, 256]" = torch.ops.aten.slice.Tensor(view_1, 1, 0, -1);  view_1 = None
        slice_7: "bf16[1, 127, 4, 128]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 128);  slice_6 = None
        copy_1: "bf16[1, 127, 4, 128]" = torch.ops.aten.copy.default(slice_5, slice_7);  slice_5 = slice_7 = None
        slice_scatter_1: "bf16[1, 127, 8, 128]" = torch.ops.aten.slice_scatter.default(slice_3, copy_1, 2, 0, 4);  slice_3 = copy_1 = None
        slice_scatter_2: "bf16[1, 128, 8, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter, slice_scatter_1, 1, 1, 9223372036854775807);  slice_scatter = slice_scatter_1 = None
        full_1: "f32[1, 128, 8, 128]" = torch.ops.aten.full.default(_shape_param_3, -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        slice_8: "f32[1, 128, 4, 128]" = torch.ops.aten.slice.Tensor(full_1, 2, 4, 9223372036854775807)
        view_2: "bf16[1, 512, 256]" = torch.ops.aten.view.default(arg1_1, _shape_param_4);  arg1_1 = _shape_param_4 = None
        view_3: "bf16[1, 128, 4, 256]" = torch.ops.aten.view.default(view_2, _shape_param_5);  view_2 = _shape_param_5 = None
        add: "f32[1, 128, 4, 256]" = torch.ops.aten.add.Tensor(view_3, arg2_1);  view_3 = arg2_1 = None
        slice_9: "f32[1, 128, 4, 128]" = torch.ops.aten.slice.Tensor(add, 3, 128, 9223372036854775807)
        copy_2: "f32[1, 128, 4, 128]" = torch.ops.aten.copy.default(slice_8, slice_9);  slice_8 = slice_9 = None
        slice_scatter_3: "f32[1, 128, 8, 128]" = torch.ops.aten.slice_scatter.default(full_1, copy_2, 2, 4, 9223372036854775807);  full_1 = copy_2 = None
        slice_10: "f32[1, 127, 8, 128]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 1, 1, 9223372036854775807)
        slice_11: "f32[1, 127, 8, 128]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 1, 1, 9223372036854775807)
        slice_12: "f32[1, 127, 4, 128]" = torch.ops.aten.slice.Tensor(slice_11, 2, 0, 4);  slice_11 = None
        slice_13: "f32[1, 127, 4, 256]" = torch.ops.aten.slice.Tensor(add, 1, 0, -1);  add = None
        slice_14: "f32[1, 127, 4, 128]" = torch.ops.aten.slice.Tensor(slice_13, 3, 0, 128);  slice_13 = None
        copy_3: "f32[1, 127, 4, 128]" = torch.ops.aten.copy.default(slice_12, slice_14);  slice_12 = slice_14 = None
        slice_scatter_4: "f32[1, 127, 8, 128]" = torch.ops.aten.slice_scatter.default(slice_10, copy_3, 2, 0, 4);  slice_10 = copy_3 = None
        slice_scatter_5: "f32[1, 128, 8, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_3, slice_scatter_4, 1, 1, 9223372036854775807);  slice_scatter_3 = slice_scatter_4 = None
        amax: "f32[1, 128, 1, 128]" = torch.ops.aten.amax.default(slice_scatter_5, [2], True)
        sub: "f32[1, 128, 8, 128]" = torch.ops.aten.sub.Tensor(slice_scatter_5, amax);  slice_scatter_5 = amax = None
        exp: "f32[1, 128, 8, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1, 128, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp, [2], True)
        div: "f32[1, 128, 8, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type: "bf16[1, 128, 8, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul: "bf16[1, 128, 8, 128]" = torch.ops.aten.mul.Tensor(slice_scatter_2, convert_element_type);  slice_scatter_2 = convert_element_type = None
        sum_2: "bf16[1, 128, 128]" = torch.ops.aten.sum.dim_IntList(mul, [2]);  mul = None
        convert_element_type_1: "f32[1, 128, 128]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        pow_1: "f32[1, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2)
        mean: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_1: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, rsqrt);  convert_element_type_1 = rsqrt = None
        mul_2: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(arg3_1, mul_1);  arg3_1 = mul_1 = None
        convert_element_type_2: "bf16[1, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        unsqueeze: "bf16[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        slice_15: "bf16[1, 1, 128, 64]" = torch.ops.aten.slice.Tensor(unsqueeze, 3, 0, -64)
        slice_16: "bf16[1, 1, 128, 64]" = torch.ops.aten.slice.Tensor(unsqueeze, 3, -64, 9223372036854775807);  unsqueeze = None
        convert_element_type_3: "f32[1, 1, 128, 64]" = torch.ops.prims.convert_element_type.default(slice_16, torch.float32)
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_3: "i64[128]" = torch.ops.aten.mul.Tensor(iota, 4);  iota = None
        unsqueeze_1: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(mul_3, 0);  mul_3 = None
        expand: "i64[1, 128]" = torch.ops.aten.expand.default(unsqueeze_1, [1, -1]);  unsqueeze_1 = None
        unsqueeze_2: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(expand, 2);  expand = None
        convert_element_type_4: "f32[1, 128, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_2, torch.float32);  unsqueeze_2 = None
        mul_4: "f32[1, 128, 32]" = torch.ops.aten.mul.Tensor(convert_element_type_4, arg4_1);  convert_element_type_4 = arg4_1 = None
        cos: "f32[1, 128, 32]" = torch.ops.aten.cos.default(mul_4)
        convert_element_type_5: "bf16[1, 128, 32]" = torch.ops.prims.convert_element_type.default(cos, torch.bfloat16);  cos = None
        unsqueeze_3: "bf16[1, 128, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 3);  convert_element_type_5 = None
        expand_1: "bf16[1, 128, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_6);  unsqueeze_3 = _shape_param_6 = None
        clone: "bf16[1, 128, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_4: "bf16[1, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_7);  clone = _shape_param_7 = None
        unsqueeze_4: "bf16[1, 1, 128, 64]" = torch.ops.aten.unsqueeze.default(view_4, 1);  view_4 = None
        mul_5: "f32[1, 1, 128, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_3, unsqueeze_4);  convert_element_type_3 = unsqueeze_4 = None
        slice_17: "bf16[1, 1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_16, 3, 1, 9223372036854775807, 2)
        neg: "bf16[1, 1, 128, 32]" = torch.ops.aten.neg.default(slice_17);  slice_17 = None
        unsqueeze_5: "bf16[1, 1, 128, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        slice_18: "bf16[1, 1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 9223372036854775807, 2);  slice_16 = None
        unsqueeze_6: "bf16[1, 1, 128, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_18, 4);  slice_18 = None
        cat: "bf16[1, 1, 128, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_5, unsqueeze_6], -1);  unsqueeze_5 = unsqueeze_6 = None
        view_5: "bf16[1, 1, 128, 64]" = torch.ops.aten.view.default(cat, _shape_param_8);  cat = _shape_param_8 = None
        convert_element_type_6: "f32[1, 1, 128, 64]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        sin: "f32[1, 128, 32]" = torch.ops.aten.sin.default(mul_4);  mul_4 = None
        convert_element_type_7: "bf16[1, 128, 32]" = torch.ops.prims.convert_element_type.default(sin, torch.bfloat16);  sin = None
        unsqueeze_7: "bf16[1, 128, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_7, 3);  convert_element_type_7 = None
        expand_2: "bf16[1, 128, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_7, _shape_param_9);  unsqueeze_7 = _shape_param_9 = None
        clone_1: "bf16[1, 128, 32, 2]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_6: "bf16[1, 128, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_10);  clone_1 = _shape_param_10 = None
        unsqueeze_8: "bf16[1, 1, 128, 64]" = torch.ops.aten.unsqueeze.default(view_6, 1);  view_6 = None
        mul_6: "f32[1, 1, 128, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_6, unsqueeze_8);  convert_element_type_6 = unsqueeze_8 = None
        add_2: "f32[1, 1, 128, 64]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        convert_element_type_8: "bf16[1, 1, 128, 64]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        cat_1: "bf16[1, 1, 128, 128]" = torch.ops.aten.cat.default([slice_15, convert_element_type_8], -1);  slice_15 = convert_element_type_8 = None
        squeeze: "bf16[1, 128, 128]" = torch.ops.aten.squeeze.dim(cat_1, 1);  cat_1 = None
        permute: "bf16[1, 128, 128]" = torch.ops.aten.permute.default(squeeze, [0, 2, 1]);  squeeze = None
        convert_element_type_9: "f32[1, 128, 128]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        unsqueeze_9: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_9, 1);  convert_element_type_9 = None
        expand_3: "f32[1, 512, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_9, _shape_param_11);  unsqueeze_9 = _shape_param_11 = None
        view_7: "f32[512, 128, 128]" = torch.ops.aten.view.default(expand_3, _shape_param_12);  expand_3 = _shape_param_12 = None
        return view_7



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
