"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 5aec53f5fcc7
Shape hash: 07c23ec3
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
    def forward(self, arg0_1: "bf16[512, 32768]", arg1_1: "i64[1, 512]", arg2_1: "f32[32]", arg3_1: "bf16[512, 512]", arg4_1: "f32[512]", arg5_1: "bf16[512, 1024]", arg6_1: "bf16[512, 1024]", arg7_1: "f32[4, 1024]", arg8_1: "f32[512]", arg9_1: "f32[32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34):
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
        unsqueeze: "i64[1, 512, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 2);  arg1_1 = None
        convert_element_type_3: "f32[1, 512, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze, torch.float32);  unsqueeze = None
        mul_1: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg2_1);  convert_element_type_3 = arg2_1 = None
        cos: "f32[1, 512, 32]" = torch.ops.aten.cos.default(mul_1)
        convert_element_type_4: "bf16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(cos, torch.bfloat16);  cos = None
        unsqueeze_1: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 3)
        expand: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_2);  unsqueeze_1 = _shape_param_2 = None
        clone: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        unsqueeze_2: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_2, 1);  view_2 = None
        mul_2: "f32[1, 64, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_2, unsqueeze_2);  convert_element_type_2 = unsqueeze_2 = None
        slice_3: "bf16[1, 64, 512, 32]" = torch.ops.aten.slice.Tensor(slice_2, 3, 1, 9223372036854775807, 2)
        neg: "bf16[1, 64, 512, 32]" = torch.ops.aten.neg.default(slice_3);  slice_3 = None
        unsqueeze_3: "bf16[1, 64, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        slice_4: "bf16[1, 64, 512, 32]" = torch.ops.aten.slice.Tensor(slice_2, 3, 0, 9223372036854775807, 2);  slice_2 = None
        unsqueeze_4: "bf16[1, 64, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_4, 4);  slice_4 = None
        cat: "bf16[1, 64, 512, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_3, unsqueeze_4], -1);  unsqueeze_3 = unsqueeze_4 = None
        view_3: "bf16[1, 64, 512, 64]" = torch.ops.aten.view.default(cat, _shape_param_4);  cat = _shape_param_4 = None
        convert_element_type_5: "f32[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        sin: "f32[1, 512, 32]" = torch.ops.aten.sin.default(mul_1);  mul_1 = None
        convert_element_type_6: "bf16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(sin, torch.bfloat16);  sin = None
        unsqueeze_5: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3)
        expand_1: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_5, _shape_param_5);  unsqueeze_5 = _shape_param_5 = None
        clone_1: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_4: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        unsqueeze_6: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_4, 1);  view_4 = None
        mul_3: "f32[1, 64, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_5, unsqueeze_6);  convert_element_type_5 = unsqueeze_6 = None
        add_1: "f32[1, 64, 512, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        convert_element_type_7: "bf16[1, 64, 512, 64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        cat_1: "bf16[1, 64, 512, 512]" = torch.ops.aten.cat.default([slice_1, convert_element_type_7], -1);  slice_1 = convert_element_type_7 = None
        expand_2: "bf16[1, 64, 512, 512]" = torch.ops.aten.expand.default(cat_1, _shape_param_7);  cat_1 = _shape_param_7 = None
        view_5: "bf16[64, 512, 512]" = torch.ops.aten.view.default(expand_2, _shape_param_8);  expand_2 = _shape_param_8 = None
        view_6: "bf16[1, 512, 512]" = torch.ops.aten.view.default(arg3_1, _shape_param_9);  arg3_1 = _shape_param_9 = None
        convert_element_type_8: "f32[1, 512, 512]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        pow_2: "f32[1, 512, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 2)
        mean_1: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4: "f32[1, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, rsqrt_1);  convert_element_type_8 = rsqrt_1 = None
        mul_5: "f32[1, 512, 512]" = torch.ops.aten.mul.Tensor(arg4_1, mul_4);  arg4_1 = mul_4 = None
        convert_element_type_9: "bf16[1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_7: "bf16[1, 512, 1, 512]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_10);  convert_element_type_9 = _shape_param_10 = None
        permute_1: "bf16[1, 1, 512, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        slice_5: "bf16[1, 1, 512, 448]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, -64)
        slice_6: "bf16[1, 1, 512, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, -64, 9223372036854775807);  permute_1 = None
        convert_element_type_10: "f32[1, 1, 512, 64]" = torch.ops.prims.convert_element_type.default(slice_6, torch.float32)
        unsqueeze_7: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 3)
        expand_3: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_7, _shape_param_11);  unsqueeze_7 = _shape_param_11 = None
        clone_2: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_8: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_12);  clone_2 = _shape_param_12 = None
        unsqueeze_8: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_8, 1);  view_8 = None
        mul_6: "f32[1, 1, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_10, unsqueeze_8);  convert_element_type_10 = unsqueeze_8 = None
        slice_7: "bf16[1, 1, 512, 32]" = torch.ops.aten.slice.Tensor(slice_6, 3, 1, 9223372036854775807, 2)
        neg_1: "bf16[1, 1, 512, 32]" = torch.ops.aten.neg.default(slice_7);  slice_7 = None
        unsqueeze_9: "bf16[1, 1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
        slice_8: "bf16[1, 1, 512, 32]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 9223372036854775807, 2);  slice_6 = None
        unsqueeze_10: "bf16[1, 1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_8, 4);  slice_8 = None
        cat_2: "bf16[1, 1, 512, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_9, unsqueeze_10], -1);  unsqueeze_9 = unsqueeze_10 = None
        view_9: "bf16[1, 1, 512, 64]" = torch.ops.aten.view.default(cat_2, _shape_param_13);  cat_2 = _shape_param_13 = None
        convert_element_type_11: "f32[1, 1, 512, 64]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        unsqueeze_11: "bf16[1, 512, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3)
        expand_4: "bf16[1, 512, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_11, _shape_param_14);  unsqueeze_11 = _shape_param_14 = None
        clone_3: "bf16[1, 512, 32, 2]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_10: "bf16[1, 512, 64]" = torch.ops.aten.view.default(clone_3, _shape_param_15);  clone_3 = _shape_param_15 = None
        unsqueeze_12: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(view_10, 1);  view_10 = None
        mul_7: "f32[1, 1, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_11, unsqueeze_12);  convert_element_type_11 = unsqueeze_12 = None
        add_3: "f32[1, 1, 512, 64]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        convert_element_type_12: "bf16[1, 1, 512, 64]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        cat_3: "bf16[1, 1, 512, 512]" = torch.ops.aten.cat.default([slice_5, convert_element_type_12], -1);  slice_5 = convert_element_type_12 = None
        full: "bf16[1, 128, 8, 512]" = torch.ops.aten.full.default(_shape_param_16, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_16 = None
        slice_9: "bf16[1, 128, 4, 512]" = torch.ops.aten.slice.Tensor(full, 2, 4, 9223372036854775807)
        view_11: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(arg5_1, _shape_param_17);  arg5_1 = _shape_param_17 = None
        view_12: "bf16[1, 128, 4, 1024]" = torch.ops.aten.view.default(view_11, _shape_param_18);  view_11 = _shape_param_18 = None
        slice_10: "bf16[1, 128, 4, 512]" = torch.ops.aten.slice.Tensor(view_12, 3, 512, 9223372036854775807)
        copy: "bf16[1, 128, 4, 512]" = torch.ops.aten.copy.default(slice_9, slice_10);  slice_9 = slice_10 = None
        slice_scatter: "bf16[1, 128, 8, 512]" = torch.ops.aten.slice_scatter.default(full, copy, 2, 4, 9223372036854775807);  full = copy = None
        slice_11: "bf16[1, 127, 8, 512]" = torch.ops.aten.slice.Tensor(slice_scatter, 1, 1, 9223372036854775807)
        slice_12: "bf16[1, 127, 8, 512]" = torch.ops.aten.slice.Tensor(slice_scatter, 1, 1, 9223372036854775807)
        slice_13: "bf16[1, 127, 4, 512]" = torch.ops.aten.slice.Tensor(slice_12, 2, 0, 4);  slice_12 = None
        slice_14: "bf16[1, 127, 4, 1024]" = torch.ops.aten.slice.Tensor(view_12, 1, 0, -1);  view_12 = None
        slice_15: "bf16[1, 127, 4, 512]" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 512);  slice_14 = None
        copy_1: "bf16[1, 127, 4, 512]" = torch.ops.aten.copy.default(slice_13, slice_15);  slice_13 = slice_15 = None
        slice_scatter_1: "bf16[1, 127, 8, 512]" = torch.ops.aten.slice_scatter.default(slice_11, copy_1, 2, 0, 4);  slice_11 = copy_1 = None
        slice_scatter_2: "bf16[1, 128, 8, 512]" = torch.ops.aten.slice_scatter.default(slice_scatter, slice_scatter_1, 1, 1, 9223372036854775807);  slice_scatter = slice_scatter_1 = None
        full_1: "f32[1, 128, 8, 512]" = torch.ops.aten.full.default(_shape_param_19, -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_19 = None
        slice_16: "f32[1, 128, 4, 512]" = torch.ops.aten.slice.Tensor(full_1, 2, 4, 9223372036854775807)
        view_13: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(arg6_1, _shape_param_20);  arg6_1 = _shape_param_20 = None
        view_14: "bf16[1, 128, 4, 1024]" = torch.ops.aten.view.default(view_13, _shape_param_21);  view_13 = _shape_param_21 = None
        add_4: "f32[1, 128, 4, 1024]" = torch.ops.aten.add.Tensor(view_14, arg7_1);  view_14 = arg7_1 = None
        slice_17: "f32[1, 128, 4, 512]" = torch.ops.aten.slice.Tensor(add_4, 3, 512, 9223372036854775807)
        copy_2: "f32[1, 128, 4, 512]" = torch.ops.aten.copy.default(slice_16, slice_17);  slice_16 = slice_17 = None
        slice_scatter_3: "f32[1, 128, 8, 512]" = torch.ops.aten.slice_scatter.default(full_1, copy_2, 2, 4, 9223372036854775807);  full_1 = copy_2 = None
        slice_18: "f32[1, 127, 8, 512]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 1, 1, 9223372036854775807)
        slice_19: "f32[1, 127, 8, 512]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 1, 1, 9223372036854775807)
        slice_20: "f32[1, 127, 4, 512]" = torch.ops.aten.slice.Tensor(slice_19, 2, 0, 4);  slice_19 = None
        slice_21: "f32[1, 127, 4, 1024]" = torch.ops.aten.slice.Tensor(add_4, 1, 0, -1);  add_4 = None
        slice_22: "f32[1, 127, 4, 512]" = torch.ops.aten.slice.Tensor(slice_21, 3, 0, 512);  slice_21 = None
        copy_3: "f32[1, 127, 4, 512]" = torch.ops.aten.copy.default(slice_20, slice_22);  slice_20 = slice_22 = None
        slice_scatter_4: "f32[1, 127, 8, 512]" = torch.ops.aten.slice_scatter.default(slice_18, copy_3, 2, 0, 4);  slice_18 = copy_3 = None
        slice_scatter_5: "f32[1, 128, 8, 512]" = torch.ops.aten.slice_scatter.default(slice_scatter_3, slice_scatter_4, 1, 1, 9223372036854775807);  slice_scatter_3 = slice_scatter_4 = None
        amax: "f32[1, 128, 1, 512]" = torch.ops.aten.amax.default(slice_scatter_5, [2], True)
        sub: "f32[1, 128, 8, 512]" = torch.ops.aten.sub.Tensor(slice_scatter_5, amax);  slice_scatter_5 = amax = None
        exp: "f32[1, 128, 8, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1, 128, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp, [2], True)
        div: "f32[1, 128, 8, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_13: "bf16[1, 128, 8, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul_8: "bf16[1, 128, 8, 512]" = torch.ops.aten.mul.Tensor(slice_scatter_2, convert_element_type_13);  slice_scatter_2 = convert_element_type_13 = None
        sum_2: "bf16[1, 128, 512]" = torch.ops.aten.sum.dim_IntList(mul_8, [2]);  mul_8 = None
        convert_element_type_14: "f32[1, 128, 512]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        pow_3: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_14, 2)
        mean_2: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_5: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_9: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_14, rsqrt_2);  convert_element_type_14 = rsqrt_2 = None
        mul_10: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg8_1, mul_9);  arg8_1 = mul_9 = None
        convert_element_type_15: "bf16[1, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        unsqueeze_13: "bf16[1, 1, 128, 512]" = torch.ops.aten.unsqueeze.default(convert_element_type_15, 1);  convert_element_type_15 = None
        slice_23: "bf16[1, 1, 128, 448]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 0, -64)
        slice_24: "bf16[1, 1, 128, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, -64, 9223372036854775807);  unsqueeze_13 = None
        convert_element_type_16: "f32[1, 1, 128, 64]" = torch.ops.prims.convert_element_type.default(slice_24, torch.float32)
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_11: "i64[128]" = torch.ops.aten.mul.Tensor(iota, 4);  iota = None
        unsqueeze_14: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(mul_11, 0);  mul_11 = None
        expand_5: "i64[1, 128]" = torch.ops.aten.expand.default(unsqueeze_14, [1, -1]);  unsqueeze_14 = None
        unsqueeze_15: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(expand_5, 2);  expand_5 = None
        convert_element_type_17: "f32[1, 128, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_15, torch.float32);  unsqueeze_15 = None
        mul_12: "f32[1, 128, 32]" = torch.ops.aten.mul.Tensor(convert_element_type_17, arg9_1);  convert_element_type_17 = arg9_1 = None
        cos_1: "f32[1, 128, 32]" = torch.ops.aten.cos.default(mul_12)
        convert_element_type_18: "bf16[1, 128, 32]" = torch.ops.prims.convert_element_type.default(cos_1, torch.bfloat16);  cos_1 = None
        unsqueeze_16: "bf16[1, 128, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, 3);  convert_element_type_18 = None
        expand_6: "bf16[1, 128, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_16, _shape_param_22);  unsqueeze_16 = _shape_param_22 = None
        clone_4: "bf16[1, 128, 32, 2]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_15: "bf16[1, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_23);  clone_4 = _shape_param_23 = None
        unsqueeze_17: "bf16[1, 1, 128, 64]" = torch.ops.aten.unsqueeze.default(view_15, 1);  view_15 = None
        mul_13: "f32[1, 1, 128, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_16, unsqueeze_17);  convert_element_type_16 = unsqueeze_17 = None
        slice_25: "bf16[1, 1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_24, 3, 1, 9223372036854775807, 2)
        neg_2: "bf16[1, 1, 128, 32]" = torch.ops.aten.neg.default(slice_25);  slice_25 = None
        unsqueeze_18: "bf16[1, 1, 128, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_2, 4);  neg_2 = None
        slice_26: "bf16[1, 1, 128, 32]" = torch.ops.aten.slice.Tensor(slice_24, 3, 0, 9223372036854775807, 2);  slice_24 = None
        unsqueeze_19: "bf16[1, 1, 128, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_26, 4);  slice_26 = None
        cat_4: "bf16[1, 1, 128, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_18, unsqueeze_19], -1);  unsqueeze_18 = unsqueeze_19 = None
        view_16: "bf16[1, 1, 128, 64]" = torch.ops.aten.view.default(cat_4, _shape_param_24);  cat_4 = _shape_param_24 = None
        convert_element_type_19: "f32[1, 1, 128, 64]" = torch.ops.prims.convert_element_type.default(view_16, torch.float32);  view_16 = None
        sin_1: "f32[1, 128, 32]" = torch.ops.aten.sin.default(mul_12);  mul_12 = None
        convert_element_type_20: "bf16[1, 128, 32]" = torch.ops.prims.convert_element_type.default(sin_1, torch.bfloat16);  sin_1 = None
        unsqueeze_20: "bf16[1, 128, 32, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, 3);  convert_element_type_20 = None
        expand_7: "bf16[1, 128, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_20, _shape_param_25);  unsqueeze_20 = _shape_param_25 = None
        clone_5: "bf16[1, 128, 32, 2]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_17: "bf16[1, 128, 64]" = torch.ops.aten.view.default(clone_5, _shape_param_26);  clone_5 = _shape_param_26 = None
        unsqueeze_21: "bf16[1, 1, 128, 64]" = torch.ops.aten.unsqueeze.default(view_17, 1);  view_17 = None
        mul_14: "f32[1, 1, 128, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_19, unsqueeze_21);  convert_element_type_19 = unsqueeze_21 = None
        add_6: "f32[1, 1, 128, 64]" = torch.ops.aten.add.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
        convert_element_type_21: "bf16[1, 1, 128, 64]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        cat_5: "bf16[1, 1, 128, 512]" = torch.ops.aten.cat.default([slice_23, convert_element_type_21], -1);  slice_23 = convert_element_type_21 = None
        squeeze: "bf16[1, 128, 512]" = torch.ops.aten.squeeze.dim(cat_5, 1);  cat_5 = None
        unsqueeze_22: "bf16[1, 1, 128, 512]" = torch.ops.aten.unsqueeze.default(squeeze, 1);  squeeze = None
        cat_6: "bf16[1, 1, 640, 512]" = torch.ops.aten.cat.default([cat_3, unsqueeze_22], 2);  cat_3 = unsqueeze_22 = None
        unsqueeze_23: "bf16[1, 1, 1, 640, 512]" = torch.ops.aten.unsqueeze.default(cat_6, 2)
        expand_8: "bf16[1, 1, 64, 640, 512]" = torch.ops.aten.expand.default(unsqueeze_23, _shape_param_27);  unsqueeze_23 = _shape_param_27 = None
        view_18: "bf16[1, 64, 640, 512]" = torch.ops.aten.view.default(expand_8, _shape_param_28);  expand_8 = _shape_param_28 = None
        permute_2: "bf16[1, 64, 512, 640]" = torch.ops.aten.permute.default(view_18, [0, 1, 3, 2]);  view_18 = None
        expand_9: "bf16[1, 64, 512, 640]" = torch.ops.aten.expand.default(permute_2, _shape_param_29);  permute_2 = _shape_param_29 = None
        view_19: "bf16[64, 512, 640]" = torch.ops.aten.view.default(expand_9, _shape_param_30);  expand_9 = _shape_param_30 = None
        unsqueeze_24: "bf16[1, 1, 1, 640, 512]" = torch.ops.aten.unsqueeze.default(cat_6, 2);  cat_6 = None
        expand_10: "bf16[1, 1, 64, 640, 512]" = torch.ops.aten.expand.default(unsqueeze_24, _shape_param_31);  unsqueeze_24 = _shape_param_31 = None
        view_20: "bf16[1, 64, 640, 512]" = torch.ops.aten.view.default(expand_10, _shape_param_32);  expand_10 = _shape_param_32 = None
        expand_11: "bf16[1, 64, 640, 512]" = torch.ops.aten.expand.default(view_20, _shape_param_33);  view_20 = _shape_param_33 = None
        view_21: "bf16[64, 640, 512]" = torch.ops.aten.view.default(expand_11, _shape_param_34);  expand_11 = _shape_param_34 = None
        return (convert_element_type_4, convert_element_type_6, view_5, view_19, view_21)



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
