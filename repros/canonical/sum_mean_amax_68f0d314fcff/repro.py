"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 68f0d314fcff
Shape hash: 622052d8
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
    def forward(self, arg0_1: "f32[512, 24]", arg1_1: "f32[24]", arg2_1: "f32[3]", arg3_1: "bf16[1, 512, 4, 4096]", arg4_1: "f32[4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        view: "f32[1, 512, 24]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view, _shape_param_1, -1);  view = _shape_param_1 = None
        getitem: "f32[1, 512, 4]" = split_with_sizes[0]
        getitem_1: "f32[1, 512, 4]" = split_with_sizes[1]
        getitem_2: "f32[1, 512, 16]" = split_with_sizes[2];  split_with_sizes = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(arg1_1, _shape_param_2);  arg1_1 = _shape_param_2 = None
        getitem_3: "f32[4]" = split_with_sizes_1[0]
        getitem_4: "f32[4]" = split_with_sizes_1[1]
        getitem_5: "f32[16]" = split_with_sizes_1[2];  split_with_sizes_1 = None
        unbind = torch.ops.aten.unbind.int(arg2_1);  arg2_1 = None
        getitem_6: "f32[]" = unbind[0]
        getitem_7: "f32[]" = unbind[1]
        getitem_8: "f32[]" = unbind[2];  unbind = None
        mul: "f32[1, 512, 4]" = torch.ops.aten.mul.Tensor(getitem, getitem_6);  getitem = getitem_6 = None
        add: "f32[1, 512, 4]" = torch.ops.aten.add.Tensor(mul, getitem_3);  mul = getitem_3 = None
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
        view_1: "bf16[512, 4096]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_3);  _shape_param_3 = None
        view_2: "bf16[512, 4096]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_4);  _shape_param_4 = None
        view_3: "bf16[512, 4096]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_5);  _shape_param_5 = None
        view_4: "f32[1, 512, 4, 4]" = torch.ops.aten.view.default(getitem_2, _shape_param_6);  getitem_2 = _shape_param_6 = None
        mul_4: "f32[1, 512, 4, 4]" = torch.ops.aten.mul.Tensor(view_4, getitem_8);  view_4 = getitem_8 = None
        view_5: "f32[4, 4]" = torch.ops.aten.view.default(getitem_5, _shape_param_7);  getitem_5 = _shape_param_7 = None
        add_3: "f32[1, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_4, view_5);  mul_4 = view_5 = None
        amax: "f32[1, 512, 4, 1]" = torch.ops.aten.amax.default(add_3, [-1], True)
        sub: "f32[1, 512, 4, 4]" = torch.ops.aten.sub.Tensor(add_3, amax);  add_3 = amax = None
        exp: "f32[1, 512, 4, 4]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_2: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(exp, sum_2);  exp = sum_2 = None
        add_4: "f32[1, 512, 4, 4]" = torch.ops.aten.add.Tensor(div, 1e-06);  div = None
        sum_3: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(add_4, [-2], True)
        add_5: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_3, 1e-06);  sum_3 = None
        div_1: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(add_4, add_5);  add_4 = add_5 = None
        sum_4: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_1, [-1], True)
        add_6: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_4, 1e-06);  sum_4 = None
        div_2: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_1, add_6);  div_1 = add_6 = None
        sum_5: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_2, [-2], True)
        add_7: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_5, 1e-06);  sum_5 = None
        div_3: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_2, add_7);  div_2 = add_7 = None
        sum_6: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_3, [-1], True)
        add_8: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_6, 1e-06);  sum_6 = None
        div_4: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_3, add_8);  div_3 = add_8 = None
        sum_7: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_4, [-2], True)
        add_9: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_7, 1e-06);  sum_7 = None
        div_5: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_4, add_9);  div_4 = add_9 = None
        sum_8: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_5, [-1], True)
        add_10: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_8, 1e-06);  sum_8 = None
        div_6: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_5, add_10);  div_5 = add_10 = None
        sum_9: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_6, [-2], True)
        add_11: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_9, 1e-06);  sum_9 = None
        div_7: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_6, add_11);  div_6 = add_11 = None
        sum_10: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_7, [-1], True)
        add_12: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_10, 1e-06);  sum_10 = None
        div_8: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_7, add_12);  div_7 = add_12 = None
        sum_11: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_8, [-2], True)
        add_13: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_11, 1e-06);  sum_11 = None
        div_9: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_8, add_13);  div_8 = add_13 = None
        sum_12: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_9, [-1], True)
        add_14: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_12, 1e-06);  sum_12 = None
        div_10: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_9, add_14);  div_9 = add_14 = None
        sum_13: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_10, [-2], True)
        add_15: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_13, 1e-06);  sum_13 = None
        div_11: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_10, add_15);  div_10 = add_15 = None
        sum_14: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_11, [-1], True)
        add_16: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_14, 1e-06);  sum_14 = None
        div_12: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_11, add_16);  div_11 = add_16 = None
        sum_15: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_12, [-2], True)
        add_17: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_15, 1e-06);  sum_15 = None
        div_13: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_12, add_17);  div_12 = add_17 = None
        sum_16: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_13, [-1], True)
        add_18: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_16, 1e-06);  sum_16 = None
        div_14: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_13, add_18);  div_13 = add_18 = None
        sum_17: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_14, [-2], True)
        add_19: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_17, 1e-06);  sum_17 = None
        div_15: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_14, add_19);  div_14 = add_19 = None
        sum_18: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_15, [-1], True)
        add_20: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_18, 1e-06);  sum_18 = None
        div_16: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_15, add_20);  div_15 = add_20 = None
        sum_19: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_16, [-2], True)
        add_21: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_19, 1e-06);  sum_19 = None
        div_17: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_16, add_21);  div_16 = add_21 = None
        sum_20: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_17, [-1], True)
        add_22: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_20, 1e-06);  sum_20 = None
        div_18: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_17, add_22);  div_17 = add_22 = None
        sum_21: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_18, [-2], True)
        add_23: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_21, 1e-06);  sum_21 = None
        div_19: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_18, add_23);  div_18 = add_23 = None
        sum_22: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_19, [-1], True)
        add_24: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_22, 1e-06);  sum_22 = None
        div_20: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_19, add_24);  div_19 = add_24 = None
        sum_23: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_20, [-2], True)
        add_25: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_23, 1e-06);  sum_23 = None
        div_21: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_20, add_25);  div_20 = add_25 = None
        sum_24: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_21, [-1], True)
        add_26: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_24, 1e-06);  sum_24 = None
        div_22: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_21, add_26);  div_21 = add_26 = None
        sum_25: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_22, [-2], True)
        add_27: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_25, 1e-06);  sum_25 = None
        div_23: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_22, add_27);  div_22 = add_27 = None
        sum_26: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_23, [-1], True)
        add_28: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_26, 1e-06);  sum_26 = None
        div_24: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_23, add_28);  div_23 = add_28 = None
        sum_27: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_24, [-2], True)
        add_29: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_27, 1e-06);  sum_27 = None
        div_25: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_24, add_29);  div_24 = add_29 = None
        sum_28: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_25, [-1], True)
        add_30: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_28, 1e-06);  sum_28 = None
        div_26: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_25, add_30);  div_25 = add_30 = None
        sum_29: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_26, [-2], True)
        add_31: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_29, 1e-06);  sum_29 = None
        div_27: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_26, add_31);  div_26 = add_31 = None
        sum_30: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_27, [-1], True)
        add_32: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_30, 1e-06);  sum_30 = None
        div_28: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_27, add_32);  div_27 = add_32 = None
        sum_31: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_28, [-2], True)
        add_33: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_31, 1e-06);  sum_31 = None
        div_29: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_28, add_33);  div_28 = add_33 = None
        sum_32: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_29, [-1], True)
        add_34: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_32, 1e-06);  sum_32 = None
        div_30: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_29, add_34);  div_29 = add_34 = None
        sum_33: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_30, [-2], True)
        add_35: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_33, 1e-06);  sum_33 = None
        div_31: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_30, add_35);  div_30 = add_35 = None
        sum_34: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_31, [-1], True)
        add_36: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_34, 1e-06);  sum_34 = None
        div_32: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_31, add_36);  div_31 = add_36 = None
        sum_35: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_32, [-2], True)
        add_37: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_35, 1e-06);  sum_35 = None
        div_33: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_32, add_37);  div_32 = add_37 = None
        sum_36: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_33, [-1], True)
        add_38: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_36, 1e-06);  sum_36 = None
        div_34: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_33, add_38);  div_33 = add_38 = None
        sum_37: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_34, [-2], True)
        add_39: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_37, 1e-06);  sum_37 = None
        div_35: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_34, add_39);  div_34 = add_39 = None
        sum_38: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_35, [-1], True)
        add_40: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_38, 1e-06);  sum_38 = None
        div_36: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_35, add_40);  div_35 = add_40 = None
        sum_39: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_36, [-2], True)
        add_41: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_39, 1e-06);  sum_39 = None
        div_37: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_36, add_41);  div_36 = add_41 = None
        sum_40: "f32[1, 512, 4, 1]" = torch.ops.aten.sum.dim_IntList(div_37, [-1], True)
        add_42: "f32[1, 512, 4, 1]" = torch.ops.aten.add.Tensor(sum_40, 1e-06);  sum_40 = None
        div_38: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_37, add_42);  div_37 = add_42 = None
        sum_41: "f32[1, 512, 1, 4]" = torch.ops.aten.sum.dim_IntList(div_38, [-2], True)
        add_43: "f32[1, 512, 1, 4]" = torch.ops.aten.add.Tensor(sum_41, 1e-06);  sum_41 = None
        div_39: "f32[1, 512, 4, 4]" = torch.ops.aten.div.Tensor(div_38, add_43);  div_38 = add_43 = None
        convert_element_type_3: "bf16[1, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(div_39, torch.bfloat16);  div_39 = None
        permute: "bf16[1, 512, 4, 4]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 1, 3, 2]);  convert_element_type_3 = None
        expand: "bf16[1, 512, 4, 4]" = torch.ops.aten.expand.default(permute, _shape_param_8);  permute = _shape_param_8 = None
        view_6: "bf16[512, 4, 4]" = torch.ops.aten.view.default(expand, _shape_param_9);  expand = _shape_param_9 = None
        return (getitem_1, getitem_4, getitem_7, convert_element_type_2, view_1, view_2, view_3, view_6)



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
