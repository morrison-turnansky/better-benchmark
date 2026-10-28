"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 36374350e8de
Shape hash: 73a1b7a7
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
    def forward(self, arg0_1: "f32[1, 512, 4]", arg1_1: "f32[]", arg2_1: "f32[4]", arg3_1: "bf16[512, 256]", arg4_1: "i64[512, 6]", arg5_1: "i64[3072]", arg6_1: "bf16[3072, 4096]", arg7_1: "bf16[512, 4096]", arg8_1: "bf16[512, 4, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        mul: "f32[1, 512, 4]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        add: "f32[1, 512, 4]" = torch.ops.aten.add.Tensor(mul, arg2_1);  mul = arg2_1 = None
        sigmoid: "f32[1, 512, 4]" = torch.ops.aten.sigmoid.default(add);  add = None
        mul_1: "f32[1, 512, 4]" = torch.ops.aten.mul.Tensor(sigmoid, 2);  sigmoid = None
        convert_element_type: "bf16[1, 512, 4]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        unsqueeze: "bf16[1, 512, 4, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        convert_element_type_1: "f32[512, 256]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        gt: "b8[512, 256]" = torch.ops.aten.gt.Scalar(convert_element_type_1, 20)
        exp: "f32[512, 256]" = torch.ops.aten.exp.default(convert_element_type_1)
        log1p: "f32[512, 256]" = torch.ops.aten.log1p.default(exp);  exp = None
        where: "f32[512, 256]" = torch.ops.aten.where.self(gt, convert_element_type_1, log1p);  gt = convert_element_type_1 = log1p = None
        convert_element_type_2: "bf16[512, 256]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        sqrt: "bf16[512, 256]" = torch.ops.aten.sqrt.default(convert_element_type_2);  convert_element_type_2 = None
        gather: "bf16[512, 6]" = torch.ops.aten.gather.default(sqrt, 1, arg4_1);  sqrt = arg4_1 = None
        sum_1: "bf16[512, 1]" = torch.ops.aten.sum.dim_IntList(gather, [-1], True)
        add_1: "bf16[512, 1]" = torch.ops.aten.add.Tensor(sum_1, 1e-20);  sum_1 = None
        div: "bf16[512, 6]" = torch.ops.aten.div.Tensor(gather, add_1);  gather = add_1 = None
        mul_2: "bf16[512, 6]" = torch.ops.aten.mul.Tensor(div, 1.5);  div = None
        view: "bf16[3072]" = torch.ops.aten.view.default(mul_2, [-1]);  mul_2 = None
        index: "bf16[3072]" = torch.ops.aten.index.Tensor(view, [arg5_1]);  view = None
        unsqueeze_1: "bf16[3072, 1]" = torch.ops.aten.unsqueeze.default(index, -1);  index = None
        mul_3: "bf16[3072, 4096]" = torch.ops.aten.mul.Tensor(arg6_1, unsqueeze_1);  arg6_1 = unsqueeze_1 = None
        empty: "i64[3072]" = torch.ops.aten.empty.memory_format(_shape_param_0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        iota: "i64[3072]" = torch.ops.prims.iota.default(3072, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put: "i64[3072]" = torch.ops.aten.index_put.default(empty, [arg5_1], iota);  empty = arg5_1 = iota = None
        index_1: "bf16[3072, 4096]" = torch.ops.aten.index.Tensor(mul_3, [index_put]);  mul_3 = index_put = None
        view_1: "bf16[512, 6, 4096]" = torch.ops.aten.view.default(index_1, _shape_param_1);  index_1 = _shape_param_1 = None
        sum_2: "bf16[512, 4096]" = torch.ops.aten.sum.dim_IntList(view_1, [1]);  view_1 = None
        view_2: "bf16[1, 512, 4096]" = torch.ops.aten.view.default(sum_2, _shape_param_2);  sum_2 = _shape_param_2 = None
        view_3: "bf16[1, 512, 4096]" = torch.ops.aten.view.default(arg7_1, _shape_param_3);  arg7_1 = _shape_param_3 = None
        add_2: "bf16[1, 512, 4096]" = torch.ops.aten.add.Tensor(view_2, view_3);  view_2 = view_3 = None
        unsqueeze_2: "bf16[1, 512, 1, 4096]" = torch.ops.aten.unsqueeze.default(add_2, -2);  add_2 = None
        mul_4: "bf16[1, 512, 4, 4096]" = torch.ops.aten.mul.Tensor(unsqueeze, unsqueeze_2);  unsqueeze = unsqueeze_2 = None
        view_4: "bf16[1, 512, 4, 4096]" = torch.ops.aten.view.default(arg8_1, _shape_param_4);  arg8_1 = _shape_param_4 = None
        add_3: "bf16[1, 512, 4, 4096]" = torch.ops.aten.add.Tensor(mul_4, view_4);  mul_4 = view_4 = None
        view_5: "bf16[1, 512, 16384]" = torch.ops.aten.view.default(add_3, _shape_param_5);  _shape_param_5 = None
        convert_element_type_3: "f32[1, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        pow_1: "f32[1, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_5: "f32[1, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt);  convert_element_type_3 = rsqrt = None
        view_6: "f32[512, 16384]" = torch.ops.aten.view.default(mul_5, _shape_param_6);  mul_5 = _shape_param_6 = None
        expand: "bf16[1, 512, 4, 4096]" = torch.ops.aten.expand.default(add_3, _shape_param_7);  _shape_param_7 = None
        view_7: "bf16[512, 4, 4096]" = torch.ops.aten.view.default(expand, _shape_param_8);  expand = _shape_param_8 = None
        return (add_3, view_6, view_7)



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
