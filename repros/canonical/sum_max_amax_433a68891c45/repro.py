"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 433a68891c45
Shape hash: 339e09ce
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f32[512, 64, 128]", arg2_1: "bf16[512, 64]", arg3_1: "bf16[64, 512, 640]", arg4_1: "bf16[1, 1, 512, 512]", arg5_1: "f32[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view: "i64[1, 1, 128]" = torch.ops.aten.view.default(iota, [1, 1, -1]);  iota = None
        add: "i64[1, 512]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
        div: "i64[1, 512]" = torch.ops.aten.div.Tensor_mode(add, 4, rounding_mode = 'floor');  add = None
        unsqueeze: "i64[1, 512, 1]" = torch.ops.aten.unsqueeze.default(div, -1)
        ge: "b8[1, 512, 128]" = torch.ops.aten.ge.Tensor(view, unsqueeze);  view = unsqueeze = None
        full: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_1: "f32[1, 512, 64, 128]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        relu: "f32[1, 512, 64, 128]" = torch.ops.aten.relu.default(view_1);  view_1 = None
        mul: "f32[1, 512, 64, 128]" = torch.ops.aten.mul.Tensor(relu, 0.08838834764831845);  relu = None
        view_2: "bf16[1, 512, 64]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type: "f32[1, 512, 64]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_1: "f32[1, 512, 64]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.125);  convert_element_type = None
        unsqueeze_1: "f32[1, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(mul_1, -1);  mul_1 = None
        mul_2: "f32[1, 512, 64, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        sum_1: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_2, [2]);  mul_2 = None
        where: "f32[1, 512, 128]" = torch.ops.aten.where.self(ge, full, sum_1);  ge = full = sum_1 = None
        topk = torch.ops.aten.topk.default(where, 128);  where = None
        getitem: "f32[1, 512, 128]" = topk[0];  getitem = None
        getitem_1: "i64[1, 512, 128]" = topk[1];  topk = None
        view_3: "bf16[1, 64, 512, 640]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        mul_3: "bf16[1, 64, 512, 640]" = torch.ops.aten.mul.Tensor(view_3, 0.04419417382415922);  view_3 = None
        full_1: "bf16[1, 1, 512, 129]" = torch.ops.aten.full.default(_shape_param_3, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        unsqueeze_2: "i64[1, 512, 1]" = torch.ops.aten.unsqueeze.default(div, -1);  div = None
        ge_1: "b8[1, 512, 128]" = torch.ops.aten.ge.Tensor(getitem_1, unsqueeze_2);  unsqueeze_2 = None
        full_2: "i64[1, 512, 128]" = torch.ops.aten.full.default(_shape_param_4, -1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        where_1: "i64[1, 512, 128]" = torch.ops.aten.where.self(ge_1, full_2, getitem_1);  ge_1 = full_2 = getitem_1 = None
        ge_2: "b8[1, 512, 128]" = torch.ops.aten.ge.Scalar(where_1, 0)
        full_3: "i64[1, 512, 128]" = torch.ops.aten.full.default(_shape_param_5, 128, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        where_2: "i64[1, 512, 128]" = torch.ops.aten.where.self(ge_2, where_1, full_3);  ge_2 = where_1 = full_3 = None
        unsqueeze_3: "i64[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        scatter: "bf16[1, 1, 512, 129]" = torch.ops.aten.scatter.value(full_1, -1, unsqueeze_3, 0.0);  full_1 = unsqueeze_3 = None
        slice_1: "bf16[1, 1, 512, 128]" = torch.ops.aten.slice.Tensor(scatter, 3, 0, 128);  scatter = None
        cat: "bf16[1, 1, 512, 640]" = torch.ops.aten.cat.default([arg4_1, slice_1], -1);  arg4_1 = slice_1 = None
        add_1: "bf16[1, 64, 512, 640]" = torch.ops.aten.add.Tensor(mul_3, cat);  mul_3 = cat = None
        view_4: "f32[1, 64, 1, 1]" = torch.ops.aten.view.default(arg5_1, [1, -1, 1, 1]);  arg5_1 = None
        expand: "f32[1, 64, 512, 1]" = torch.ops.aten.expand.default(view_4, _shape_param_6);  view_4 = _shape_param_6 = None
        cat_1: "f32[1, 64, 512, 641]" = torch.ops.aten.cat.default([add_1, expand], -1);  add_1 = expand = None
        max_1 = torch.ops.aten.max.dim(cat_1, -1, True)
        getitem_2: "f32[1, 64, 512, 1]" = max_1[0]
        getitem_3: "i64[1, 64, 512, 1]" = max_1[1];  max_1 = getitem_3 = None
        sub: "f32[1, 64, 512, 641]" = torch.ops.aten.sub.Tensor(cat_1, getitem_2);  cat_1 = getitem_2 = None
        amax: "f32[1, 64, 512, 1]" = torch.ops.aten.amax.default(sub, [-1], True)
        sub_1: "f32[1, 64, 512, 641]" = torch.ops.aten.sub.Tensor(sub, amax);  sub = amax = None
        exp: "f32[1, 64, 512, 641]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_2: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[1, 64, 512, 641]" = torch.ops.aten.div.Tensor(exp, sum_2);  exp = sum_2 = None
        slice_2: "f32[1, 64, 512, 640]" = torch.ops.aten.slice.Tensor(div_1, 3, 0, -1);  div_1 = None
        convert_element_type_1: "bf16[1, 64, 512, 640]" = torch.ops.prims.convert_element_type.default(slice_2, torch.bfloat16);  slice_2 = None
        expand_1: "bf16[1, 64, 512, 640]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_7);  convert_element_type_1 = _shape_param_7 = None
        view_5: "bf16[64, 512, 640]" = torch.ops.aten.view.default(expand_1, _shape_param_8);  expand_1 = _shape_param_8 = None
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
