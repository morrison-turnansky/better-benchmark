"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 1527efa5ddfb
Shape hash: fee2783e
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
    def forward(self, arg0_1: "bf16[64, 512, 520]", arg1_1: "i64[1, 512]", arg2_1: "bf16[1, 1, 512, 512]", arg3_1: "f32[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        slice_1: "bf16[64, 512, 516]" = torch.ops.aten.slice.Tensor(arg0_1, 2, 0, -4);  arg0_1 = None
        view: "bf16[1, 64, 512, 516]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        mul: "bf16[1, 64, 512, 516]" = torch.ops.aten.mul.Tensor(view, 0.04419417382415922);  view = None
        iota: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 1, 1, 4]" = torch.ops.aten.view.default(iota, [1, 1, 1, -1]);  iota = None
        add: "i64[1, 512]" = torch.ops.aten.add.Tensor(arg1_1, 1);  arg1_1 = None
        div: "i64[1, 512]" = torch.ops.aten.div.Tensor_mode(add, 128, rounding_mode = 'floor');  add = None
        unsqueeze: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(div, 1);  div = None
        unsqueeze_1: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        ge: "b8[1, 1, 512, 4]" = torch.ops.aten.ge.Tensor(view_1, unsqueeze_1);  view_1 = unsqueeze_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "bf16[1, 1, 512, 4]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        where: "bf16[1, 1, 512, 4]" = torch.ops.aten.where.self(ge, full, full_1);  ge = full = full_1 = None
        cat: "bf16[1, 1, 512, 516]" = torch.ops.aten.cat.default([arg2_1, where], -1);  arg2_1 = where = None
        add_1: "bf16[1, 64, 512, 516]" = torch.ops.aten.add.Tensor(mul, cat);  mul = cat = None
        view_2: "f32[1, 64, 1, 1]" = torch.ops.aten.view.default(arg3_1, [1, -1, 1, 1]);  arg3_1 = None
        expand: "f32[1, 64, 512, 1]" = torch.ops.aten.expand.default(view_2, _shape_param_2);  view_2 = _shape_param_2 = None
        cat_1: "f32[1, 64, 512, 517]" = torch.ops.aten.cat.default([add_1, expand], -1);  add_1 = expand = None
        max_1 = torch.ops.aten.max.dim(cat_1, -1, True)
        getitem: "f32[1, 64, 512, 1]" = max_1[0]
        getitem_1: "i64[1, 64, 512, 1]" = max_1[1];  max_1 = getitem_1 = None
        sub: "f32[1, 64, 512, 517]" = torch.ops.aten.sub.Tensor(cat_1, getitem);  cat_1 = getitem = None
        amax: "f32[1, 64, 512, 1]" = torch.ops.aten.amax.default(sub, [-1], True)
        sub_1: "f32[1, 64, 512, 517]" = torch.ops.aten.sub.Tensor(sub, amax);  sub = amax = None
        exp: "f32[1, 64, 512, 517]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[1, 64, 512, 517]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        slice_2: "f32[1, 64, 512, 516]" = torch.ops.aten.slice.Tensor(div_1, 3, 0, -1);  div_1 = None
        convert_element_type: "bf16[1, 64, 512, 516]" = torch.ops.prims.convert_element_type.default(slice_2, torch.bfloat16);  slice_2 = None
        expand_1: "bf16[1, 64, 512, 516]" = torch.ops.aten.expand.default(convert_element_type, _shape_param_3);  convert_element_type = _shape_param_3 = None
        view_3: "bf16[64, 512, 516]" = torch.ops.aten.view.default(expand_1, _shape_param_4);  expand_1 = _shape_param_4 = None
        constant_pad_nd: "bf16[64, 512, 520]" = torch.ops.aten.constant_pad_nd.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        return constant_pad_nd



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
