"""
Standalone repro captured via capture_hook.
Label: Gemma4_31B_text_6l_prefill_b1_s2048_sdpa_math_bf16
Pattern hash: 7bf91bcadfc1
Shape hash: 0b40ae89
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
    def forward(self, arg0_1: "f32[32, 2048, 2048]", arg1_1: "i64[1, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "f32[1, 32, 2048, 2048]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        unsqueeze_1: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        iota_1: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(iota_1, 0);  iota_1 = None
        unsqueeze_4: "i64[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        sub: "i64[1, 1, 2048, 1]" = torch.ops.aten.sub.Tensor(unsqueeze_5, 1024)
        gt: "b8[1, 1, 2048, 2048]" = torch.ops.aten.gt.Tensor(unsqueeze_2, sub);  sub = None
        bitwise_and: "b8[1, 1, 2048, 2048]" = torch.ops.aten.bitwise_and.Tensor(full_1, gt);  full_1 = gt = None
        le: "b8[1, 1, 2048, 2048]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5)
        bitwise_and_1: "b8[1, 1, 2048, 2048]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, le);  bitwise_and = le = None
        bitwise_and_2: "b8[1, 1, 2048, 2048]" = torch.ops.aten.bitwise_and.Tensor(full, bitwise_and_1);  full = bitwise_and_1 = None
        iota_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 1);  iota_2 = None
        unsqueeze_7: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        index: "i64[1, 1, 2048, 1]" = torch.ops.aten.index.Tensor(arg1_1, [unsqueeze_8, unsqueeze_5]);  unsqueeze_5 = None
        index_1: "i64[1, 1, 1, 2048]" = torch.ops.aten.index.Tensor(arg1_1, [unsqueeze_8, unsqueeze_2]);  arg1_1 = unsqueeze_8 = unsqueeze_2 = None
        eq: "b8[1, 1, 2048, 2048]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_3: "b8[1, 1, 2048, 2048]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, eq);  bitwise_and_2 = eq = None
        expand: "b8[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(bitwise_and_3, _shape_param_1);  bitwise_and_3 = _shape_param_1 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 2048, 2048]" = torch.ops.aten.where.self(expand, full_2, full_3);  full_2 = full_3 = None
        add: "f32[1, 32, 2048, 2048]" = torch.ops.aten.add.Tensor(view, where);  view = where = None
        eq_1: "b8[1, 32, 2048, 2048]" = torch.ops.aten.eq.Scalar(add, -inf)
        logical_not: "b8[1, 32, 2048, 2048]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_1: "b8[1, 32, 2048, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[1, 32, 2048, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_4: "f32[1, 32, 2048, 2048]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        amax: "f32[1, 32, 2048, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub_1: "f32[1, 32, 2048, 2048]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[1, 32, 2048, 2048]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 32, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 32, 2048, 2048]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_1: "f32[1, 32, 2048, 2048]" = torch.ops.aten.where.self(logical_not_1, full_4, div);  logical_not_1 = full_4 = div = None
        expand_1: "f32[1, 32, 2048, 2048]" = torch.ops.aten.expand.default(where_1, _shape_param_3);  where_1 = _shape_param_3 = None
        view_1: "f32[32, 2048, 2048]" = torch.ops.aten.view.default(expand_1, _shape_param_4);  expand_1 = _shape_param_4 = None
        return (expand, view_1)



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
