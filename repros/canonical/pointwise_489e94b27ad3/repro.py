"""
Standalone repro captured via capture_hook.
Label: deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm
Pattern hash: 489e94b27ad3
Shape hash: 29aadb49
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
    def forward(self, arg0_1: "bf16[512, 256]", arg1_1: "bf16[256]", arg2_1: "bf16[1, 512, 4096]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[512, 256]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        gt: "b8[512, 256]" = torch.ops.aten.gt.Scalar(convert_element_type, 20)
        exp: "f32[512, 256]" = torch.ops.aten.exp.default(convert_element_type)
        log1p: "f32[512, 256]" = torch.ops.aten.log1p.default(exp);  exp = None
        where: "f32[512, 256]" = torch.ops.aten.where.self(gt, convert_element_type, log1p);  gt = convert_element_type = log1p = None
        convert_element_type_1: "bf16[512, 256]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        sqrt: "bf16[512, 256]" = torch.ops.aten.sqrt.default(convert_element_type_1);  convert_element_type_1 = None
        add: "bf16[512, 256]" = torch.ops.aten.add.Tensor(sqrt, arg1_1);  arg1_1 = None
        topk = torch.ops.aten.topk.default(add, 6, -1, True, False);  add = None
        getitem: "bf16[512, 6]" = topk[0];  getitem = None
        getitem_1: "i64[512, 6]" = topk[1];  topk = None
        view: "i64[3072]" = torch.ops.aten.view.default(getitem_1, [-1])
        sort = torch.ops.aten.sort.default(view);  view = None
        getitem_2: "i64[3072]" = sort[0]
        getitem_3: "i64[3072]" = sort[1];  sort = None
        view_1: "bf16[512, 4096]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        div: "i64[3072]" = torch.ops.aten.div.Tensor_mode(getitem_3, 6, rounding_mode = 'floor')
        index: "bf16[3072, 4096]" = torch.ops.aten.index.Tensor(view_1, [div]);  view_1 = div = None
        convert_element_type_2: "i32[3072]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.int32);  getitem_2 = None
        return (sqrt, getitem_1, getitem_3, index, convert_element_type_2)



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
