class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[262144, 5376][5376, 1]cuda:0", arg1_1: "i64[1, 2048][2048, 1]cuda:0", arg2_1: "bf16[][]cuda:0", arg3_1: "f32[128][1]cuda:0", arg4_1: "f32[256][1]cuda:0", arg5_1: "bf16[5376][1]cuda:0", arg6_1: "bf16[8192, 5376][5376, 1]cuda:0", arg7_1: "bf16[256][1]cuda:0", arg8_1: "bf16[4096, 5376][5376, 1]cuda:0", arg9_1: "bf16[4096, 5376][5376, 1]cuda:0", arg10_1: "bf16[256][1]cuda:0", arg11_1: "bf16[5376, 8192][8192, 1]cuda:0", arg12_1: "bf16[5376][1]cuda:0", arg13_1: "bf16[5376][1]cuda:0", arg14_1: "bf16[21504, 5376][5376, 1]cuda:0", arg15_1: "bf16[21504, 5376][5376, 1]cuda:0", arg16_1: "bf16[5376, 21504][21504, 1]cuda:0", arg17_1: "bf16[5376][1]cuda:0", arg18_1: "bf16[1][1]cuda:0", arg19_1: "bf16[5376][1]cuda:0", arg20_1: "bf16[8192, 5376][5376, 1]cuda:0", arg21_1: "bf16[256][1]cuda:0", arg22_1: "bf16[4096, 5376][5376, 1]cuda:0", arg23_1: "bf16[4096, 5376][5376, 1]cuda:0", arg24_1: "bf16[256][1]cuda:0", arg25_1: "bf16[5376, 8192][8192, 1]cuda:0", arg26_1: "bf16[5376][1]cuda:0", arg27_1: "bf16[5376][1]cuda:0", arg28_1: "bf16[21504, 5376][5376, 1]cuda:0", arg29_1: "bf16[21504, 5376][5376, 1]cuda:0", arg30_1: "bf16[5376, 21504][21504, 1]cuda:0", arg31_1: "bf16[5376][1]cuda:0", arg32_1: "bf16[1][1]cuda:0", arg33_1: "bf16[5376][1]cuda:0", arg34_1: "bf16[8192, 5376][5376, 1]cuda:0", arg35_1: "bf16[256][1]cuda:0", arg36_1: "bf16[4096, 5376][5376, 1]cuda:0", arg37_1: "bf16[4096, 5376][5376, 1]cuda:0", arg38_1: "bf16[256][1]cuda:0", arg39_1: "bf16[5376, 8192][8192, 1]cuda:0", arg40_1: "bf16[5376][1]cuda:0", arg41_1: "bf16[5376][1]cuda:0", arg42_1: "bf16[21504, 5376][5376, 1]cuda:0", arg43_1: "bf16[21504, 5376][5376, 1]cuda:0", arg44_1: "bf16[5376, 21504][21504, 1]cuda:0", arg45_1: "bf16[5376][1]cuda:0", arg46_1: "bf16[1][1]cuda:0", arg47_1: "bf16[5376][1]cuda:0", arg48_1: "bf16[8192, 5376][5376, 1]cuda:0", arg49_1: "bf16[256][1]cuda:0", arg50_1: "bf16[4096, 5376][5376, 1]cuda:0", arg51_1: "bf16[4096, 5376][5376, 1]cuda:0", arg52_1: "bf16[256][1]cuda:0", arg53_1: "bf16[5376, 8192][8192, 1]cuda:0", arg54_1: "bf16[5376][1]cuda:0", arg55_1: "bf16[5376][1]cuda:0", arg56_1: "bf16[21504, 5376][5376, 1]cuda:0", arg57_1: "bf16[21504, 5376][5376, 1]cuda:0", arg58_1: "bf16[5376, 21504][21504, 1]cuda:0", arg59_1: "bf16[5376][1]cuda:0", arg60_1: "bf16[1][1]cuda:0", arg61_1: "bf16[5376][1]cuda:0", arg62_1: "bf16[8192, 5376][5376, 1]cuda:0", arg63_1: "bf16[256][1]cuda:0", arg64_1: "bf16[4096, 5376][5376, 1]cuda:0", arg65_1: "bf16[4096, 5376][5376, 1]cuda:0", arg66_1: "bf16[256][1]cuda:0", arg67_1: "bf16[5376, 8192][8192, 1]cuda:0", arg68_1: "bf16[5376][1]cuda:0", arg69_1: "bf16[5376][1]cuda:0", arg70_1: "bf16[21504, 5376][5376, 1]cuda:0", arg71_1: "bf16[21504, 5376][5376, 1]cuda:0", arg72_1: "bf16[5376, 21504][21504, 1]cuda:0", arg73_1: "bf16[5376][1]cuda:0", arg74_1: "bf16[1][1]cuda:0", arg75_1: "bf16[5376][1]cuda:0", arg76_1: "bf16[16384, 5376][5376, 1]cuda:0", arg77_1: "bf16[512][1]cuda:0", arg78_1: "bf16[2048, 5376][5376, 1]cuda:0", arg79_1: "bf16[512][1]cuda:0", arg80_1: "bf16[5376, 16384][16384, 1]cuda:0", arg81_1: "bf16[5376][1]cuda:0", arg82_1: "bf16[5376][1]cuda:0", arg83_1: "bf16[21504, 5376][5376, 1]cuda:0", arg84_1: "bf16[21504, 5376][5376, 1]cuda:0", arg85_1: "bf16[5376, 21504][21504, 1]cuda:0", arg86_1: "bf16[5376][1]cuda:0", arg87_1: "bf16[1][1]cuda:0", arg88_1: "bf16[5376][1]cuda:0"):
        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:76 in forward, code: embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None
        embedding: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg1_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:77 in forward, code: mul = torch.ops.aten.mul.Tensor(embedding, arg2_1);  embedding = arg2_1 = None
        mul: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, arg2_1);  embedding = arg2_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:163 in forward, code: convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul, torch.float32)
        convert_element_type_6: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:164 in forward, code: pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
        pow_1: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:165 in forward, code: mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        mean: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:166 in forward, code: add_5 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        add: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:167 in forward, code: pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_5, -0.5);  add_5 = None
        pow_2: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add, -0.5);  add = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:168 in forward, code: mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_6, pow_2);  convert_element_type_6 = pow_2 = None
        mul_3: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, pow_2);  convert_element_type_6 = pow_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:169 in forward, code: convert_element_type_7 = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        convert_element_type_7: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:170 in forward, code: mul_8 = torch.ops.aten.mul.Tensor(mul_7, convert_element_type_7);  mul_7 = convert_element_type_7 = None
        mul_4: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, convert_element_type_7);  mul_3 = convert_element_type_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:171 in forward, code: convert_element_type_8 = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        convert_element_type_8: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:173 in forward, code: view_2 = torch.ops.aten.view.default(convert_element_type_8, [2048, 5376])
        view_2: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_8, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:172 in forward, code: permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        permute: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:174 in forward, code: mm = torch.ops.aten.mm.default(view_2, permute);  view_2 = permute = None
        mm: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_2, permute);  view_2 = permute = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:175 in forward, code: view_3 = torch.ops.aten.view.default(mm, [1, 2048, 8192]);  mm = None
        view_3: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [1, 2048, 8192]);  mm = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:176 in forward, code: view_4 = torch.ops.aten.view.default(view_3, [1, 2048, -1, 256]);  view_3 = None
        view_4: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [1, 2048, -1, 256]);  view_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:177 in forward, code: convert_element_type_11 = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        convert_element_type_11: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:178 in forward, code: pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 2)
        pow_3: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:179 in forward, code: mean_1 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        mean_1: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:180 in forward, code: add_6 = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        add_1: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:181 in forward, code: pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_6, -0.5);  add_6 = None
        pow_4: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_1, -0.5);  add_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:182 in forward, code: mul_9 = torch.ops.aten.mul.Tensor(convert_element_type_11, pow_4);  convert_element_type_11 = pow_4 = None
        mul_5: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, pow_4);  convert_element_type_11 = pow_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:183 in forward, code: convert_element_type_12 = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        convert_element_type_12: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:184 in forward, code: mul_10 = torch.ops.aten.mul.Tensor(mul_9, convert_element_type_12);  mul_9 = convert_element_type_12 = None
        mul_6: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, convert_element_type_12);  mul_5 = convert_element_type_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:185 in forward, code: convert_element_type_13 = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_13: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:78 in forward, code: iota = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:79 in forward, code: unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        unsqueeze: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:141 in forward, code: unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        unsqueeze_19: "i64[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:142 in forward, code: convert_element_type = torch.ops.prims.convert_element_type.default(unsqueeze_19, torch.float32);  unsqueeze_19 = None
        convert_element_type: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_19, torch.float32);  unsqueeze_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:143 in forward, code: mul_1 = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  convert_element_type = arg3_1 = None
        mul_1: "f32[1, 2048, 128][262144, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  convert_element_type = arg3_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:144 in forward, code: unsqueeze_20 = torch.ops.aten.unsqueeze.default(mul_1, 2);  mul_1 = None
        unsqueeze_20: "f32[1, 2048, 1, 128][262144, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1, 2);  mul_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:145 in forward, code: expand_2 = torch.ops.aten.expand.default(unsqueeze_20, [1, 2048, 2, 128]);  unsqueeze_20 = None
        expand_2: "f32[1, 2048, 2, 128][262144, 128, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_20, [1, 2048, 2, 128]);  unsqueeze_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:146 in forward, code: clone = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        clone: "f32[1, 2048, 2, 128][524288, 256, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:147 in forward, code: view = torch.ops.aten.view.default(clone, [1, 2048, 256]);  clone = None
        view: "f32[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1, 2048, 256]);  clone = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:148 in forward, code: cos = torch.ops.aten.cos.default(view)
        cos: "f32[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.cos.default(view)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:150 in forward, code: convert_element_type_1 = torch.ops.prims.convert_element_type.default(cos, torch.bfloat16);  cos = None
        convert_element_type_1: "bf16[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cos, torch.bfloat16);  cos = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:186 in forward, code: unsqueeze_23 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_23: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:188 in forward, code: mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_13, unsqueeze_23);  unsqueeze_23 = None
        mul_7: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, unsqueeze_23);  unsqueeze_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:190 in forward, code: slice_8 = torch.ops.aten.slice.Tensor(convert_element_type_13, 3, 128, 9223372036854775807);  convert_element_type_13 = None
        slice_8: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_13, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:191 in forward, code: neg = torch.ops.aten.neg.default(slice_8);  slice_8 = None
        neg: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_8);  slice_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:189 in forward, code: slice_7 = torch.ops.aten.slice.Tensor(convert_element_type_13, 3, 0, 128)
        slice_7: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_13, 3, 0, 128);  convert_element_type_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:192 in forward, code: cat_2 = torch.ops.aten.cat.default([neg, slice_7], -1);  neg = slice_7 = None
        cat_2: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg, slice_7], -1);  neg = slice_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:149 in forward, code: sin = torch.ops.aten.sin.default(view);  view = None
        sin: "f32[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.sin.default(view);  view = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:151 in forward, code: convert_element_type_2 = torch.ops.prims.convert_element_type.default(sin, torch.bfloat16);  sin = None
        convert_element_type_2: "bf16[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sin, torch.bfloat16);  sin = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:187 in forward, code: unsqueeze_24 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_24: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:193 in forward, code: mul_12 = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_24);  cat_2 = unsqueeze_24 = None
        mul_8: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_24);  cat_2 = unsqueeze_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:194 in forward, code: add_7 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        add_2: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:195 in forward, code: permute_1 = torch.ops.aten.permute.default(add_7, [0, 2, 1, 3]);  add_7 = None
        permute_1: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_2, [0, 2, 1, 3]);  add_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:244 in forward, code: convert_element_type_23 = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        convert_element_type_23: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:247 in forward, code: mul_18 = torch.ops.aten.mul.Scalar(convert_element_type_23, 1.0);  convert_element_type_23 = None
        mul_14: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(convert_element_type_23, 1.0);  convert_element_type_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:250 in forward, code: expand_6 = torch.ops.aten.expand.default(mul_18, [1, 32, 2048, 256]);  mul_18 = None
        expand_6: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.expand.default(mul_14, [1, 32, 2048, 256]);  mul_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:251 in forward, code: view_13 = torch.ops.aten.view.default(expand_6, [32, 2048, 256]);  expand_6 = None
        view_13: "f32[32, 2048, 256][256, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(expand_6, [32, 2048, 256]);  expand_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:197 in forward, code: view_5 = torch.ops.aten.view.default(convert_element_type_8, [2048, 5376])
        view_5: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_8, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:196 in forward, code: permute_2 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        permute_2: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:198 in forward, code: mm_1 = torch.ops.aten.mm.default(view_5, permute_2);  view_5 = permute_2 = None
        mm_1: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_5, permute_2);  view_5 = permute_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:199 in forward, code: view_6 = torch.ops.aten.view.default(mm_1, [1, 2048, 4096]);  mm_1 = None
        view_6: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [1, 2048, 4096]);  mm_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:200 in forward, code: view_7 = torch.ops.aten.view.default(view_6, [1, 2048, -1, 256]);  view_6 = None
        view_7: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [1, 2048, -1, 256]);  view_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:206 in forward, code: convert_element_type_18 = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        convert_element_type_18: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:207 in forward, code: pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_18, 2)
        pow_5: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_18, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:208 in forward, code: mean_2 = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        mean_2: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:209 in forward, code: add_8 = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        add_3: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:210 in forward, code: pow_6 = torch.ops.aten.pow.Tensor_Scalar(add_8, -0.5);  add_8 = None
        pow_6: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_3, -0.5);  add_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:211 in forward, code: mul_13 = torch.ops.aten.mul.Tensor(convert_element_type_18, pow_6);  convert_element_type_18 = pow_6 = None
        mul_9: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, pow_6);  convert_element_type_18 = pow_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:212 in forward, code: convert_element_type_19 = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None
        convert_element_type_19: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:213 in forward, code: mul_14 = torch.ops.aten.mul.Tensor(mul_13, convert_element_type_19);  mul_13 = convert_element_type_19 = None
        mul_10: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, convert_element_type_19);  mul_9 = convert_element_type_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:214 in forward, code: convert_element_type_20 = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        convert_element_type_20: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:215 in forward, code: unsqueeze_25 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_25: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:217 in forward, code: mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_20, unsqueeze_25);  unsqueeze_25 = None
        mul_11: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, unsqueeze_25);  unsqueeze_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:219 in forward, code: slice_10 = torch.ops.aten.slice.Tensor(convert_element_type_20, 3, 128, 9223372036854775807);  convert_element_type_20 = None
        slice_10: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_20, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:220 in forward, code: neg_1 = torch.ops.aten.neg.default(slice_10);  slice_10 = None
        neg_1: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_10);  slice_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:218 in forward, code: slice_9 = torch.ops.aten.slice.Tensor(convert_element_type_20, 3, 0, 128)
        slice_9: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_20, 3, 0, 128);  convert_element_type_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:221 in forward, code: cat_3 = torch.ops.aten.cat.default([neg_1, slice_9], -1);  neg_1 = slice_9 = None
        cat_3: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_1, slice_9], -1);  neg_1 = slice_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:216 in forward, code: unsqueeze_26 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_26: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:222 in forward, code: mul_16 = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_26);  cat_3 = unsqueeze_26 = None
        mul_12: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_26);  cat_3 = unsqueeze_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:223 in forward, code: add_9 = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        add_4: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:224 in forward, code: permute_4 = torch.ops.aten.permute.default(add_9, [0, 2, 1, 3]);  add_9 = None
        permute_4: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_4, [0, 2, 1, 3]);  add_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:233 in forward, code: unsqueeze_27 = torch.ops.aten.unsqueeze.default(permute_4, 2);  permute_4 = None
        unsqueeze_27: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_4, 2);  permute_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:234 in forward, code: expand_4 = torch.ops.aten.expand.default(unsqueeze_27, [1, 16, 2, 2048, 256]);  unsqueeze_27 = None
        expand_4: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_27, [1, 16, 2, 2048, 256]);  unsqueeze_27 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:235 in forward, code: clone_4 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        clone_2: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:236 in forward, code: view_11 = torch.ops.aten.view.default(clone_4, [1, 32, 2048, 256]);  clone_4 = None
        view_11: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 32, 2048, 256]);  clone_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:245 in forward, code: convert_element_type_24 = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        convert_element_type_24: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:248 in forward, code: permute_6 = torch.ops.aten.permute.default(convert_element_type_24, [0, 1, 3, 2]);  convert_element_type_24 = None
        permute_6: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [0, 1, 3, 2]);  convert_element_type_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:249 in forward, code: mul_19 = torch.ops.aten.mul.Scalar(permute_6, 1.0);  permute_6 = None
        mul_15: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_6, 1.0);  permute_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:252 in forward, code: expand_7 = torch.ops.aten.expand.default(mul_19, [1, 32, 256, 2048]);  mul_19 = None
        expand_7: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_15, [1, 32, 256, 2048]);  mul_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:253 in forward, code: view_14 = torch.ops.aten.view.default(expand_7, [32, 256, 2048]);  expand_7 = None
        view_14: "f32[32, 256, 2048][524288, 1, 256]cuda:0" = torch.ops.aten.reshape.default(expand_7, [32, 256, 2048]);  expand_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:254 in forward, code: bmm = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
        bmm: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:255 in forward, code: view_15 = torch.ops.aten.view.default(bmm, [1, 32, 2048, 2048]);  bmm = None
        view_15: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [1, 32, 2048, 2048]);  bmm = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:128 in forward, code: full_default_1 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:129 in forward, code: full_default_2 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:118 in forward, code: iota_8 = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_6: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:125 in forward, code: unsqueeze_16 = torch.ops.aten.unsqueeze.default(iota_8, 0);  iota_8 = None
        unsqueeze_16: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:126 in forward, code: unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        unsqueeze_17: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:127 in forward, code: unsqueeze_18 = torch.ops.aten.unsqueeze.default(unsqueeze_17, 2);  unsqueeze_17 = None
        unsqueeze_18: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 2);  unsqueeze_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:117 in forward, code: iota_7 = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_5: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:122 in forward, code: unsqueeze_13 = torch.ops.aten.unsqueeze.default(iota_7, 0);  iota_7 = None
        unsqueeze_13: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_5, 0);  iota_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:123 in forward, code: unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None
        unsqueeze_14: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:124 in forward, code: unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        unsqueeze_15: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:130 in forward, code: sub_4 = torch.ops.aten.sub.Tensor(unsqueeze_15, 1024)
        sub_4: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_15, 1024)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:131 in forward, code: gt = torch.ops.aten.gt.Tensor(unsqueeze_18, sub_4);  sub_4 = None
        gt: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.gt.Tensor(unsqueeze_18, sub_4);  sub_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:132 in forward, code: bitwise_and_2 = torch.ops.aten.bitwise_and.Tensor(full_default_2, gt);  full_default_2 = gt = None
        bitwise_and_2: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full_default_2, gt);  full_default_2 = gt = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:133 in forward, code: le_1 = torch.ops.aten.le.Tensor(unsqueeze_18, unsqueeze_15)
        le_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_18, unsqueeze_15)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:134 in forward, code: bitwise_and_3 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, le_1);  bitwise_and_2 = le_1 = None
        bitwise_and_3: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, le_1);  bitwise_and_2 = le_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:135 in forward, code: bitwise_and_4 = torch.ops.aten.bitwise_and.Tensor(full_default_1, bitwise_and_3);  full_default_1 = bitwise_and_3 = None
        bitwise_and_4: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full_default_1, bitwise_and_3);  full_default_1 = bitwise_and_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:108 in forward, code: slice_4 = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        slice_4: "i64[1, 1][2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:109 in forward, code: sub_2 = torch.ops.aten.sub.Tensor(slice_4, 1);  slice_4 = None
        sub_2: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_4, 1);  slice_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:110 in forward, code: cat_1 = torch.ops.aten.cat.default([sub_2, unsqueeze], -1);  sub_2 = None
        cat_1: "i64[1, 2049][2049, 1]cuda:0" = torch.ops.aten.cat.default([sub_2, unsqueeze], -1);  sub_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:112 in forward, code: slice_6 = torch.ops.aten.slice.Tensor(cat_1, -1, 1, 2049);  cat_1 = None
        slice_6: "i64[1, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat_1, -1, 1, 2049)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:111 in forward, code: slice_5 = torch.ops.aten.slice.Tensor(cat_1, -1, 0, 2048)
        slice_5: "i64[1, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat_1, -1, 0, 2048);  cat_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:113 in forward, code: sub_3 = torch.ops.aten.sub.Tensor(slice_6, slice_5);  slice_6 = slice_5 = None
        sub_3: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_6, slice_5);  slice_6 = slice_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:114 in forward, code: ne_1 = torch.ops.aten.ne.Scalar(sub_3, 1);  sub_3 = None
        ne_1: "b8[1, 2048][2048, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_3, 1);  sub_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:115 in forward, code: cumsum_1 = torch.ops.aten.cumsum.default(ne_1, -1);  ne_1 = None
        cumsum_1: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.cumsum.default(ne_1, -1);  ne_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:116 in forward, code: iota_5 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_4: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:119 in forward, code: unsqueeze_10 = torch.ops.aten.unsqueeze.default(iota_5, 1);  iota_5 = None
        unsqueeze_10: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 1);  iota_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:120 in forward, code: unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_11: "i64[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:121 in forward, code: unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        unsqueeze_12: "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:136 in forward, code: index_2 = torch.ops.aten.index.Tensor(cumsum_1, [unsqueeze_12, unsqueeze_15]);  unsqueeze_15 = None
        index_2: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum_1, [unsqueeze_12, unsqueeze_15]);  unsqueeze_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:137 in forward, code: index_3 = torch.ops.aten.index.Tensor(cumsum_1, [unsqueeze_12, unsqueeze_18]);  cumsum_1 = unsqueeze_12 = unsqueeze_18 = None
        index_3: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum_1, [unsqueeze_12, unsqueeze_18]);  cumsum_1 = unsqueeze_12 = unsqueeze_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:138 in forward, code: eq_1 = torch.ops.aten.eq.Tensor(index_2, index_3);  index_2 = index_3 = None
        eq_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Tensor(index_2, index_3);  index_2 = index_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:139 in forward, code: bitwise_and_5 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, eq_1);  bitwise_and_4 = eq_1 = None
        bitwise_and_5: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, eq_1);  bitwise_and_4 = eq_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:140 in forward, code: expand_1 = torch.ops.aten.expand.default(bitwise_and_5, [1, -1, 2048, 2048]);  bitwise_and_5 = None
        expand_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_5, [1, -1, 2048, 2048]);  bitwise_and_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:242 in forward, code: full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:241 in forward, code: full_default_3 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:243 in forward, code: where = torch.ops.aten.where.self(expand_1, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        where: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_4, full_default_3);  full_default_4 = full_default_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:256 in forward, code: add_11 = torch.ops.aten.add.Tensor(view_15, where);  view_15 = where = None
        add_6: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_15, where);  view_15 = where = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:262 in forward, code: eq_2 = torch.ops.aten.eq.Scalar(add_11, -inf);  add_11 = None
        eq_2: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Scalar(add_6, -inf)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:263 in forward, code: logical_not = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        logical_not: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:264 in forward, code: any_1 = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        any_1: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:265 in forward, code: logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        logical_not_1: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_1);  any_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:266 in forward, code: full_default_5 = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:257 in forward, code: amax = torch.ops.aten.amax.default(add_11, [-1], True)
        amax: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_6, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:258 in forward, code: sub_5 = torch.ops.aten.sub.Tensor(add_11, amax);  amax = None
        sub_5: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:259 in forward, code: exp = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        exp: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:260 in forward, code: sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        sum_1: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:261 in forward, code: div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        div: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:267 in forward, code: where_1 = torch.ops.aten.where.self(logical_not_1, full_default_5, div);  logical_not_1 = full_default_5 = div = None
        where_1: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_5, div);  logical_not_1 = full_default_5 = div = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:268 in forward, code: expand_8 = torch.ops.aten.expand.default(where_1, [1, 32, 2048, 2048]);  where_1 = None
        expand_8: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_1, [1, 32, 2048, 2048]);  where_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:269 in forward, code: view_16 = torch.ops.aten.view.default(expand_8, [32, 2048, 2048]);  expand_8 = None
        view_16: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(expand_8, [32, 2048, 2048]);  expand_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:202 in forward, code: view_8 = torch.ops.aten.view.default(convert_element_type_8, [2048, 5376]);  convert_element_type_8 = None
        view_8: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_8, [2048, 5376]);  convert_element_type_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:201 in forward, code: permute_3 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        permute_3: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:203 in forward, code: mm_2 = torch.ops.aten.mm.default(view_8, permute_3);  view_8 = permute_3 = None
        mm_2: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_8, permute_3);  view_8 = permute_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:204 in forward, code: view_9 = torch.ops.aten.view.default(mm_2, [1, 2048, 4096]);  mm_2 = None
        view_9: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [1, 2048, 4096]);  mm_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:205 in forward, code: view_10 = torch.ops.aten.view.default(view_9, [1, 2048, -1, 256]);  view_9 = None
        view_10: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [1, 2048, -1, 256]);  view_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:225 in forward, code: convert_element_type_21 = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_21: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:226 in forward, code: pow_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2)
        pow_7: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:227 in forward, code: mean_3 = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        mean_3: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:228 in forward, code: add_10 = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        add_5: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:229 in forward, code: pow_8 = torch.ops.aten.pow.Tensor_Scalar(add_10, -0.5);  add_10 = None
        pow_8: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_5, -0.5);  add_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:230 in forward, code: mul_17 = torch.ops.aten.mul.Tensor(convert_element_type_21, pow_8);  convert_element_type_21 = pow_8 = None
        mul_13: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, pow_8);  convert_element_type_21 = pow_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:231 in forward, code: convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        convert_element_type_22: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:232 in forward, code: permute_5 = torch.ops.aten.permute.default(convert_element_type_22, [0, 2, 1, 3]);  convert_element_type_22 = None
        permute_5: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_22, [0, 2, 1, 3]);  convert_element_type_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:237 in forward, code: unsqueeze_28 = torch.ops.aten.unsqueeze.default(permute_5, 2);  permute_5 = None
        unsqueeze_28: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_5, 2);  permute_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:238 in forward, code: expand_5 = torch.ops.aten.expand.default(unsqueeze_28, [1, 16, 2, 2048, 256]);  unsqueeze_28 = None
        expand_5: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_28, [1, 16, 2, 2048, 256]);  unsqueeze_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:239 in forward, code: clone_5 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        clone_3: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:240 in forward, code: view_12 = torch.ops.aten.view.default(clone_5, [1, 32, 2048, 256]);  clone_5 = None
        view_12: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [1, 32, 2048, 256]);  clone_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:246 in forward, code: convert_element_type_25 = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        convert_element_type_25: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:270 in forward, code: expand_9 = torch.ops.aten.expand.default(convert_element_type_25, [1, 32, 2048, 256]);  convert_element_type_25 = None
        expand_9: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_25, [1, 32, 2048, 256]);  convert_element_type_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:271 in forward, code: view_17 = torch.ops.aten.view.default(expand_9, [32, 2048, 256]);  expand_9 = None
        view_17: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(expand_9, [32, 2048, 256]);  expand_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:272 in forward, code: bmm_1 = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = view_17 = None
        bmm_1: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = view_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:273 in forward, code: view_18 = torch.ops.aten.view.default(bmm_1, [1, 32, 2048, 256]);  bmm_1 = None
        view_18: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_1, [1, 32, 2048, 256]);  bmm_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:274 in forward, code: convert_element_type_27 = torch.ops.prims.convert_element_type.default(view_18, torch.bfloat16);  view_18 = None
        convert_element_type_26: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_18, torch.bfloat16);  view_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:275 in forward, code: permute_7 = torch.ops.aten.permute.default(convert_element_type_27, [0, 2, 1, 3]);  convert_element_type_27 = None
        permute_7: "bf16[1, 2048, 32, 256][16777216, 256, 524288, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_26, [0, 2, 1, 3]);  convert_element_type_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:276 in forward, code: clone_6 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        clone_4: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:277 in forward, code: view_19 = torch.ops.aten.view.default(clone_6, [1, 2048, -1]);  clone_6 = None
        view_19: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 2048, -1]);  clone_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:279 in forward, code: view_20 = torch.ops.aten.view.default(view_19, [2048, 8192]);  view_19 = None
        view_20: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_19, [2048, 8192]);  view_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:278 in forward, code: permute_8 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        permute_8: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:280 in forward, code: mm_3 = torch.ops.aten.mm.default(view_20, permute_8);  view_20 = permute_8 = None
        mm_3: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_20, permute_8);  view_20 = permute_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:281 in forward, code: view_21 = torch.ops.aten.view.default(mm_3, [1, 2048, 5376]);  mm_3 = None
        view_21: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [1, 2048, 5376]);  mm_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:282 in forward, code: convert_element_type_30 = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        convert_element_type_29: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:283 in forward, code: pow_9 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_30, 2)
        pow_9: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_29, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:284 in forward, code: mean_4 = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        mean_4: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:285 in forward, code: add_12 = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        add_7: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:286 in forward, code: pow_10 = torch.ops.aten.pow.Tensor_Scalar(add_12, -0.5);  add_12 = None
        pow_10: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_7, -0.5);  add_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:287 in forward, code: mul_20 = torch.ops.aten.mul.Tensor(convert_element_type_30, pow_10);  convert_element_type_30 = pow_10 = None
        mul_16: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, pow_10);  convert_element_type_29 = pow_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:288 in forward, code: convert_element_type_31 = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        convert_element_type_30: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:289 in forward, code: mul_21 = torch.ops.aten.mul.Tensor(mul_20, convert_element_type_31);  mul_20 = convert_element_type_31 = None
        mul_17: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, convert_element_type_30);  mul_16 = convert_element_type_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:290 in forward, code: convert_element_type_32 = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        convert_element_type_31: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:291 in forward, code: add_13 = torch.ops.aten.add.Tensor(mul, convert_element_type_32);  mul = convert_element_type_32 = None
        add_8: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, convert_element_type_31);  mul = convert_element_type_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:292 in forward, code: convert_element_type_33 = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        convert_element_type_32: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:293 in forward, code: pow_11 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_33, 2)
        pow_11: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_32, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:294 in forward, code: mean_5 = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        mean_5: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:295 in forward, code: add_14 = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        add_9: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:296 in forward, code: pow_12 = torch.ops.aten.pow.Tensor_Scalar(add_14, -0.5);  add_14 = None
        pow_12: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_9, -0.5);  add_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:297 in forward, code: mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_33, pow_12);  convert_element_type_33 = pow_12 = None
        mul_18: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_32, pow_12);  convert_element_type_32 = pow_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:298 in forward, code: convert_element_type_34 = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        convert_element_type_33: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:299 in forward, code: mul_23 = torch.ops.aten.mul.Tensor(mul_22, convert_element_type_34);  mul_22 = convert_element_type_34 = None
        mul_19: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, convert_element_type_33);  mul_18 = convert_element_type_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:300 in forward, code: convert_element_type_35 = torch.ops.prims.convert_element_type.default(mul_23, torch.bfloat16);  mul_23 = None
        convert_element_type_34: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:302 in forward, code: view_22 = torch.ops.aten.view.default(convert_element_type_35, [2048, 5376])
        view_22: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:301 in forward, code: permute_9 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        permute_9: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:303 in forward, code: mm_4 = torch.ops.aten.mm.default(view_22, permute_9);  view_22 = permute_9 = None
        mm_4: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_9);  view_22 = permute_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:304 in forward, code: view_23 = torch.ops.aten.view.default(mm_4, [1, 2048, 21504]);  mm_4 = None
        view_23: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [1, 2048, 21504]);  mm_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:305 in forward, code: convert_element_type_38 = torch.ops.prims.convert_element_type.default(view_23, torch.float32);  view_23 = None
        convert_element_type_37: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.float32);  view_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:311 in forward, code: mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5);  convert_element_type_38 = None
        mul_24: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, 0.5)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:306 in forward, code: mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_38, convert_element_type_38)
        mul_20: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, convert_element_type_37)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:307 in forward, code: mul_25 = torch.ops.aten.mul.Tensor(mul_24, convert_element_type_38);  mul_24 = None
        mul_21: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, convert_element_type_37);  mul_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:308 in forward, code: mul_26 = torch.ops.aten.mul.Tensor(mul_25, 0.044715);  mul_25 = None
        mul_22: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, 0.044715);  mul_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:309 in forward, code: add_15 = torch.ops.aten.add.Tensor(convert_element_type_38, mul_26);  mul_26 = None
        add_10: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_37, mul_22);  convert_element_type_37 = mul_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:310 in forward, code: mul_27 = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        mul_23: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:312 in forward, code: tanh = torch.ops.aten.tanh.default(mul_27);  mul_27 = None
        tanh: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_23);  mul_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:313 in forward, code: add_16 = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        add_11: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:314 in forward, code: mul_29 = torch.ops.aten.mul.Tensor(mul_28, add_16);  mul_28 = add_16 = None
        mul_25: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_11);  mul_24 = add_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:315 in forward, code: convert_element_type_39 = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        convert_element_type_38: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:317 in forward, code: view_24 = torch.ops.aten.view.default(convert_element_type_35, [2048, 5376]);  convert_element_type_35 = None
        view_24: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [2048, 5376]);  convert_element_type_34 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:316 in forward, code: permute_10 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        permute_10: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:318 in forward, code: mm_5 = torch.ops.aten.mm.default(view_24, permute_10);  view_24 = permute_10 = None
        mm_5: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_24, permute_10);  view_24 = permute_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:319 in forward, code: view_25 = torch.ops.aten.view.default(mm_5, [1, 2048, 21504]);  mm_5 = None
        view_25: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [1, 2048, 21504]);  mm_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:320 in forward, code: mul_30 = torch.ops.aten.mul.Tensor(convert_element_type_39, view_25);  convert_element_type_39 = view_25 = None
        mul_26: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_38, view_25);  convert_element_type_38 = view_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:322 in forward, code: view_26 = torch.ops.aten.view.default(mul_30, [2048, 21504]);  mul_30 = None
        view_26: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_26, [2048, 21504]);  mul_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:321 in forward, code: permute_11 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        permute_11: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:323 in forward, code: mm_6 = torch.ops.aten.mm.default(view_26, permute_11);  view_26 = permute_11 = None
        mm_6: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_26, permute_11);  view_26 = permute_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:324 in forward, code: view_27 = torch.ops.aten.view.default(mm_6, [1, 2048, 5376]);  mm_6 = None
        view_27: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [1, 2048, 5376]);  mm_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:325 in forward, code: convert_element_type_44 = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        convert_element_type_43: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:326 in forward, code: pow_13 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_44, 2)
        pow_13: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_43, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:327 in forward, code: mean_6 = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        mean_6: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:328 in forward, code: add_17 = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        add_12: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:329 in forward, code: pow_14 = torch.ops.aten.pow.Tensor_Scalar(add_17, -0.5);  add_17 = None
        pow_14: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_12, -0.5);  add_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:330 in forward, code: mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_44, pow_14);  convert_element_type_44 = pow_14 = None
        mul_27: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, pow_14);  convert_element_type_43 = pow_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:331 in forward, code: convert_element_type_45 = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        convert_element_type_44: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:332 in forward, code: mul_32 = torch.ops.aten.mul.Tensor(mul_31, convert_element_type_45);  mul_31 = convert_element_type_45 = None
        mul_28: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, convert_element_type_44);  mul_27 = convert_element_type_44 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:333 in forward, code: convert_element_type_46 = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        convert_element_type_45: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:334 in forward, code: add_18 = torch.ops.aten.add.Tensor(add_13, convert_element_type_46);  add_13 = convert_element_type_46 = None
        add_13: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_8, convert_element_type_45);  add_8 = convert_element_type_45 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:335 in forward, code: mul_33 = torch.ops.aten.mul.Tensor(add_18, arg18_1);  add_18 = arg18_1 = None
        mul_29: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_13, arg18_1);  add_13 = arg18_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:336 in forward, code: convert_element_type_47 = torch.ops.prims.convert_element_type.default(mul_33, torch.float32)
        convert_element_type_46: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:337 in forward, code: pow_15 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 2)
        pow_15: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_46, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:338 in forward, code: mean_7 = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        mean_7: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:339 in forward, code: add_19 = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        add_14: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:340 in forward, code: pow_16 = torch.ops.aten.pow.Tensor_Scalar(add_19, -0.5);  add_19 = None
        pow_16: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_14, -0.5);  add_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:341 in forward, code: mul_34 = torch.ops.aten.mul.Tensor(convert_element_type_47, pow_16);  convert_element_type_47 = pow_16 = None
        mul_30: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_46, pow_16);  convert_element_type_46 = pow_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:342 in forward, code: convert_element_type_48 = torch.ops.prims.convert_element_type.default(arg19_1, torch.float32);  arg19_1 = None
        convert_element_type_47: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg19_1, torch.float32);  arg19_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:343 in forward, code: mul_35 = torch.ops.aten.mul.Tensor(mul_34, convert_element_type_48);  mul_34 = convert_element_type_48 = None
        mul_31: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, convert_element_type_47);  mul_30 = convert_element_type_47 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:344 in forward, code: convert_element_type_49 = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        convert_element_type_48: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:346 in forward, code: view_28 = torch.ops.aten.view.default(convert_element_type_49, [2048, 5376])
        view_28: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_48, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:345 in forward, code: permute_12 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        permute_12: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:347 in forward, code: mm_7 = torch.ops.aten.mm.default(view_28, permute_12);  view_28 = permute_12 = None
        mm_7: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_12);  view_28 = permute_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:348 in forward, code: view_29 = torch.ops.aten.view.default(mm_7, [1, 2048, 8192]);  mm_7 = None
        view_29: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [1, 2048, 8192]);  mm_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:349 in forward, code: view_30 = torch.ops.aten.view.default(view_29, [1, 2048, -1, 256]);  view_29 = None
        view_30: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [1, 2048, -1, 256]);  view_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:350 in forward, code: convert_element_type_52 = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        convert_element_type_51: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:351 in forward, code: pow_17 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_52, 2)
        pow_17: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_51, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:352 in forward, code: mean_8 = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        mean_8: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:353 in forward, code: add_20 = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        add_15: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:354 in forward, code: pow_18 = torch.ops.aten.pow.Tensor_Scalar(add_20, -0.5);  add_20 = None
        pow_18: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_15, -0.5);  add_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:355 in forward, code: mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_52, pow_18);  convert_element_type_52 = pow_18 = None
        mul_32: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, pow_18);  convert_element_type_51 = pow_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:356 in forward, code: convert_element_type_53 = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        convert_element_type_52: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:357 in forward, code: mul_37 = torch.ops.aten.mul.Tensor(mul_36, convert_element_type_53);  mul_36 = convert_element_type_53 = None
        mul_33: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, convert_element_type_52);  mul_32 = convert_element_type_52 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:358 in forward, code: convert_element_type_54 = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        convert_element_type_53: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:359 in forward, code: unsqueeze_29 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_29: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:361 in forward, code: mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_54, unsqueeze_29);  unsqueeze_29 = None
        mul_34: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, unsqueeze_29);  unsqueeze_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:363 in forward, code: slice_12 = torch.ops.aten.slice.Tensor(convert_element_type_54, 3, 128, 9223372036854775807);  convert_element_type_54 = None
        slice_12: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_53, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:364 in forward, code: neg_2 = torch.ops.aten.neg.default(slice_12);  slice_12 = None
        neg_2: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_12);  slice_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:362 in forward, code: slice_11 = torch.ops.aten.slice.Tensor(convert_element_type_54, 3, 0, 128)
        slice_11: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_53, 3, 0, 128);  convert_element_type_53 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:365 in forward, code: cat_4 = torch.ops.aten.cat.default([neg_2, slice_11], -1);  neg_2 = slice_11 = None
        cat_4: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_2, slice_11], -1);  neg_2 = slice_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:360 in forward, code: unsqueeze_30 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_30: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:366 in forward, code: mul_39 = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_30);  cat_4 = unsqueeze_30 = None
        mul_35: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_30);  cat_4 = unsqueeze_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:367 in forward, code: add_21 = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        add_16: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:368 in forward, code: permute_13 = torch.ops.aten.permute.default(add_21, [0, 2, 1, 3]);  add_21 = None
        permute_13: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_16, [0, 2, 1, 3]);  add_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:417 in forward, code: convert_element_type_64 = torch.ops.prims.convert_element_type.default(permute_13, torch.float32);  permute_13 = None
        convert_element_type_63: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_13, torch.float32);  permute_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:420 in forward, code: mul_45 = torch.ops.aten.mul.Scalar(convert_element_type_64, 1.0);  convert_element_type_64 = None
        mul_41: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(convert_element_type_63, 1.0);  convert_element_type_63 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:423 in forward, code: expand_12 = torch.ops.aten.expand.default(mul_45, [1, 32, 2048, 256]);  mul_45 = None
        expand_12: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.expand.default(mul_41, [1, 32, 2048, 256]);  mul_41 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:424 in forward, code: view_39 = torch.ops.aten.view.default(expand_12, [32, 2048, 256]);  expand_12 = None
        view_39: "f32[32, 2048, 256][256, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(expand_12, [32, 2048, 256]);  expand_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:370 in forward, code: view_31 = torch.ops.aten.view.default(convert_element_type_49, [2048, 5376])
        view_31: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_48, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:369 in forward, code: permute_14 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        permute_14: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:371 in forward, code: mm_8 = torch.ops.aten.mm.default(view_31, permute_14);  view_31 = permute_14 = None
        mm_8: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_31, permute_14);  view_31 = permute_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:372 in forward, code: view_32 = torch.ops.aten.view.default(mm_8, [1, 2048, 4096]);  mm_8 = None
        view_32: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [1, 2048, 4096]);  mm_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:373 in forward, code: view_33 = torch.ops.aten.view.default(view_32, [1, 2048, -1, 256]);  view_32 = None
        view_33: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [1, 2048, -1, 256]);  view_32 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:379 in forward, code: convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        convert_element_type_58: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:380 in forward, code: pow_19 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_59, 2)
        pow_19: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_58, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:381 in forward, code: mean_9 = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        mean_9: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:382 in forward, code: add_22 = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        add_17: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:383 in forward, code: pow_20 = torch.ops.aten.pow.Tensor_Scalar(add_22, -0.5);  add_22 = None
        pow_20: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_17, -0.5);  add_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:384 in forward, code: mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_59, pow_20);  convert_element_type_59 = pow_20 = None
        mul_36: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, pow_20);  convert_element_type_58 = pow_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:385 in forward, code: convert_element_type_60 = torch.ops.prims.convert_element_type.default(arg24_1, torch.float32);  arg24_1 = None
        convert_element_type_59: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg24_1, torch.float32);  arg24_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:386 in forward, code: mul_41 = torch.ops.aten.mul.Tensor(mul_40, convert_element_type_60);  mul_40 = convert_element_type_60 = None
        mul_37: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, convert_element_type_59);  mul_36 = convert_element_type_59 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:387 in forward, code: convert_element_type_61 = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        convert_element_type_60: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:388 in forward, code: unsqueeze_31 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_31: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:390 in forward, code: mul_42 = torch.ops.aten.mul.Tensor(convert_element_type_61, unsqueeze_31);  unsqueeze_31 = None
        mul_38: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_60, unsqueeze_31);  unsqueeze_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:392 in forward, code: slice_14 = torch.ops.aten.slice.Tensor(convert_element_type_61, 3, 128, 9223372036854775807);  convert_element_type_61 = None
        slice_14: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_60, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:393 in forward, code: neg_3 = torch.ops.aten.neg.default(slice_14);  slice_14 = None
        neg_3: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_14);  slice_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:391 in forward, code: slice_13 = torch.ops.aten.slice.Tensor(convert_element_type_61, 3, 0, 128)
        slice_13: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_60, 3, 0, 128);  convert_element_type_60 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:394 in forward, code: cat_5 = torch.ops.aten.cat.default([neg_3, slice_13], -1);  neg_3 = slice_13 = None
        cat_5: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_3, slice_13], -1);  neg_3 = slice_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:389 in forward, code: unsqueeze_32 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_32: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:395 in forward, code: mul_43 = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_32);  cat_5 = unsqueeze_32 = None
        mul_39: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_32);  cat_5 = unsqueeze_32 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:396 in forward, code: add_23 = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
        add_18: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:397 in forward, code: permute_16 = torch.ops.aten.permute.default(add_23, [0, 2, 1, 3]);  add_23 = None
        permute_16: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_18, [0, 2, 1, 3]);  add_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:406 in forward, code: unsqueeze_33 = torch.ops.aten.unsqueeze.default(permute_16, 2);  permute_16 = None
        unsqueeze_33: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_16, 2);  permute_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:407 in forward, code: expand_10 = torch.ops.aten.expand.default(unsqueeze_33, [1, 16, 2, 2048, 256]);  unsqueeze_33 = None
        expand_10: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_33, [1, 16, 2, 2048, 256]);  unsqueeze_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:408 in forward, code: clone_7 = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        clone_5: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:409 in forward, code: view_37 = torch.ops.aten.view.default(clone_7, [1, 32, 2048, 256]);  clone_7 = None
        view_37: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [1, 32, 2048, 256]);  clone_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:418 in forward, code: convert_element_type_65 = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        convert_element_type_64: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:421 in forward, code: permute_18 = torch.ops.aten.permute.default(convert_element_type_65, [0, 1, 3, 2]);  convert_element_type_65 = None
        permute_18: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_64, [0, 1, 3, 2]);  convert_element_type_64 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:422 in forward, code: mul_46 = torch.ops.aten.mul.Scalar(permute_18, 1.0);  permute_18 = None
        mul_42: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_18, 1.0);  permute_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:425 in forward, code: expand_13 = torch.ops.aten.expand.default(mul_46, [1, 32, 256, 2048]);  mul_46 = None
        expand_13: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_42, [1, 32, 256, 2048]);  mul_42 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:426 in forward, code: view_40 = torch.ops.aten.view.default(expand_13, [32, 256, 2048]);  expand_13 = None
        view_40: "f32[32, 256, 2048][524288, 1, 256]cuda:0" = torch.ops.aten.reshape.default(expand_13, [32, 256, 2048]);  expand_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:427 in forward, code: bmm_2 = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = view_40 = None
        bmm_2: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = view_40 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:428 in forward, code: view_41 = torch.ops.aten.view.default(bmm_2, [1, 32, 2048, 2048]);  bmm_2 = None
        view_41: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [1, 32, 2048, 2048]);  bmm_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:415 in forward, code: full_default_7 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:414 in forward, code: full_default_6 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:416 in forward, code: where_2 = torch.ops.aten.where.self(expand_1, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        where_2: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_7, full_default_6);  full_default_7 = full_default_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:429 in forward, code: add_25 = torch.ops.aten.add.Tensor(view_41, where_2);  view_41 = where_2 = None
        add_20: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, where_2);  view_41 = where_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:435 in forward, code: eq_3 = torch.ops.aten.eq.Scalar(add_25, -inf);  add_25 = None
        eq_3: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Scalar(add_20, -inf)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:436 in forward, code: logical_not_2 = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        logical_not_2: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:437 in forward, code: any_2 = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        any_2: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:438 in forward, code: logical_not_3 = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        logical_not_3: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_2);  any_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:439 in forward, code: full_default_8 = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:430 in forward, code: amax_1 = torch.ops.aten.amax.default(add_25, [-1], True)
        amax_1: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_20, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:431 in forward, code: sub_6 = torch.ops.aten.sub.Tensor(add_25, amax_1);  amax_1 = None
        sub_6: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_20, amax_1);  add_20 = amax_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:432 in forward, code: exp_1 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        exp_1: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:433 in forward, code: sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        sum_2: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:434 in forward, code: div_1 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        div_1: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:440 in forward, code: where_3 = torch.ops.aten.where.self(logical_not_3, full_default_8, div_1);  logical_not_3 = full_default_8 = div_1 = None
        where_3: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(logical_not_3, full_default_8, div_1);  logical_not_3 = full_default_8 = div_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:441 in forward, code: expand_14 = torch.ops.aten.expand.default(where_3, [1, 32, 2048, 2048]);  where_3 = None
        expand_14: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_3, [1, 32, 2048, 2048]);  where_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:442 in forward, code: view_42 = torch.ops.aten.view.default(expand_14, [32, 2048, 2048]);  expand_14 = None
        view_42: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(expand_14, [32, 2048, 2048]);  expand_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:375 in forward, code: view_34 = torch.ops.aten.view.default(convert_element_type_49, [2048, 5376]);  convert_element_type_49 = None
        view_34: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_48, [2048, 5376]);  convert_element_type_48 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:374 in forward, code: permute_15 = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        permute_15: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:376 in forward, code: mm_9 = torch.ops.aten.mm.default(view_34, permute_15);  view_34 = permute_15 = None
        mm_9: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_34, permute_15);  view_34 = permute_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:377 in forward, code: view_35 = torch.ops.aten.view.default(mm_9, [1, 2048, 4096]);  mm_9 = None
        view_35: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [1, 2048, 4096]);  mm_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:378 in forward, code: view_36 = torch.ops.aten.view.default(view_35, [1, 2048, -1, 256]);  view_35 = None
        view_36: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_35, [1, 2048, -1, 256]);  view_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:398 in forward, code: convert_element_type_62 = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        convert_element_type_61: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:399 in forward, code: pow_21 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_62, 2)
        pow_21: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_61, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:400 in forward, code: mean_10 = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        mean_10: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:401 in forward, code: add_24 = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        add_19: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:402 in forward, code: pow_22 = torch.ops.aten.pow.Tensor_Scalar(add_24, -0.5);  add_24 = None
        pow_22: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_19, -0.5);  add_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:403 in forward, code: mul_44 = torch.ops.aten.mul.Tensor(convert_element_type_62, pow_22);  convert_element_type_62 = pow_22 = None
        mul_40: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, pow_22);  convert_element_type_61 = pow_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:404 in forward, code: convert_element_type_63 = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None
        convert_element_type_62: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:405 in forward, code: permute_17 = torch.ops.aten.permute.default(convert_element_type_63, [0, 2, 1, 3]);  convert_element_type_63 = None
        permute_17: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_62, [0, 2, 1, 3]);  convert_element_type_62 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:410 in forward, code: unsqueeze_34 = torch.ops.aten.unsqueeze.default(permute_17, 2);  permute_17 = None
        unsqueeze_34: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_17, 2);  permute_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:411 in forward, code: expand_11 = torch.ops.aten.expand.default(unsqueeze_34, [1, 16, 2, 2048, 256]);  unsqueeze_34 = None
        expand_11: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_34, [1, 16, 2, 2048, 256]);  unsqueeze_34 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:412 in forward, code: clone_8 = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        clone_6: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:413 in forward, code: view_38 = torch.ops.aten.view.default(clone_8, [1, 32, 2048, 256]);  clone_8 = None
        view_38: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [1, 32, 2048, 256]);  clone_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:419 in forward, code: convert_element_type_66 = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        convert_element_type_65: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:443 in forward, code: expand_15 = torch.ops.aten.expand.default(convert_element_type_66, [1, 32, 2048, 256]);  convert_element_type_66 = None
        expand_15: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_65, [1, 32, 2048, 256]);  convert_element_type_65 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:444 in forward, code: view_43 = torch.ops.aten.view.default(expand_15, [32, 2048, 256]);  expand_15 = None
        view_43: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(expand_15, [32, 2048, 256]);  expand_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:445 in forward, code: bmm_3 = torch.ops.aten.bmm.default(view_42, view_43);  view_42 = view_43 = None
        bmm_3: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_42, view_43);  view_42 = view_43 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:446 in forward, code: view_44 = torch.ops.aten.view.default(bmm_3, [1, 32, 2048, 256]);  bmm_3 = None
        view_44: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [1, 32, 2048, 256]);  bmm_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:447 in forward, code: convert_element_type_68 = torch.ops.prims.convert_element_type.default(view_44, torch.bfloat16);  view_44 = None
        convert_element_type_66: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_44, torch.bfloat16);  view_44 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:448 in forward, code: permute_19 = torch.ops.aten.permute.default(convert_element_type_68, [0, 2, 1, 3]);  convert_element_type_68 = None
        permute_19: "bf16[1, 2048, 32, 256][16777216, 256, 524288, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_66, [0, 2, 1, 3]);  convert_element_type_66 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:449 in forward, code: clone_9 = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        clone_7: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:450 in forward, code: view_45 = torch.ops.aten.view.default(clone_9, [1, 2048, -1]);  clone_9 = None
        view_45: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [1, 2048, -1]);  clone_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:452 in forward, code: view_46 = torch.ops.aten.view.default(view_45, [2048, 8192]);  view_45 = None
        view_46: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [2048, 8192]);  view_45 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:451 in forward, code: permute_20 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        permute_20: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:453 in forward, code: mm_10 = torch.ops.aten.mm.default(view_46, permute_20);  view_46 = permute_20 = None
        mm_10: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_46, permute_20);  view_46 = permute_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:454 in forward, code: view_47 = torch.ops.aten.view.default(mm_10, [1, 2048, 5376]);  mm_10 = None
        view_47: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [1, 2048, 5376]);  mm_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:455 in forward, code: convert_element_type_71 = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        convert_element_type_69: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:456 in forward, code: pow_23 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_71, 2)
        pow_23: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:457 in forward, code: mean_11 = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        mean_11: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:458 in forward, code: add_26 = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        add_21: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:459 in forward, code: pow_24 = torch.ops.aten.pow.Tensor_Scalar(add_26, -0.5);  add_26 = None
        pow_24: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_21, -0.5);  add_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:460 in forward, code: mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_71, pow_24);  convert_element_type_71 = pow_24 = None
        mul_43: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_69, pow_24);  convert_element_type_69 = pow_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:461 in forward, code: convert_element_type_72 = torch.ops.prims.convert_element_type.default(arg26_1, torch.float32);  arg26_1 = None
        convert_element_type_70: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg26_1, torch.float32);  arg26_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:462 in forward, code: mul_48 = torch.ops.aten.mul.Tensor(mul_47, convert_element_type_72);  mul_47 = convert_element_type_72 = None
        mul_44: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, convert_element_type_70);  mul_43 = convert_element_type_70 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:463 in forward, code: convert_element_type_73 = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        convert_element_type_71: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:464 in forward, code: add_27 = torch.ops.aten.add.Tensor(mul_33, convert_element_type_73);  mul_33 = convert_element_type_73 = None
        add_22: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, convert_element_type_71);  mul_29 = convert_element_type_71 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:465 in forward, code: convert_element_type_74 = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        convert_element_type_72: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:466 in forward, code: pow_25 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_74, 2)
        pow_25: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_72, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:467 in forward, code: mean_12 = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        mean_12: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:468 in forward, code: add_28 = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        add_23: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:469 in forward, code: pow_26 = torch.ops.aten.pow.Tensor_Scalar(add_28, -0.5);  add_28 = None
        pow_26: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_23, -0.5);  add_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:470 in forward, code: mul_49 = torch.ops.aten.mul.Tensor(convert_element_type_74, pow_26);  convert_element_type_74 = pow_26 = None
        mul_45: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_72, pow_26);  convert_element_type_72 = pow_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:471 in forward, code: convert_element_type_75 = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        convert_element_type_73: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:472 in forward, code: mul_50 = torch.ops.aten.mul.Tensor(mul_49, convert_element_type_75);  mul_49 = convert_element_type_75 = None
        mul_46: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, convert_element_type_73);  mul_45 = convert_element_type_73 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:473 in forward, code: convert_element_type_76 = torch.ops.prims.convert_element_type.default(mul_50, torch.bfloat16);  mul_50 = None
        convert_element_type_74: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:475 in forward, code: view_48 = torch.ops.aten.view.default(convert_element_type_76, [2048, 5376])
        view_48: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_74, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:474 in forward, code: permute_21 = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        permute_21: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:476 in forward, code: mm_11 = torch.ops.aten.mm.default(view_48, permute_21);  view_48 = permute_21 = None
        mm_11: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_48, permute_21);  view_48 = permute_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:477 in forward, code: view_49 = torch.ops.aten.view.default(mm_11, [1, 2048, 21504]);  mm_11 = None
        view_49: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [1, 2048, 21504]);  mm_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:478 in forward, code: convert_element_type_79 = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        convert_element_type_77: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:484 in forward, code: mul_55 = torch.ops.aten.mul.Tensor(convert_element_type_79, 0.5);  convert_element_type_79 = None
        mul_51: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, 0.5)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:479 in forward, code: mul_51 = torch.ops.aten.mul.Tensor(convert_element_type_79, convert_element_type_79)
        mul_47: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, convert_element_type_77)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:480 in forward, code: mul_52 = torch.ops.aten.mul.Tensor(mul_51, convert_element_type_79);  mul_51 = None
        mul_48: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, convert_element_type_77);  mul_47 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:481 in forward, code: mul_53 = torch.ops.aten.mul.Tensor(mul_52, 0.044715);  mul_52 = None
        mul_49: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 0.044715);  mul_48 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:482 in forward, code: add_29 = torch.ops.aten.add.Tensor(convert_element_type_79, mul_53);  mul_53 = None
        add_24: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_77, mul_49);  convert_element_type_77 = mul_49 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:483 in forward, code: mul_54 = torch.ops.aten.mul.Tensor(add_29, 0.7978845608028654);  add_29 = None
        mul_50: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_24, 0.7978845608028654);  add_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:485 in forward, code: tanh_1 = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        tanh_1: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_50);  mul_50 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:486 in forward, code: add_30 = torch.ops.aten.add.Tensor(tanh_1, 1);  tanh_1 = None
        add_25: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1);  tanh_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:487 in forward, code: mul_56 = torch.ops.aten.mul.Tensor(mul_55, add_30);  mul_55 = add_30 = None
        mul_52: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, add_25);  mul_51 = add_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:488 in forward, code: convert_element_type_80 = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None
        convert_element_type_78: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:490 in forward, code: view_50 = torch.ops.aten.view.default(convert_element_type_76, [2048, 5376]);  convert_element_type_76 = None
        view_50: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_74, [2048, 5376]);  convert_element_type_74 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:489 in forward, code: permute_22 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        permute_22: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:491 in forward, code: mm_12 = torch.ops.aten.mm.default(view_50, permute_22);  view_50 = permute_22 = None
        mm_12: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_50, permute_22);  view_50 = permute_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:492 in forward, code: view_51 = torch.ops.aten.view.default(mm_12, [1, 2048, 21504]);  mm_12 = None
        view_51: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [1, 2048, 21504]);  mm_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:493 in forward, code: mul_57 = torch.ops.aten.mul.Tensor(convert_element_type_80, view_51);  convert_element_type_80 = view_51 = None
        mul_53: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_78, view_51);  convert_element_type_78 = view_51 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:495 in forward, code: view_52 = torch.ops.aten.view.default(mul_57, [2048, 21504]);  mul_57 = None
        view_52: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_53, [2048, 21504]);  mul_53 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:494 in forward, code: permute_23 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        permute_23: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:496 in forward, code: mm_13 = torch.ops.aten.mm.default(view_52, permute_23);  view_52 = permute_23 = None
        mm_13: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_52, permute_23);  view_52 = permute_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:497 in forward, code: view_53 = torch.ops.aten.view.default(mm_13, [1, 2048, 5376]);  mm_13 = None
        view_53: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [1, 2048, 5376]);  mm_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:498 in forward, code: convert_element_type_85 = torch.ops.prims.convert_element_type.default(view_53, torch.float32);  view_53 = None
        convert_element_type_83: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_53, torch.float32);  view_53 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:499 in forward, code: pow_27 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_85, 2)
        pow_27: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:500 in forward, code: mean_13 = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        mean_13: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:501 in forward, code: add_31 = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        add_26: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:502 in forward, code: pow_28 = torch.ops.aten.pow.Tensor_Scalar(add_31, -0.5);  add_31 = None
        pow_28: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_26, -0.5);  add_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:503 in forward, code: mul_58 = torch.ops.aten.mul.Tensor(convert_element_type_85, pow_28);  convert_element_type_85 = pow_28 = None
        mul_54: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_83, pow_28);  convert_element_type_83 = pow_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:504 in forward, code: convert_element_type_86 = torch.ops.prims.convert_element_type.default(arg31_1, torch.float32);  arg31_1 = None
        convert_element_type_84: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg31_1, torch.float32);  arg31_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:505 in forward, code: mul_59 = torch.ops.aten.mul.Tensor(mul_58, convert_element_type_86);  mul_58 = convert_element_type_86 = None
        mul_55: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, convert_element_type_84);  mul_54 = convert_element_type_84 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:506 in forward, code: convert_element_type_87 = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        convert_element_type_85: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:507 in forward, code: add_32 = torch.ops.aten.add.Tensor(add_27, convert_element_type_87);  add_27 = convert_element_type_87 = None
        add_27: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_22, convert_element_type_85);  add_22 = convert_element_type_85 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:508 in forward, code: mul_60 = torch.ops.aten.mul.Tensor(add_32, arg32_1);  add_32 = arg32_1 = None
        mul_56: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, arg32_1);  add_27 = arg32_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:509 in forward, code: convert_element_type_88 = torch.ops.prims.convert_element_type.default(mul_60, torch.float32)
        convert_element_type_86: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:510 in forward, code: pow_29 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_88, 2)
        pow_29: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_86, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:511 in forward, code: mean_14 = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
        mean_14: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:512 in forward, code: add_33 = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        add_28: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:513 in forward, code: pow_30 = torch.ops.aten.pow.Tensor_Scalar(add_33, -0.5);  add_33 = None
        pow_30: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_28, -0.5);  add_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:514 in forward, code: mul_61 = torch.ops.aten.mul.Tensor(convert_element_type_88, pow_30);  convert_element_type_88 = pow_30 = None
        mul_57: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_86, pow_30);  convert_element_type_86 = pow_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:515 in forward, code: convert_element_type_89 = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        convert_element_type_87: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:516 in forward, code: mul_62 = torch.ops.aten.mul.Tensor(mul_61, convert_element_type_89);  mul_61 = convert_element_type_89 = None
        mul_58: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, convert_element_type_87);  mul_57 = convert_element_type_87 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:517 in forward, code: convert_element_type_90 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        convert_element_type_88: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:519 in forward, code: view_54 = torch.ops.aten.view.default(convert_element_type_90, [2048, 5376])
        view_54: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:518 in forward, code: permute_24 = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        permute_24: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:520 in forward, code: mm_14 = torch.ops.aten.mm.default(view_54, permute_24);  view_54 = permute_24 = None
        mm_14: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_54, permute_24);  view_54 = permute_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:521 in forward, code: view_55 = torch.ops.aten.view.default(mm_14, [1, 2048, 8192]);  mm_14 = None
        view_55: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [1, 2048, 8192]);  mm_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:522 in forward, code: view_56 = torch.ops.aten.view.default(view_55, [1, 2048, -1, 256]);  view_55 = None
        view_56: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_55, [1, 2048, -1, 256]);  view_55 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:523 in forward, code: convert_element_type_93 = torch.ops.prims.convert_element_type.default(view_56, torch.float32);  view_56 = None
        convert_element_type_91: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_56, torch.float32);  view_56 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:524 in forward, code: pow_31 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2)
        pow_31: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_91, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:525 in forward, code: mean_15 = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        mean_15: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:526 in forward, code: add_34 = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        add_29: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:527 in forward, code: pow_32 = torch.ops.aten.pow.Tensor_Scalar(add_34, -0.5);  add_34 = None
        pow_32: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_29, -0.5);  add_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:528 in forward, code: mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_93, pow_32);  convert_element_type_93 = pow_32 = None
        mul_59: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, pow_32);  convert_element_type_91 = pow_32 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:529 in forward, code: convert_element_type_94 = torch.ops.prims.convert_element_type.default(arg35_1, torch.float32);  arg35_1 = None
        convert_element_type_92: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg35_1, torch.float32);  arg35_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:530 in forward, code: mul_64 = torch.ops.aten.mul.Tensor(mul_63, convert_element_type_94);  mul_63 = convert_element_type_94 = None
        mul_60: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, convert_element_type_92);  mul_59 = convert_element_type_92 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:531 in forward, code: convert_element_type_95 = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        convert_element_type_93: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_60, torch.bfloat16);  mul_60 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:532 in forward, code: unsqueeze_35 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_35: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:534 in forward, code: mul_65 = torch.ops.aten.mul.Tensor(convert_element_type_95, unsqueeze_35);  unsqueeze_35 = None
        mul_61: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, unsqueeze_35);  unsqueeze_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:536 in forward, code: slice_16 = torch.ops.aten.slice.Tensor(convert_element_type_95, 3, 128, 9223372036854775807);  convert_element_type_95 = None
        slice_16: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_93, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:537 in forward, code: neg_4 = torch.ops.aten.neg.default(slice_16);  slice_16 = None
        neg_4: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_16);  slice_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:535 in forward, code: slice_15 = torch.ops.aten.slice.Tensor(convert_element_type_95, 3, 0, 128)
        slice_15: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_93, 3, 0, 128);  convert_element_type_93 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:538 in forward, code: cat_6 = torch.ops.aten.cat.default([neg_4, slice_15], -1);  neg_4 = slice_15 = None
        cat_6: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_4, slice_15], -1);  neg_4 = slice_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:533 in forward, code: unsqueeze_36 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_36: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:539 in forward, code: mul_66 = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_36);  cat_6 = unsqueeze_36 = None
        mul_62: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_36);  cat_6 = unsqueeze_36 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:540 in forward, code: add_35 = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
        add_30: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:541 in forward, code: permute_25 = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None
        permute_25: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_30, [0, 2, 1, 3]);  add_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:590 in forward, code: convert_element_type_105 = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None
        convert_element_type_103: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:593 in forward, code: mul_72 = torch.ops.aten.mul.Scalar(convert_element_type_105, 1.0);  convert_element_type_105 = None
        mul_68: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(convert_element_type_103, 1.0);  convert_element_type_103 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:596 in forward, code: expand_18 = torch.ops.aten.expand.default(mul_72, [1, 32, 2048, 256]);  mul_72 = None
        expand_18: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.expand.default(mul_68, [1, 32, 2048, 256]);  mul_68 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:597 in forward, code: view_65 = torch.ops.aten.view.default(expand_18, [32, 2048, 256]);  expand_18 = None
        view_65: "f32[32, 2048, 256][256, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(expand_18, [32, 2048, 256]);  expand_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:543 in forward, code: view_57 = torch.ops.aten.view.default(convert_element_type_90, [2048, 5376])
        view_57: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:542 in forward, code: permute_26 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        permute_26: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:544 in forward, code: mm_15 = torch.ops.aten.mm.default(view_57, permute_26);  view_57 = permute_26 = None
        mm_15: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_57, permute_26);  view_57 = permute_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:545 in forward, code: view_58 = torch.ops.aten.view.default(mm_15, [1, 2048, 4096]);  mm_15 = None
        view_58: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [1, 2048, 4096]);  mm_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:546 in forward, code: view_59 = torch.ops.aten.view.default(view_58, [1, 2048, -1, 256]);  view_58 = None
        view_59: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_58, [1, 2048, -1, 256]);  view_58 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:552 in forward, code: convert_element_type_100 = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        convert_element_type_98: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:553 in forward, code: pow_33 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_100, 2)
        pow_33: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_98, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:554 in forward, code: mean_16 = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None
        mean_16: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:555 in forward, code: add_36 = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        add_31: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:556 in forward, code: pow_34 = torch.ops.aten.pow.Tensor_Scalar(add_36, -0.5);  add_36 = None
        pow_34: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_31, -0.5);  add_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:557 in forward, code: mul_67 = torch.ops.aten.mul.Tensor(convert_element_type_100, pow_34);  convert_element_type_100 = pow_34 = None
        mul_63: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_98, pow_34);  convert_element_type_98 = pow_34 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:558 in forward, code: convert_element_type_101 = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None
        convert_element_type_99: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:559 in forward, code: mul_68 = torch.ops.aten.mul.Tensor(mul_67, convert_element_type_101);  mul_67 = convert_element_type_101 = None
        mul_64: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, convert_element_type_99);  mul_63 = convert_element_type_99 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:560 in forward, code: convert_element_type_102 = torch.ops.prims.convert_element_type.default(mul_68, torch.bfloat16);  mul_68 = None
        convert_element_type_100: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:561 in forward, code: unsqueeze_37 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_37: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:563 in forward, code: mul_69 = torch.ops.aten.mul.Tensor(convert_element_type_102, unsqueeze_37);  unsqueeze_37 = None
        mul_65: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_100, unsqueeze_37);  unsqueeze_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:565 in forward, code: slice_18 = torch.ops.aten.slice.Tensor(convert_element_type_102, 3, 128, 9223372036854775807);  convert_element_type_102 = None
        slice_18: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_100, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:566 in forward, code: neg_5 = torch.ops.aten.neg.default(slice_18);  slice_18 = None
        neg_5: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_18);  slice_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:564 in forward, code: slice_17 = torch.ops.aten.slice.Tensor(convert_element_type_102, 3, 0, 128)
        slice_17: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_100, 3, 0, 128);  convert_element_type_100 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:567 in forward, code: cat_7 = torch.ops.aten.cat.default([neg_5, slice_17], -1);  neg_5 = slice_17 = None
        cat_7: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_5, slice_17], -1);  neg_5 = slice_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:562 in forward, code: unsqueeze_38 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_38: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:568 in forward, code: mul_70 = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_38);  cat_7 = unsqueeze_38 = None
        mul_66: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_38);  cat_7 = unsqueeze_38 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:569 in forward, code: add_37 = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
        add_32: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:570 in forward, code: permute_28 = torch.ops.aten.permute.default(add_37, [0, 2, 1, 3]);  add_37 = None
        permute_28: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_32, [0, 2, 1, 3]);  add_32 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:579 in forward, code: unsqueeze_39 = torch.ops.aten.unsqueeze.default(permute_28, 2);  permute_28 = None
        unsqueeze_39: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_28, 2);  permute_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:580 in forward, code: expand_16 = torch.ops.aten.expand.default(unsqueeze_39, [1, 16, 2, 2048, 256]);  unsqueeze_39 = None
        expand_16: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_39, [1, 16, 2, 2048, 256]);  unsqueeze_39 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:581 in forward, code: clone_10 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        clone_8: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:582 in forward, code: view_63 = torch.ops.aten.view.default(clone_10, [1, 32, 2048, 256]);  clone_10 = None
        view_63: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1, 32, 2048, 256]);  clone_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:591 in forward, code: convert_element_type_106 = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        convert_element_type_104: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:594 in forward, code: permute_30 = torch.ops.aten.permute.default(convert_element_type_106, [0, 1, 3, 2]);  convert_element_type_106 = None
        permute_30: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_104, [0, 1, 3, 2]);  convert_element_type_104 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:595 in forward, code: mul_73 = torch.ops.aten.mul.Scalar(permute_30, 1.0);  permute_30 = None
        mul_69: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_30, 1.0);  permute_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:598 in forward, code: expand_19 = torch.ops.aten.expand.default(mul_73, [1, 32, 256, 2048]);  mul_73 = None
        expand_19: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_69, [1, 32, 256, 2048]);  mul_69 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:599 in forward, code: view_66 = torch.ops.aten.view.default(expand_19, [32, 256, 2048]);  expand_19 = None
        view_66: "f32[32, 256, 2048][524288, 1, 256]cuda:0" = torch.ops.aten.reshape.default(expand_19, [32, 256, 2048]);  expand_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:600 in forward, code: bmm_4 = torch.ops.aten.bmm.default(view_65, view_66);  view_65 = view_66 = None
        bmm_4: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.bmm.default(view_65, view_66);  view_65 = view_66 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:601 in forward, code: view_67 = torch.ops.aten.view.default(bmm_4, [1, 32, 2048, 2048]);  bmm_4 = None
        view_67: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [1, 32, 2048, 2048]);  bmm_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:588 in forward, code: full_default_10 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:587 in forward, code: full_default_9 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:589 in forward, code: where_4 = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        where_4: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:602 in forward, code: add_39 = torch.ops.aten.add.Tensor(view_67, where_4);  view_67 = where_4 = None
        add_34: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_67, where_4);  view_67 = where_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:608 in forward, code: eq_4 = torch.ops.aten.eq.Scalar(add_39, -inf);  add_39 = None
        eq_4: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Scalar(add_34, -inf)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:609 in forward, code: logical_not_4 = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        logical_not_4: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:610 in forward, code: any_3 = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        any_3: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:611 in forward, code: logical_not_5 = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        logical_not_5: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_3);  any_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:612 in forward, code: full_default_11 = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:603 in forward, code: amax_2 = torch.ops.aten.amax.default(add_39, [-1], True)
        amax_2: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_34, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:604 in forward, code: sub_7 = torch.ops.aten.sub.Tensor(add_39, amax_2);  amax_2 = None
        sub_7: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_34, amax_2);  add_34 = amax_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:605 in forward, code: exp_2 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        exp_2: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:606 in forward, code: sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        sum_3: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:607 in forward, code: div_2 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        div_2: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:613 in forward, code: where_5 = torch.ops.aten.where.self(logical_not_5, full_default_11, div_2);  logical_not_5 = full_default_11 = div_2 = None
        where_5: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(logical_not_5, full_default_11, div_2);  logical_not_5 = full_default_11 = div_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:614 in forward, code: expand_20 = torch.ops.aten.expand.default(where_5, [1, 32, 2048, 2048]);  where_5 = None
        expand_20: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_5, [1, 32, 2048, 2048]);  where_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:615 in forward, code: view_68 = torch.ops.aten.view.default(expand_20, [32, 2048, 2048]);  expand_20 = None
        view_68: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(expand_20, [32, 2048, 2048]);  expand_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:548 in forward, code: view_60 = torch.ops.aten.view.default(convert_element_type_90, [2048, 5376]);  convert_element_type_90 = None
        view_60: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_88, [2048, 5376]);  convert_element_type_88 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:547 in forward, code: permute_27 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        permute_27: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:549 in forward, code: mm_16 = torch.ops.aten.mm.default(view_60, permute_27);  view_60 = permute_27 = None
        mm_16: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_60, permute_27);  view_60 = permute_27 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:550 in forward, code: view_61 = torch.ops.aten.view.default(mm_16, [1, 2048, 4096]);  mm_16 = None
        view_61: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [1, 2048, 4096]);  mm_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:551 in forward, code: view_62 = torch.ops.aten.view.default(view_61, [1, 2048, -1, 256]);  view_61 = None
        view_62: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_61, [1, 2048, -1, 256]);  view_61 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:571 in forward, code: convert_element_type_103 = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        convert_element_type_101: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:572 in forward, code: pow_35 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 2)
        pow_35: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:573 in forward, code: mean_17 = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        mean_17: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:574 in forward, code: add_38 = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        add_33: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:575 in forward, code: pow_36 = torch.ops.aten.pow.Tensor_Scalar(add_38, -0.5);  add_38 = None
        pow_36: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_33, -0.5);  add_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:576 in forward, code: mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_103, pow_36);  convert_element_type_103 = pow_36 = None
        mul_67: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_101, pow_36);  convert_element_type_101 = pow_36 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:577 in forward, code: convert_element_type_104 = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        convert_element_type_102: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_67, torch.bfloat16);  mul_67 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:578 in forward, code: permute_29 = torch.ops.aten.permute.default(convert_element_type_104, [0, 2, 1, 3]);  convert_element_type_104 = None
        permute_29: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_102, [0, 2, 1, 3]);  convert_element_type_102 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:583 in forward, code: unsqueeze_40 = torch.ops.aten.unsqueeze.default(permute_29, 2);  permute_29 = None
        unsqueeze_40: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_29, 2);  permute_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:584 in forward, code: expand_17 = torch.ops.aten.expand.default(unsqueeze_40, [1, 16, 2, 2048, 256]);  unsqueeze_40 = None
        expand_17: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_40, [1, 16, 2, 2048, 256]);  unsqueeze_40 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:585 in forward, code: clone_11 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        clone_9: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:586 in forward, code: view_64 = torch.ops.aten.view.default(clone_11, [1, 32, 2048, 256]);  clone_11 = None
        view_64: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1, 32, 2048, 256]);  clone_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:592 in forward, code: convert_element_type_107 = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None
        convert_element_type_105: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:616 in forward, code: expand_21 = torch.ops.aten.expand.default(convert_element_type_107, [1, 32, 2048, 256]);  convert_element_type_107 = None
        expand_21: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_105, [1, 32, 2048, 256]);  convert_element_type_105 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:617 in forward, code: view_69 = torch.ops.aten.view.default(expand_21, [32, 2048, 256]);  expand_21 = None
        view_69: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(expand_21, [32, 2048, 256]);  expand_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:618 in forward, code: bmm_5 = torch.ops.aten.bmm.default(view_68, view_69);  view_68 = view_69 = None
        bmm_5: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_68, view_69);  view_68 = view_69 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:619 in forward, code: view_70 = torch.ops.aten.view.default(bmm_5, [1, 32, 2048, 256]);  bmm_5 = None
        view_70: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [1, 32, 2048, 256]);  bmm_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:620 in forward, code: convert_element_type_109 = torch.ops.prims.convert_element_type.default(view_70, torch.bfloat16);  view_70 = None
        convert_element_type_106: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.bfloat16);  view_70 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:621 in forward, code: permute_31 = torch.ops.aten.permute.default(convert_element_type_109, [0, 2, 1, 3]);  convert_element_type_109 = None
        permute_31: "bf16[1, 2048, 32, 256][16777216, 256, 524288, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_106, [0, 2, 1, 3]);  convert_element_type_106 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:622 in forward, code: clone_12 = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        clone_10: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:623 in forward, code: view_71 = torch.ops.aten.view.default(clone_12, [1, 2048, -1]);  clone_12 = None
        view_71: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 2048, -1]);  clone_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:625 in forward, code: view_72 = torch.ops.aten.view.default(view_71, [2048, 8192]);  view_71 = None
        view_72: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_71, [2048, 8192]);  view_71 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:624 in forward, code: permute_32 = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        permute_32: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:626 in forward, code: mm_17 = torch.ops.aten.mm.default(view_72, permute_32);  view_72 = permute_32 = None
        mm_17: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_72, permute_32);  view_72 = permute_32 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:627 in forward, code: view_73 = torch.ops.aten.view.default(mm_17, [1, 2048, 5376]);  mm_17 = None
        view_73: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [1, 2048, 5376]);  mm_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:628 in forward, code: convert_element_type_112 = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        convert_element_type_109: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:629 in forward, code: pow_37 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_112, 2)
        pow_37: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_109, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:630 in forward, code: mean_18 = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None
        mean_18: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:631 in forward, code: add_40 = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        add_35: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:632 in forward, code: pow_38 = torch.ops.aten.pow.Tensor_Scalar(add_40, -0.5);  add_40 = None
        pow_38: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_35, -0.5);  add_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:633 in forward, code: mul_74 = torch.ops.aten.mul.Tensor(convert_element_type_112, pow_38);  convert_element_type_112 = pow_38 = None
        mul_70: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_109, pow_38);  convert_element_type_109 = pow_38 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:634 in forward, code: convert_element_type_113 = torch.ops.prims.convert_element_type.default(arg40_1, torch.float32);  arg40_1 = None
        convert_element_type_110: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg40_1, torch.float32);  arg40_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:635 in forward, code: mul_75 = torch.ops.aten.mul.Tensor(mul_74, convert_element_type_113);  mul_74 = convert_element_type_113 = None
        mul_71: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, convert_element_type_110);  mul_70 = convert_element_type_110 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:636 in forward, code: convert_element_type_114 = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None
        convert_element_type_111: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:637 in forward, code: add_41 = torch.ops.aten.add.Tensor(mul_60, convert_element_type_114);  mul_60 = convert_element_type_114 = None
        add_36: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, convert_element_type_111);  mul_56 = convert_element_type_111 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:638 in forward, code: convert_element_type_115 = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        convert_element_type_112: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:639 in forward, code: pow_39 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_115, 2)
        pow_39: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_112, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:640 in forward, code: mean_19 = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        mean_19: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:641 in forward, code: add_42 = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        add_37: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:642 in forward, code: pow_40 = torch.ops.aten.pow.Tensor_Scalar(add_42, -0.5);  add_42 = None
        pow_40: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_37, -0.5);  add_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:643 in forward, code: mul_76 = torch.ops.aten.mul.Tensor(convert_element_type_115, pow_40);  convert_element_type_115 = pow_40 = None
        mul_72: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_112, pow_40);  convert_element_type_112 = pow_40 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:644 in forward, code: convert_element_type_116 = torch.ops.prims.convert_element_type.default(arg41_1, torch.float32);  arg41_1 = None
        convert_element_type_113: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg41_1, torch.float32);  arg41_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:645 in forward, code: mul_77 = torch.ops.aten.mul.Tensor(mul_76, convert_element_type_116);  mul_76 = convert_element_type_116 = None
        mul_73: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, convert_element_type_113);  mul_72 = convert_element_type_113 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:646 in forward, code: convert_element_type_117 = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        convert_element_type_114: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:648 in forward, code: view_74 = torch.ops.aten.view.default(convert_element_type_117, [2048, 5376])
        view_74: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_114, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:647 in forward, code: permute_33 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        permute_33: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:649 in forward, code: mm_18 = torch.ops.aten.mm.default(view_74, permute_33);  view_74 = permute_33 = None
        mm_18: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_74, permute_33);  view_74 = permute_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:650 in forward, code: view_75 = torch.ops.aten.view.default(mm_18, [1, 2048, 21504]);  mm_18 = None
        view_75: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [1, 2048, 21504]);  mm_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:651 in forward, code: convert_element_type_120 = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        convert_element_type_117: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:657 in forward, code: mul_82 = torch.ops.aten.mul.Tensor(convert_element_type_120, 0.5);  convert_element_type_120 = None
        mul_78: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_117, 0.5)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:652 in forward, code: mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_120, convert_element_type_120)
        mul_74: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_117, convert_element_type_117)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:653 in forward, code: mul_79 = torch.ops.aten.mul.Tensor(mul_78, convert_element_type_120);  mul_78 = None
        mul_75: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, convert_element_type_117);  mul_74 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:654 in forward, code: mul_80 = torch.ops.aten.mul.Tensor(mul_79, 0.044715);  mul_79 = None
        mul_76: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, 0.044715);  mul_75 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:655 in forward, code: add_43 = torch.ops.aten.add.Tensor(convert_element_type_120, mul_80);  mul_80 = None
        add_38: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_117, mul_76);  convert_element_type_117 = mul_76 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:656 in forward, code: mul_81 = torch.ops.aten.mul.Tensor(add_43, 0.7978845608028654);  add_43 = None
        mul_77: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:658 in forward, code: tanh_2 = torch.ops.aten.tanh.default(mul_81);  mul_81 = None
        tanh_2: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_77);  mul_77 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:659 in forward, code: add_44 = torch.ops.aten.add.Tensor(tanh_2, 1);  tanh_2 = None
        add_39: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1);  tanh_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:660 in forward, code: mul_83 = torch.ops.aten.mul.Tensor(mul_82, add_44);  mul_82 = add_44 = None
        mul_79: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, add_39);  mul_78 = add_39 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:661 in forward, code: convert_element_type_121 = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        convert_element_type_118: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:663 in forward, code: view_76 = torch.ops.aten.view.default(convert_element_type_117, [2048, 5376]);  convert_element_type_117 = None
        view_76: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_114, [2048, 5376]);  convert_element_type_114 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:662 in forward, code: permute_34 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        permute_34: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:664 in forward, code: mm_19 = torch.ops.aten.mm.default(view_76, permute_34);  view_76 = permute_34 = None
        mm_19: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_34);  view_76 = permute_34 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:665 in forward, code: view_77 = torch.ops.aten.view.default(mm_19, [1, 2048, 21504]);  mm_19 = None
        view_77: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [1, 2048, 21504]);  mm_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:666 in forward, code: mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_121, view_77);  convert_element_type_121 = view_77 = None
        mul_80: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_118, view_77);  convert_element_type_118 = view_77 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:668 in forward, code: view_78 = torch.ops.aten.view.default(mul_84, [2048, 21504]);  mul_84 = None
        view_78: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_80, [2048, 21504]);  mul_80 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:667 in forward, code: permute_35 = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        permute_35: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:669 in forward, code: mm_20 = torch.ops.aten.mm.default(view_78, permute_35);  view_78 = permute_35 = None
        mm_20: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_78, permute_35);  view_78 = permute_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:670 in forward, code: view_79 = torch.ops.aten.view.default(mm_20, [1, 2048, 5376]);  mm_20 = None
        view_79: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [1, 2048, 5376]);  mm_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:671 in forward, code: convert_element_type_126 = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        convert_element_type_123: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:672 in forward, code: pow_41 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_126, 2)
        pow_41: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:673 in forward, code: mean_20 = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None
        mean_20: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:674 in forward, code: add_45 = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        add_40: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:675 in forward, code: pow_42 = torch.ops.aten.pow.Tensor_Scalar(add_45, -0.5);  add_45 = None
        pow_42: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_40, -0.5);  add_40 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:676 in forward, code: mul_85 = torch.ops.aten.mul.Tensor(convert_element_type_126, pow_42);  convert_element_type_126 = pow_42 = None
        mul_81: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_123, pow_42);  convert_element_type_123 = pow_42 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:677 in forward, code: convert_element_type_127 = torch.ops.prims.convert_element_type.default(arg45_1, torch.float32);  arg45_1 = None
        convert_element_type_124: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg45_1, torch.float32);  arg45_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:678 in forward, code: mul_86 = torch.ops.aten.mul.Tensor(mul_85, convert_element_type_127);  mul_85 = convert_element_type_127 = None
        mul_82: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, convert_element_type_124);  mul_81 = convert_element_type_124 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:679 in forward, code: convert_element_type_128 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        convert_element_type_125: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_82, torch.bfloat16);  mul_82 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:680 in forward, code: add_46 = torch.ops.aten.add.Tensor(add_41, convert_element_type_128);  add_41 = convert_element_type_128 = None
        add_41: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_36, convert_element_type_125);  add_36 = convert_element_type_125 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:681 in forward, code: mul_87 = torch.ops.aten.mul.Tensor(add_46, arg46_1);  add_46 = arg46_1 = None
        mul_83: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, arg46_1);  add_41 = arg46_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:682 in forward, code: convert_element_type_129 = torch.ops.prims.convert_element_type.default(mul_87, torch.float32)
        convert_element_type_126: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_83, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:683 in forward, code: pow_43 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_129, 2)
        pow_43: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_126, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:684 in forward, code: mean_21 = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        mean_21: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:685 in forward, code: add_47 = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        add_42: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:686 in forward, code: pow_44 = torch.ops.aten.pow.Tensor_Scalar(add_47, -0.5);  add_47 = None
        pow_44: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_42, -0.5);  add_42 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:687 in forward, code: mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_129, pow_44);  convert_element_type_129 = pow_44 = None
        mul_84: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, pow_44);  convert_element_type_126 = pow_44 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:688 in forward, code: convert_element_type_130 = torch.ops.prims.convert_element_type.default(arg47_1, torch.float32);  arg47_1 = None
        convert_element_type_127: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg47_1, torch.float32);  arg47_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:689 in forward, code: mul_89 = torch.ops.aten.mul.Tensor(mul_88, convert_element_type_130);  mul_88 = convert_element_type_130 = None
        mul_85: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, convert_element_type_127);  mul_84 = convert_element_type_127 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:690 in forward, code: convert_element_type_131 = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        convert_element_type_128: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:692 in forward, code: view_80 = torch.ops.aten.view.default(convert_element_type_131, [2048, 5376])
        view_80: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_128, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:691 in forward, code: permute_36 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        permute_36: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:693 in forward, code: mm_21 = torch.ops.aten.mm.default(view_80, permute_36);  view_80 = permute_36 = None
        mm_21: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_80, permute_36);  view_80 = permute_36 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:694 in forward, code: view_81 = torch.ops.aten.view.default(mm_21, [1, 2048, 8192]);  mm_21 = None
        view_81: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [1, 2048, 8192]);  mm_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:695 in forward, code: view_82 = torch.ops.aten.view.default(view_81, [1, 2048, -1, 256]);  view_81 = None
        view_82: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [1, 2048, -1, 256]);  view_81 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:696 in forward, code: convert_element_type_134 = torch.ops.prims.convert_element_type.default(view_82, torch.float32);  view_82 = None
        convert_element_type_131: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_82, torch.float32);  view_82 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:697 in forward, code: pow_45 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        pow_45: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_131, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:698 in forward, code: mean_22 = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None
        mean_22: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:699 in forward, code: add_48 = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        add_43: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:700 in forward, code: pow_46 = torch.ops.aten.pow.Tensor_Scalar(add_48, -0.5);  add_48 = None
        pow_46: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_43, -0.5);  add_43 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:701 in forward, code: mul_90 = torch.ops.aten.mul.Tensor(convert_element_type_134, pow_46);  convert_element_type_134 = pow_46 = None
        mul_86: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_131, pow_46);  convert_element_type_131 = pow_46 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:702 in forward, code: convert_element_type_135 = torch.ops.prims.convert_element_type.default(arg49_1, torch.float32);  arg49_1 = None
        convert_element_type_132: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg49_1, torch.float32);  arg49_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:703 in forward, code: mul_91 = torch.ops.aten.mul.Tensor(mul_90, convert_element_type_135);  mul_90 = convert_element_type_135 = None
        mul_87: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, convert_element_type_132);  mul_86 = convert_element_type_132 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:704 in forward, code: convert_element_type_136 = torch.ops.prims.convert_element_type.default(mul_91, torch.bfloat16);  mul_91 = None
        convert_element_type_133: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:705 in forward, code: unsqueeze_41 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_41: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:707 in forward, code: mul_92 = torch.ops.aten.mul.Tensor(convert_element_type_136, unsqueeze_41);  unsqueeze_41 = None
        mul_88: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_133, unsqueeze_41);  unsqueeze_41 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:709 in forward, code: slice_20 = torch.ops.aten.slice.Tensor(convert_element_type_136, 3, 128, 9223372036854775807);  convert_element_type_136 = None
        slice_20: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_133, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:710 in forward, code: neg_6 = torch.ops.aten.neg.default(slice_20);  slice_20 = None
        neg_6: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_20);  slice_20 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:708 in forward, code: slice_19 = torch.ops.aten.slice.Tensor(convert_element_type_136, 3, 0, 128)
        slice_19: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_133, 3, 0, 128);  convert_element_type_133 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:711 in forward, code: cat_8 = torch.ops.aten.cat.default([neg_6, slice_19], -1);  neg_6 = slice_19 = None
        cat_8: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_6, slice_19], -1);  neg_6 = slice_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:706 in forward, code: unsqueeze_42 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_42: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:712 in forward, code: mul_93 = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_42);  cat_8 = unsqueeze_42 = None
        mul_89: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_42);  cat_8 = unsqueeze_42 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:713 in forward, code: add_49 = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        add_44: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:714 in forward, code: permute_37 = torch.ops.aten.permute.default(add_49, [0, 2, 1, 3]);  add_49 = None
        permute_37: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_44, [0, 2, 1, 3]);  add_44 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:763 in forward, code: convert_element_type_146 = torch.ops.prims.convert_element_type.default(permute_37, torch.float32);  permute_37 = None
        convert_element_type_143: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_37, torch.float32);  permute_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:766 in forward, code: mul_99 = torch.ops.aten.mul.Scalar(convert_element_type_146, 1.0);  convert_element_type_146 = None
        mul_95: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(convert_element_type_143, 1.0);  convert_element_type_143 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:769 in forward, code: expand_24 = torch.ops.aten.expand.default(mul_99, [1, 32, 2048, 256]);  mul_99 = None
        expand_24: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.expand.default(mul_95, [1, 32, 2048, 256]);  mul_95 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:770 in forward, code: view_91 = torch.ops.aten.view.default(expand_24, [32, 2048, 256]);  expand_24 = None
        view_91: "f32[32, 2048, 256][256, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(expand_24, [32, 2048, 256]);  expand_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:716 in forward, code: view_83 = torch.ops.aten.view.default(convert_element_type_131, [2048, 5376])
        view_83: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_128, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:715 in forward, code: permute_38 = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        permute_38: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:717 in forward, code: mm_22 = torch.ops.aten.mm.default(view_83, permute_38);  view_83 = permute_38 = None
        mm_22: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_83, permute_38);  view_83 = permute_38 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:718 in forward, code: view_84 = torch.ops.aten.view.default(mm_22, [1, 2048, 4096]);  mm_22 = None
        view_84: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [1, 2048, 4096]);  mm_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:719 in forward, code: view_85 = torch.ops.aten.view.default(view_84, [1, 2048, -1, 256]);  view_84 = None
        view_85: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [1, 2048, -1, 256]);  view_84 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:725 in forward, code: convert_element_type_141 = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        convert_element_type_138: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:726 in forward, code: pow_47 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_141, 2)
        pow_47: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_138, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:727 in forward, code: mean_23 = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        mean_23: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:728 in forward, code: add_50 = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        add_45: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:729 in forward, code: pow_48 = torch.ops.aten.pow.Tensor_Scalar(add_50, -0.5);  add_50 = None
        pow_48: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_45, -0.5);  add_45 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:730 in forward, code: mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_141, pow_48);  convert_element_type_141 = pow_48 = None
        mul_90: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_138, pow_48);  convert_element_type_138 = pow_48 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:731 in forward, code: convert_element_type_142 = torch.ops.prims.convert_element_type.default(arg52_1, torch.float32);  arg52_1 = None
        convert_element_type_139: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg52_1, torch.float32);  arg52_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:732 in forward, code: mul_95 = torch.ops.aten.mul.Tensor(mul_94, convert_element_type_142);  mul_94 = convert_element_type_142 = None
        mul_91: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, convert_element_type_139);  mul_90 = convert_element_type_139 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:733 in forward, code: convert_element_type_143 = torch.ops.prims.convert_element_type.default(mul_95, torch.bfloat16);  mul_95 = None
        convert_element_type_140: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_91, torch.bfloat16);  mul_91 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:734 in forward, code: unsqueeze_43 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_43: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:736 in forward, code: mul_96 = torch.ops.aten.mul.Tensor(convert_element_type_143, unsqueeze_43);  unsqueeze_43 = None
        mul_92: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, unsqueeze_43);  unsqueeze_43 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:738 in forward, code: slice_22 = torch.ops.aten.slice.Tensor(convert_element_type_143, 3, 128, 9223372036854775807);  convert_element_type_143 = None
        slice_22: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_140, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:739 in forward, code: neg_7 = torch.ops.aten.neg.default(slice_22);  slice_22 = None
        neg_7: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_22);  slice_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:737 in forward, code: slice_21 = torch.ops.aten.slice.Tensor(convert_element_type_143, 3, 0, 128)
        slice_21: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_140, 3, 0, 128);  convert_element_type_140 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:740 in forward, code: cat_9 = torch.ops.aten.cat.default([neg_7, slice_21], -1);  neg_7 = slice_21 = None
        cat_9: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_7, slice_21], -1);  neg_7 = slice_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:735 in forward, code: unsqueeze_44 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_44: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:741 in forward, code: mul_97 = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_44);  cat_9 = unsqueeze_44 = None
        mul_93: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_44);  cat_9 = unsqueeze_44 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:742 in forward, code: add_51 = torch.ops.aten.add.Tensor(mul_96, mul_97);  mul_96 = mul_97 = None
        add_46: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:743 in forward, code: permute_40 = torch.ops.aten.permute.default(add_51, [0, 2, 1, 3]);  add_51 = None
        permute_40: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_46, [0, 2, 1, 3]);  add_46 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:752 in forward, code: unsqueeze_45 = torch.ops.aten.unsqueeze.default(permute_40, 2);  permute_40 = None
        unsqueeze_45: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_40, 2);  permute_40 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:753 in forward, code: expand_22 = torch.ops.aten.expand.default(unsqueeze_45, [1, 16, 2, 2048, 256]);  unsqueeze_45 = None
        expand_22: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_45, [1, 16, 2, 2048, 256]);  unsqueeze_45 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:754 in forward, code: clone_13 = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        clone_11: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:755 in forward, code: view_89 = torch.ops.aten.view.default(clone_13, [1, 32, 2048, 256]);  clone_13 = None
        view_89: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [1, 32, 2048, 256]);  clone_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:764 in forward, code: convert_element_type_147 = torch.ops.prims.convert_element_type.default(view_89, torch.float32);  view_89 = None
        convert_element_type_144: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_89, torch.float32);  view_89 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:767 in forward, code: permute_42 = torch.ops.aten.permute.default(convert_element_type_147, [0, 1, 3, 2]);  convert_element_type_147 = None
        permute_42: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_144, [0, 1, 3, 2]);  convert_element_type_144 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:768 in forward, code: mul_100 = torch.ops.aten.mul.Scalar(permute_42, 1.0);  permute_42 = None
        mul_96: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_42, 1.0);  permute_42 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:771 in forward, code: expand_25 = torch.ops.aten.expand.default(mul_100, [1, 32, 256, 2048]);  mul_100 = None
        expand_25: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_96, [1, 32, 256, 2048]);  mul_96 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:772 in forward, code: view_92 = torch.ops.aten.view.default(expand_25, [32, 256, 2048]);  expand_25 = None
        view_92: "f32[32, 256, 2048][524288, 1, 256]cuda:0" = torch.ops.aten.reshape.default(expand_25, [32, 256, 2048]);  expand_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:773 in forward, code: bmm_6 = torch.ops.aten.bmm.default(view_91, view_92);  view_91 = view_92 = None
        bmm_6: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.bmm.default(view_91, view_92);  view_91 = view_92 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:774 in forward, code: view_93 = torch.ops.aten.view.default(bmm_6, [1, 32, 2048, 2048]);  bmm_6 = None
        view_93: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [1, 32, 2048, 2048]);  bmm_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:761 in forward, code: full_default_13 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:760 in forward, code: full_default_12 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:762 in forward, code: where_6 = torch.ops.aten.where.self(expand_1, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        where_6: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_13, full_default_12);  full_default_13 = full_default_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:775 in forward, code: add_53 = torch.ops.aten.add.Tensor(view_93, where_6);  view_93 = where_6 = None
        add_48: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_93, where_6);  view_93 = where_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:781 in forward, code: eq_5 = torch.ops.aten.eq.Scalar(add_53, -inf);  add_53 = None
        eq_5: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Scalar(add_48, -inf)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:782 in forward, code: logical_not_6 = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        logical_not_6: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:783 in forward, code: any_4 = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        any_4: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:784 in forward, code: logical_not_7 = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        logical_not_7: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_4);  any_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:785 in forward, code: full_default_14 = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:776 in forward, code: amax_3 = torch.ops.aten.amax.default(add_53, [-1], True)
        amax_3: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_48, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:777 in forward, code: sub_8 = torch.ops.aten.sub.Tensor(add_53, amax_3);  amax_3 = None
        sub_8: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_48, amax_3);  add_48 = amax_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:778 in forward, code: exp_3 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        exp_3: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.exp.default(sub_8);  sub_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:779 in forward, code: sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        sum_4: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:780 in forward, code: div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        div_3: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:786 in forward, code: where_7 = torch.ops.aten.where.self(logical_not_7, full_default_14, div_3);  logical_not_7 = full_default_14 = div_3 = None
        where_7: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(logical_not_7, full_default_14, div_3);  logical_not_7 = full_default_14 = div_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:787 in forward, code: expand_26 = torch.ops.aten.expand.default(where_7, [1, 32, 2048, 2048]);  where_7 = None
        expand_26: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_7, [1, 32, 2048, 2048]);  where_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:788 in forward, code: view_94 = torch.ops.aten.view.default(expand_26, [32, 2048, 2048]);  expand_26 = None
        view_94: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(expand_26, [32, 2048, 2048]);  expand_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:721 in forward, code: view_86 = torch.ops.aten.view.default(convert_element_type_131, [2048, 5376]);  convert_element_type_131 = None
        view_86: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_128, [2048, 5376]);  convert_element_type_128 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:720 in forward, code: permute_39 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        permute_39: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:722 in forward, code: mm_23 = torch.ops.aten.mm.default(view_86, permute_39);  view_86 = permute_39 = None
        mm_23: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_86, permute_39);  view_86 = permute_39 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:723 in forward, code: view_87 = torch.ops.aten.view.default(mm_23, [1, 2048, 4096]);  mm_23 = None
        view_87: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [1, 2048, 4096]);  mm_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:724 in forward, code: view_88 = torch.ops.aten.view.default(view_87, [1, 2048, -1, 256]);  view_87 = None
        view_88: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_87, [1, 2048, -1, 256]);  view_87 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:744 in forward, code: convert_element_type_144 = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None
        convert_element_type_141: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:745 in forward, code: pow_49 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_144, 2)
        pow_49: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_141, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:746 in forward, code: mean_24 = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None
        mean_24: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:747 in forward, code: add_52 = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        add_47: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:748 in forward, code: pow_50 = torch.ops.aten.pow.Tensor_Scalar(add_52, -0.5);  add_52 = None
        pow_50: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_47, -0.5);  add_47 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:749 in forward, code: mul_98 = torch.ops.aten.mul.Tensor(convert_element_type_144, pow_50);  convert_element_type_144 = pow_50 = None
        mul_94: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, pow_50);  convert_element_type_141 = pow_50 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:750 in forward, code: convert_element_type_145 = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None
        convert_element_type_142: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:751 in forward, code: permute_41 = torch.ops.aten.permute.default(convert_element_type_145, [0, 2, 1, 3]);  convert_element_type_145 = None
        permute_41: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_142, [0, 2, 1, 3]);  convert_element_type_142 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:756 in forward, code: unsqueeze_46 = torch.ops.aten.unsqueeze.default(permute_41, 2);  permute_41 = None
        unsqueeze_46: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_41, 2);  permute_41 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:757 in forward, code: expand_23 = torch.ops.aten.expand.default(unsqueeze_46, [1, 16, 2, 2048, 256]);  unsqueeze_46 = None
        expand_23: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_46, [1, 16, 2, 2048, 256]);  unsqueeze_46 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:758 in forward, code: clone_14 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        clone_12: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:759 in forward, code: view_90 = torch.ops.aten.view.default(clone_14, [1, 32, 2048, 256]);  clone_14 = None
        view_90: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [1, 32, 2048, 256]);  clone_12 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:765 in forward, code: convert_element_type_148 = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None
        convert_element_type_145: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:789 in forward, code: expand_27 = torch.ops.aten.expand.default(convert_element_type_148, [1, 32, 2048, 256]);  convert_element_type_148 = None
        expand_27: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_145, [1, 32, 2048, 256]);  convert_element_type_145 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:790 in forward, code: view_95 = torch.ops.aten.view.default(expand_27, [32, 2048, 256]);  expand_27 = None
        view_95: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(expand_27, [32, 2048, 256]);  expand_27 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:791 in forward, code: bmm_7 = torch.ops.aten.bmm.default(view_94, view_95);  view_94 = view_95 = None
        bmm_7: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_94, view_95);  view_94 = view_95 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:792 in forward, code: view_96 = torch.ops.aten.view.default(bmm_7, [1, 32, 2048, 256]);  bmm_7 = None
        view_96: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_7, [1, 32, 2048, 256]);  bmm_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:793 in forward, code: convert_element_type_150 = torch.ops.prims.convert_element_type.default(view_96, torch.bfloat16);  view_96 = None
        convert_element_type_146: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_96, torch.bfloat16);  view_96 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:794 in forward, code: permute_43 = torch.ops.aten.permute.default(convert_element_type_150, [0, 2, 1, 3]);  convert_element_type_150 = None
        permute_43: "bf16[1, 2048, 32, 256][16777216, 256, 524288, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_146, [0, 2, 1, 3]);  convert_element_type_146 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:795 in forward, code: clone_15 = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        clone_13: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:796 in forward, code: view_97 = torch.ops.aten.view.default(clone_15, [1, 2048, -1]);  clone_15 = None
        view_97: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [1, 2048, -1]);  clone_13 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:798 in forward, code: view_98 = torch.ops.aten.view.default(view_97, [2048, 8192]);  view_97 = None
        view_98: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_97, [2048, 8192]);  view_97 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:797 in forward, code: permute_44 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        permute_44: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:799 in forward, code: mm_24 = torch.ops.aten.mm.default(view_98, permute_44);  view_98 = permute_44 = None
        mm_24: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_98, permute_44);  view_98 = permute_44 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:800 in forward, code: view_99 = torch.ops.aten.view.default(mm_24, [1, 2048, 5376]);  mm_24 = None
        view_99: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [1, 2048, 5376]);  mm_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:801 in forward, code: convert_element_type_153 = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        convert_element_type_149: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:802 in forward, code: pow_51 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2)
        pow_51: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_149, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:803 in forward, code: mean_25 = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        mean_25: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:804 in forward, code: add_54 = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        add_49: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:805 in forward, code: pow_52 = torch.ops.aten.pow.Tensor_Scalar(add_54, -0.5);  add_54 = None
        pow_52: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_49, -0.5);  add_49 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:806 in forward, code: mul_101 = torch.ops.aten.mul.Tensor(convert_element_type_153, pow_52);  convert_element_type_153 = pow_52 = None
        mul_97: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_149, pow_52);  convert_element_type_149 = pow_52 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:807 in forward, code: convert_element_type_154 = torch.ops.prims.convert_element_type.default(arg54_1, torch.float32);  arg54_1 = None
        convert_element_type_150: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg54_1, torch.float32);  arg54_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:808 in forward, code: mul_102 = torch.ops.aten.mul.Tensor(mul_101, convert_element_type_154);  mul_101 = convert_element_type_154 = None
        mul_98: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, convert_element_type_150);  mul_97 = convert_element_type_150 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:809 in forward, code: convert_element_type_155 = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        convert_element_type_151: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:810 in forward, code: add_55 = torch.ops.aten.add.Tensor(mul_87, convert_element_type_155);  mul_87 = convert_element_type_155 = None
        add_50: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, convert_element_type_151);  mul_83 = convert_element_type_151 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:811 in forward, code: convert_element_type_156 = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        convert_element_type_152: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:812 in forward, code: pow_53 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_156, 2)
        pow_53: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_152, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:813 in forward, code: mean_26 = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None
        mean_26: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:814 in forward, code: add_56 = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        add_51: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:815 in forward, code: pow_54 = torch.ops.aten.pow.Tensor_Scalar(add_56, -0.5);  add_56 = None
        pow_54: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_51, -0.5);  add_51 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:816 in forward, code: mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_156, pow_54);  convert_element_type_156 = pow_54 = None
        mul_99: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_152, pow_54);  convert_element_type_152 = pow_54 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:817 in forward, code: convert_element_type_157 = torch.ops.prims.convert_element_type.default(arg55_1, torch.float32);  arg55_1 = None
        convert_element_type_153: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg55_1, torch.float32);  arg55_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:818 in forward, code: mul_104 = torch.ops.aten.mul.Tensor(mul_103, convert_element_type_157);  mul_103 = convert_element_type_157 = None
        mul_100: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, convert_element_type_153);  mul_99 = convert_element_type_153 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:819 in forward, code: convert_element_type_158 = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        convert_element_type_154: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16);  mul_100 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:821 in forward, code: view_100 = torch.ops.aten.view.default(convert_element_type_158, [2048, 5376])
        view_100: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_154, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:820 in forward, code: permute_45 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        permute_45: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:822 in forward, code: mm_25 = torch.ops.aten.mm.default(view_100, permute_45);  view_100 = permute_45 = None
        mm_25: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_100, permute_45);  view_100 = permute_45 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:823 in forward, code: view_101 = torch.ops.aten.view.default(mm_25, [1, 2048, 21504]);  mm_25 = None
        view_101: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [1, 2048, 21504]);  mm_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:824 in forward, code: convert_element_type_161 = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        convert_element_type_157: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:830 in forward, code: mul_109 = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.5);  convert_element_type_161 = None
        mul_105: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.5)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:825 in forward, code: mul_105 = torch.ops.aten.mul.Tensor(convert_element_type_161, convert_element_type_161)
        mul_101: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, convert_element_type_157)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:826 in forward, code: mul_106 = torch.ops.aten.mul.Tensor(mul_105, convert_element_type_161);  mul_105 = None
        mul_102: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, convert_element_type_157);  mul_101 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:827 in forward, code: mul_107 = torch.ops.aten.mul.Tensor(mul_106, 0.044715);  mul_106 = None
        mul_103: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, 0.044715);  mul_102 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:828 in forward, code: add_57 = torch.ops.aten.add.Tensor(convert_element_type_161, mul_107);  mul_107 = None
        add_52: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_157, mul_103);  convert_element_type_157 = mul_103 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:829 in forward, code: mul_108 = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        mul_104: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_52, 0.7978845608028654);  add_52 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:831 in forward, code: tanh_3 = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        tanh_3: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_104);  mul_104 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:832 in forward, code: add_58 = torch.ops.aten.add.Tensor(tanh_3, 1);  tanh_3 = None
        add_53: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1);  tanh_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:833 in forward, code: mul_110 = torch.ops.aten.mul.Tensor(mul_109, add_58);  mul_109 = add_58 = None
        mul_106: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, add_53);  mul_105 = add_53 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:834 in forward, code: convert_element_type_162 = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        convert_element_type_158: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_106, torch.bfloat16);  mul_106 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:836 in forward, code: view_102 = torch.ops.aten.view.default(convert_element_type_158, [2048, 5376]);  convert_element_type_158 = None
        view_102: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_154, [2048, 5376]);  convert_element_type_154 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:835 in forward, code: permute_46 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        permute_46: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:837 in forward, code: mm_26 = torch.ops.aten.mm.default(view_102, permute_46);  view_102 = permute_46 = None
        mm_26: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_102, permute_46);  view_102 = permute_46 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:838 in forward, code: view_103 = torch.ops.aten.view.default(mm_26, [1, 2048, 21504]);  mm_26 = None
        view_103: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [1, 2048, 21504]);  mm_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:839 in forward, code: mul_111 = torch.ops.aten.mul.Tensor(convert_element_type_162, view_103);  convert_element_type_162 = view_103 = None
        mul_107: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_158, view_103);  convert_element_type_158 = view_103 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:841 in forward, code: view_104 = torch.ops.aten.view.default(mul_111, [2048, 21504]);  mul_111 = None
        view_104: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_107, [2048, 21504]);  mul_107 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:840 in forward, code: permute_47 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        permute_47: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:842 in forward, code: mm_27 = torch.ops.aten.mm.default(view_104, permute_47);  view_104 = permute_47 = None
        mm_27: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_104, permute_47);  view_104 = permute_47 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:843 in forward, code: view_105 = torch.ops.aten.view.default(mm_27, [1, 2048, 5376]);  mm_27 = None
        view_105: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [1, 2048, 5376]);  mm_27 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:844 in forward, code: convert_element_type_167 = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        convert_element_type_163: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:845 in forward, code: pow_55 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_167, 2)
        pow_55: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_163, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:846 in forward, code: mean_27 = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        mean_27: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:847 in forward, code: add_59 = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        add_54: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:848 in forward, code: pow_56 = torch.ops.aten.pow.Tensor_Scalar(add_59, -0.5);  add_59 = None
        pow_56: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_54, -0.5);  add_54 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:849 in forward, code: mul_112 = torch.ops.aten.mul.Tensor(convert_element_type_167, pow_56);  convert_element_type_167 = pow_56 = None
        mul_108: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_163, pow_56);  convert_element_type_163 = pow_56 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:850 in forward, code: convert_element_type_168 = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None
        convert_element_type_164: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:851 in forward, code: mul_113 = torch.ops.aten.mul.Tensor(mul_112, convert_element_type_168);  mul_112 = convert_element_type_168 = None
        mul_109: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, convert_element_type_164);  mul_108 = convert_element_type_164 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:852 in forward, code: convert_element_type_169 = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        convert_element_type_165: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:853 in forward, code: add_60 = torch.ops.aten.add.Tensor(add_55, convert_element_type_169);  add_55 = convert_element_type_169 = None
        add_55: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_50, convert_element_type_165);  add_50 = convert_element_type_165 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:854 in forward, code: mul_114 = torch.ops.aten.mul.Tensor(add_60, arg60_1);  add_60 = arg60_1 = None
        mul_110: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, arg60_1);  add_55 = arg60_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:855 in forward, code: convert_element_type_170 = torch.ops.prims.convert_element_type.default(mul_114, torch.float32)
        convert_element_type_166: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_110, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:856 in forward, code: pow_57 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_170, 2)
        pow_57: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_166, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:857 in forward, code: mean_28 = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None
        mean_28: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:858 in forward, code: add_61 = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        add_56: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:859 in forward, code: pow_58 = torch.ops.aten.pow.Tensor_Scalar(add_61, -0.5);  add_61 = None
        pow_58: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_56, -0.5);  add_56 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:860 in forward, code: mul_115 = torch.ops.aten.mul.Tensor(convert_element_type_170, pow_58);  convert_element_type_170 = pow_58 = None
        mul_111: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_166, pow_58);  convert_element_type_166 = pow_58 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:861 in forward, code: convert_element_type_171 = torch.ops.prims.convert_element_type.default(arg61_1, torch.float32);  arg61_1 = None
        convert_element_type_167: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg61_1, torch.float32);  arg61_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:862 in forward, code: mul_116 = torch.ops.aten.mul.Tensor(mul_115, convert_element_type_171);  mul_115 = convert_element_type_171 = None
        mul_112: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, convert_element_type_167);  mul_111 = convert_element_type_167 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:863 in forward, code: convert_element_type_172 = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None
        convert_element_type_168: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:865 in forward, code: view_106 = torch.ops.aten.view.default(convert_element_type_172, [2048, 5376])
        view_106: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_168, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:864 in forward, code: permute_48 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        permute_48: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:866 in forward, code: mm_28 = torch.ops.aten.mm.default(view_106, permute_48);  view_106 = permute_48 = None
        mm_28: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_106, permute_48);  view_106 = permute_48 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:867 in forward, code: view_107 = torch.ops.aten.view.default(mm_28, [1, 2048, 8192]);  mm_28 = None
        view_107: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [1, 2048, 8192]);  mm_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:868 in forward, code: view_108 = torch.ops.aten.view.default(view_107, [1, 2048, -1, 256]);  view_107 = None
        view_108: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_107, [1, 2048, -1, 256]);  view_107 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:869 in forward, code: convert_element_type_175 = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        convert_element_type_171: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:870 in forward, code: pow_59 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_175, 2)
        pow_59: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_171, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:871 in forward, code: mean_29 = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None
        mean_29: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:872 in forward, code: add_62 = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        add_57: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:873 in forward, code: pow_60 = torch.ops.aten.pow.Tensor_Scalar(add_62, -0.5);  add_62 = None
        pow_60: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_57, -0.5);  add_57 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:874 in forward, code: mul_117 = torch.ops.aten.mul.Tensor(convert_element_type_175, pow_60);  convert_element_type_175 = pow_60 = None
        mul_113: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_171, pow_60);  convert_element_type_171 = pow_60 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:875 in forward, code: convert_element_type_176 = torch.ops.prims.convert_element_type.default(arg63_1, torch.float32);  arg63_1 = None
        convert_element_type_172: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg63_1, torch.float32);  arg63_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:876 in forward, code: mul_118 = torch.ops.aten.mul.Tensor(mul_117, convert_element_type_176);  mul_117 = convert_element_type_176 = None
        mul_114: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, convert_element_type_172);  mul_113 = convert_element_type_172 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:877 in forward, code: convert_element_type_177 = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        convert_element_type_173: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_114, torch.bfloat16);  mul_114 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:878 in forward, code: unsqueeze_47 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)
        unsqueeze_47: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:880 in forward, code: mul_119 = torch.ops.aten.mul.Tensor(convert_element_type_177, unsqueeze_47);  unsqueeze_47 = None
        mul_115: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_173, unsqueeze_47);  unsqueeze_47 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:882 in forward, code: slice_24 = torch.ops.aten.slice.Tensor(convert_element_type_177, 3, 128, 9223372036854775807);  convert_element_type_177 = None
        slice_24: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_173, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:883 in forward, code: neg_8 = torch.ops.aten.neg.default(slice_24);  slice_24 = None
        neg_8: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_24);  slice_24 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:881 in forward, code: slice_23 = torch.ops.aten.slice.Tensor(convert_element_type_177, 3, 0, 128)
        slice_23: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_173, 3, 0, 128);  convert_element_type_173 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:884 in forward, code: cat_10 = torch.ops.aten.cat.default([neg_8, slice_23], -1);  neg_8 = slice_23 = None
        cat_10: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_8, slice_23], -1);  neg_8 = slice_23 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:879 in forward, code: unsqueeze_48 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)
        unsqueeze_48: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:885 in forward, code: mul_120 = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_48);  cat_10 = unsqueeze_48 = None
        mul_116: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_48);  cat_10 = unsqueeze_48 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:886 in forward, code: add_63 = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
        add_58: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:887 in forward, code: permute_49 = torch.ops.aten.permute.default(add_63, [0, 2, 1, 3]);  add_63 = None
        permute_49: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_58, [0, 2, 1, 3]);  add_58 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:936 in forward, code: convert_element_type_187 = torch.ops.prims.convert_element_type.default(permute_49, torch.float32);  permute_49 = None
        convert_element_type_183: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_49, torch.float32);  permute_49 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:939 in forward, code: mul_126 = torch.ops.aten.mul.Scalar(convert_element_type_187, 1.0);  convert_element_type_187 = None
        mul_122: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(convert_element_type_183, 1.0);  convert_element_type_183 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:942 in forward, code: expand_30 = torch.ops.aten.expand.default(mul_126, [1, 32, 2048, 256]);  mul_126 = None
        expand_30: "f32[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.expand.default(mul_122, [1, 32, 2048, 256]);  mul_122 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:943 in forward, code: view_117 = torch.ops.aten.view.default(expand_30, [32, 2048, 256]);  expand_30 = None
        view_117: "f32[32, 2048, 256][256, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(expand_30, [32, 2048, 256]);  expand_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:889 in forward, code: view_109 = torch.ops.aten.view.default(convert_element_type_172, [2048, 5376])
        view_109: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_168, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:888 in forward, code: permute_50 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        permute_50: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:890 in forward, code: mm_29 = torch.ops.aten.mm.default(view_109, permute_50);  view_109 = permute_50 = None
        mm_29: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_109, permute_50);  view_109 = permute_50 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:891 in forward, code: view_110 = torch.ops.aten.view.default(mm_29, [1, 2048, 4096]);  mm_29 = None
        view_110: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [1, 2048, 4096]);  mm_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:892 in forward, code: view_111 = torch.ops.aten.view.default(view_110, [1, 2048, -1, 256]);  view_110 = None
        view_111: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_110, [1, 2048, -1, 256]);  view_110 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:898 in forward, code: convert_element_type_182 = torch.ops.prims.convert_element_type.default(view_111, torch.float32);  view_111 = None
        convert_element_type_178: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_111, torch.float32);  view_111 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:899 in forward, code: pow_61 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_182, 2)
        pow_61: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_178, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:900 in forward, code: mean_30 = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None
        mean_30: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:901 in forward, code: add_64 = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        add_59: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:902 in forward, code: pow_62 = torch.ops.aten.pow.Tensor_Scalar(add_64, -0.5);  add_64 = None
        pow_62: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_59, -0.5);  add_59 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:903 in forward, code: mul_121 = torch.ops.aten.mul.Tensor(convert_element_type_182, pow_62);  convert_element_type_182 = pow_62 = None
        mul_117: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_178, pow_62);  convert_element_type_178 = pow_62 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:904 in forward, code: convert_element_type_183 = torch.ops.prims.convert_element_type.default(arg66_1, torch.float32);  arg66_1 = None
        convert_element_type_179: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg66_1, torch.float32);  arg66_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:905 in forward, code: mul_122 = torch.ops.aten.mul.Tensor(mul_121, convert_element_type_183);  mul_121 = convert_element_type_183 = None
        mul_118: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, convert_element_type_179);  mul_117 = convert_element_type_179 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:906 in forward, code: convert_element_type_184 = torch.ops.prims.convert_element_type.default(mul_122, torch.bfloat16);  mul_122 = None
        convert_element_type_180: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:907 in forward, code: unsqueeze_49 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2);  convert_element_type_1 = None
        unsqueeze_49: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2);  convert_element_type_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:909 in forward, code: mul_123 = torch.ops.aten.mul.Tensor(convert_element_type_184, unsqueeze_49);  unsqueeze_49 = None
        mul_119: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_180, unsqueeze_49);  unsqueeze_49 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:911 in forward, code: slice_26 = torch.ops.aten.slice.Tensor(convert_element_type_184, 3, 128, 9223372036854775807);  convert_element_type_184 = None
        slice_26: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_180, 3, 128, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:912 in forward, code: neg_9 = torch.ops.aten.neg.default(slice_26);  slice_26 = None
        neg_9: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_26);  slice_26 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:910 in forward, code: slice_25 = torch.ops.aten.slice.Tensor(convert_element_type_184, 3, 0, 128)
        slice_25: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_180, 3, 0, 128);  convert_element_type_180 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:913 in forward, code: cat_11 = torch.ops.aten.cat.default([neg_9, slice_25], -1);  neg_9 = slice_25 = None
        cat_11: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_9, slice_25], -1);  neg_9 = slice_25 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:908 in forward, code: unsqueeze_50 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2);  convert_element_type_2 = None
        unsqueeze_50: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2);  convert_element_type_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:914 in forward, code: mul_124 = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_50);  cat_11 = unsqueeze_50 = None
        mul_120: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_50);  cat_11 = unsqueeze_50 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:915 in forward, code: add_65 = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        add_60: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:916 in forward, code: permute_52 = torch.ops.aten.permute.default(add_65, [0, 2, 1, 3]);  add_65 = None
        permute_52: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_60, [0, 2, 1, 3]);  add_60 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:925 in forward, code: unsqueeze_51 = torch.ops.aten.unsqueeze.default(permute_52, 2);  permute_52 = None
        unsqueeze_51: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_52, 2);  permute_52 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:926 in forward, code: expand_28 = torch.ops.aten.expand.default(unsqueeze_51, [1, 16, 2, 2048, 256]);  unsqueeze_51 = None
        expand_28: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_51, [1, 16, 2, 2048, 256]);  unsqueeze_51 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:927 in forward, code: clone_16 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        clone_14: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:928 in forward, code: view_115 = torch.ops.aten.view.default(clone_16, [1, 32, 2048, 256]);  clone_16 = None
        view_115: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [1, 32, 2048, 256]);  clone_14 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:937 in forward, code: convert_element_type_188 = torch.ops.prims.convert_element_type.default(view_115, torch.float32);  view_115 = None
        convert_element_type_184: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_115, torch.float32);  view_115 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:940 in forward, code: permute_54 = torch.ops.aten.permute.default(convert_element_type_188, [0, 1, 3, 2]);  convert_element_type_188 = None
        permute_54: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.permute.default(convert_element_type_184, [0, 1, 3, 2]);  convert_element_type_184 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:941 in forward, code: mul_127 = torch.ops.aten.mul.Scalar(permute_54, 1.0);  permute_54 = None
        mul_123: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.mul.Scalar(permute_54, 1.0);  permute_54 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:944 in forward, code: expand_31 = torch.ops.aten.expand.default(mul_127, [1, 32, 256, 2048]);  mul_127 = None
        expand_31: "f32[1, 32, 256, 2048][16777216, 524288, 1, 256]cuda:0" = torch.ops.aten.expand.default(mul_123, [1, 32, 256, 2048]);  mul_123 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:945 in forward, code: view_118 = torch.ops.aten.view.default(expand_31, [32, 256, 2048]);  expand_31 = None
        view_118: "f32[32, 256, 2048][524288, 1, 256]cuda:0" = torch.ops.aten.reshape.default(expand_31, [32, 256, 2048]);  expand_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:946 in forward, code: bmm_8 = torch.ops.aten.bmm.default(view_117, view_118);  view_117 = view_118 = None
        bmm_8: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.bmm.default(view_117, view_118);  view_117 = view_118 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:947 in forward, code: view_119 = torch.ops.aten.view.default(bmm_8, [1, 32, 2048, 2048]);  bmm_8 = None
        view_119: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [1, 32, 2048, 2048]);  bmm_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:934 in forward, code: full_default_16 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:933 in forward, code: full_default_15 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:935 in forward, code: where_8 = torch.ops.aten.where.self(expand_1, full_default_16, full_default_15);  expand_1 = full_default_16 = full_default_15 = None
        where_8: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_16, full_default_15);  expand_1 = full_default_16 = full_default_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:948 in forward, code: add_67 = torch.ops.aten.add.Tensor(view_119, where_8);  view_119 = where_8 = None
        add_62: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_119, where_8);  view_119 = where_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:954 in forward, code: eq_6 = torch.ops.aten.eq.Scalar(add_67, -inf);  add_67 = None
        eq_6: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Scalar(add_62, -inf)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:955 in forward, code: logical_not_8 = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        logical_not_8: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:956 in forward, code: any_5 = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        any_5: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:957 in forward, code: logical_not_9 = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        logical_not_9: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_5);  any_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:958 in forward, code: full_default_17 = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:949 in forward, code: amax_4 = torch.ops.aten.amax.default(add_67, [-1], True)
        amax_4: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_62, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:950 in forward, code: sub_9 = torch.ops.aten.sub.Tensor(add_67, amax_4);  amax_4 = None
        sub_9: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_62, amax_4);  add_62 = amax_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:951 in forward, code: exp_4 = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        exp_4: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:952 in forward, code: sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        sum_5: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:953 in forward, code: div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        div_4: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:959 in forward, code: where_9 = torch.ops.aten.where.self(logical_not_9, full_default_17, div_4);  logical_not_9 = full_default_17 = div_4 = None
        where_9: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(logical_not_9, full_default_17, div_4);  logical_not_9 = full_default_17 = div_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:960 in forward, code: expand_32 = torch.ops.aten.expand.default(where_9, [1, 32, 2048, 2048]);  where_9 = None
        expand_32: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_9, [1, 32, 2048, 2048]);  where_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:961 in forward, code: view_120 = torch.ops.aten.view.default(expand_32, [32, 2048, 2048]);  expand_32 = None
        view_120: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(expand_32, [32, 2048, 2048]);  expand_32 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:894 in forward, code: view_112 = torch.ops.aten.view.default(convert_element_type_172, [2048, 5376]);  convert_element_type_172 = None
        view_112: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_168, [2048, 5376]);  convert_element_type_168 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:893 in forward, code: permute_51 = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        permute_51: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:895 in forward, code: mm_30 = torch.ops.aten.mm.default(view_112, permute_51);  view_112 = permute_51 = None
        mm_30: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_112, permute_51);  view_112 = permute_51 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:896 in forward, code: view_113 = torch.ops.aten.view.default(mm_30, [1, 2048, 4096]);  mm_30 = None
        view_113: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [1, 2048, 4096]);  mm_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:897 in forward, code: view_114 = torch.ops.aten.view.default(view_113, [1, 2048, -1, 256]);  view_113 = None
        view_114: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [1, 2048, -1, 256]);  view_113 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:917 in forward, code: convert_element_type_185 = torch.ops.prims.convert_element_type.default(view_114, torch.float32);  view_114 = None
        convert_element_type_181: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_114, torch.float32);  view_114 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:918 in forward, code: pow_63 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_185, 2)
        pow_63: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_181, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:919 in forward, code: mean_31 = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None
        mean_31: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:920 in forward, code: add_66 = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        add_61: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:921 in forward, code: pow_64 = torch.ops.aten.pow.Tensor_Scalar(add_66, -0.5);  add_66 = None
        pow_64: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_61, -0.5);  add_61 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:922 in forward, code: mul_125 = torch.ops.aten.mul.Tensor(convert_element_type_185, pow_64);  convert_element_type_185 = pow_64 = None
        mul_121: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_181, pow_64);  convert_element_type_181 = pow_64 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:923 in forward, code: convert_element_type_186 = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        convert_element_type_182: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:924 in forward, code: permute_53 = torch.ops.aten.permute.default(convert_element_type_186, [0, 2, 1, 3]);  convert_element_type_186 = None
        permute_53: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_182, [0, 2, 1, 3]);  convert_element_type_182 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:929 in forward, code: unsqueeze_52 = torch.ops.aten.unsqueeze.default(permute_53, 2);  permute_53 = None
        unsqueeze_52: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_53, 2);  permute_53 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:930 in forward, code: expand_29 = torch.ops.aten.expand.default(unsqueeze_52, [1, 16, 2, 2048, 256]);  unsqueeze_52 = None
        expand_29: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_52, [1, 16, 2, 2048, 256]);  unsqueeze_52 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:931 in forward, code: clone_17 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        clone_15: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:932 in forward, code: view_116 = torch.ops.aten.view.default(clone_17, [1, 32, 2048, 256]);  clone_17 = None
        view_116: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [1, 32, 2048, 256]);  clone_15 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:938 in forward, code: convert_element_type_189 = torch.ops.prims.convert_element_type.default(view_116, torch.float32);  view_116 = None
        convert_element_type_185: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_116, torch.float32);  view_116 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:962 in forward, code: expand_33 = torch.ops.aten.expand.default(convert_element_type_189, [1, 32, 2048, 256]);  convert_element_type_189 = None
        expand_33: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_185, [1, 32, 2048, 256]);  convert_element_type_185 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:963 in forward, code: view_121 = torch.ops.aten.view.default(expand_33, [32, 2048, 256]);  expand_33 = None
        view_121: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(expand_33, [32, 2048, 256]);  expand_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:964 in forward, code: bmm_9 = torch.ops.aten.bmm.default(view_120, view_121);  view_120 = view_121 = None
        bmm_9: "f32[32, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_120, view_121);  view_120 = view_121 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:965 in forward, code: view_122 = torch.ops.aten.view.default(bmm_9, [1, 32, 2048, 256]);  bmm_9 = None
        view_122: "f32[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_9, [1, 32, 2048, 256]);  bmm_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:966 in forward, code: convert_element_type_191 = torch.ops.prims.convert_element_type.default(view_122, torch.bfloat16);  view_122 = None
        convert_element_type_186: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_122, torch.bfloat16);  view_122 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:967 in forward, code: permute_55 = torch.ops.aten.permute.default(convert_element_type_191, [0, 2, 1, 3]);  convert_element_type_191 = None
        permute_55: "bf16[1, 2048, 32, 256][16777216, 256, 524288, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_186, [0, 2, 1, 3]);  convert_element_type_186 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:968 in forward, code: clone_18 = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        clone_16: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:969 in forward, code: view_123 = torch.ops.aten.view.default(clone_18, [1, 2048, -1]);  clone_18 = None
        view_123: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1, 2048, -1]);  clone_16 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:971 in forward, code: view_124 = torch.ops.aten.view.default(view_123, [2048, 8192]);  view_123 = None
        view_124: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_123, [2048, 8192]);  view_123 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:970 in forward, code: permute_56 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        permute_56: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:972 in forward, code: mm_31 = torch.ops.aten.mm.default(view_124, permute_56);  view_124 = permute_56 = None
        mm_31: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_124, permute_56);  view_124 = permute_56 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:973 in forward, code: view_125 = torch.ops.aten.view.default(mm_31, [1, 2048, 5376]);  mm_31 = None
        view_125: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [1, 2048, 5376]);  mm_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:974 in forward, code: convert_element_type_194 = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        convert_element_type_189: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:975 in forward, code: pow_65 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_194, 2)
        pow_65: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_189, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:976 in forward, code: mean_32 = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None
        mean_32: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:977 in forward, code: add_68 = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        add_63: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:978 in forward, code: pow_66 = torch.ops.aten.pow.Tensor_Scalar(add_68, -0.5);  add_68 = None
        pow_66: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_63, -0.5);  add_63 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:979 in forward, code: mul_128 = torch.ops.aten.mul.Tensor(convert_element_type_194, pow_66);  convert_element_type_194 = pow_66 = None
        mul_124: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_189, pow_66);  convert_element_type_189 = pow_66 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:980 in forward, code: convert_element_type_195 = torch.ops.prims.convert_element_type.default(arg68_1, torch.float32);  arg68_1 = None
        convert_element_type_190: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg68_1, torch.float32);  arg68_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:981 in forward, code: mul_129 = torch.ops.aten.mul.Tensor(mul_128, convert_element_type_195);  mul_128 = convert_element_type_195 = None
        mul_125: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, convert_element_type_190);  mul_124 = convert_element_type_190 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:982 in forward, code: convert_element_type_196 = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        convert_element_type_191: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:983 in forward, code: add_69 = torch.ops.aten.add.Tensor(mul_114, convert_element_type_196);  mul_114 = convert_element_type_196 = None
        add_64: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_110, convert_element_type_191);  mul_110 = convert_element_type_191 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:984 in forward, code: convert_element_type_197 = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        convert_element_type_192: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:985 in forward, code: pow_67 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_197, 2)
        pow_67: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_192, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:986 in forward, code: mean_33 = torch.ops.aten.mean.dim(pow_67, [-1], True);  pow_67 = None
        mean_33: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_67, [-1], True);  pow_67 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:987 in forward, code: add_70 = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        add_65: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:988 in forward, code: pow_68 = torch.ops.aten.pow.Tensor_Scalar(add_70, -0.5);  add_70 = None
        pow_68: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_65, -0.5);  add_65 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:989 in forward, code: mul_130 = torch.ops.aten.mul.Tensor(convert_element_type_197, pow_68);  convert_element_type_197 = pow_68 = None
        mul_126: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_192, pow_68);  convert_element_type_192 = pow_68 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:990 in forward, code: convert_element_type_198 = torch.ops.prims.convert_element_type.default(arg69_1, torch.float32);  arg69_1 = None
        convert_element_type_193: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg69_1, torch.float32);  arg69_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:991 in forward, code: mul_131 = torch.ops.aten.mul.Tensor(mul_130, convert_element_type_198);  mul_130 = convert_element_type_198 = None
        mul_127: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, convert_element_type_193);  mul_126 = convert_element_type_193 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:992 in forward, code: convert_element_type_199 = torch.ops.prims.convert_element_type.default(mul_131, torch.bfloat16);  mul_131 = None
        convert_element_type_194: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16);  mul_127 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:994 in forward, code: view_126 = torch.ops.aten.view.default(convert_element_type_199, [2048, 5376])
        view_126: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_194, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:993 in forward, code: permute_57 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        permute_57: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:995 in forward, code: mm_32 = torch.ops.aten.mm.default(view_126, permute_57);  view_126 = permute_57 = None
        mm_32: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_126, permute_57);  view_126 = permute_57 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:996 in forward, code: view_127 = torch.ops.aten.view.default(mm_32, [1, 2048, 21504]);  mm_32 = None
        view_127: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [1, 2048, 21504]);  mm_32 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:997 in forward, code: convert_element_type_202 = torch.ops.prims.convert_element_type.default(view_127, torch.float32);  view_127 = None
        convert_element_type_197: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_127, torch.float32);  view_127 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1003 in forward, code: mul_136 = torch.ops.aten.mul.Tensor(convert_element_type_202, 0.5);  convert_element_type_202 = None
        mul_132: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, 0.5)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:998 in forward, code: mul_132 = torch.ops.aten.mul.Tensor(convert_element_type_202, convert_element_type_202)
        mul_128: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, convert_element_type_197)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:999 in forward, code: mul_133 = torch.ops.aten.mul.Tensor(mul_132, convert_element_type_202);  mul_132 = None
        mul_129: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, convert_element_type_197);  mul_128 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1000 in forward, code: mul_134 = torch.ops.aten.mul.Tensor(mul_133, 0.044715);  mul_133 = None
        mul_130: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 0.044715);  mul_129 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1001 in forward, code: add_71 = torch.ops.aten.add.Tensor(convert_element_type_202, mul_134);  mul_134 = None
        add_66: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_197, mul_130);  convert_element_type_197 = mul_130 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1002 in forward, code: mul_135 = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        mul_131: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_66, 0.7978845608028654);  add_66 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1004 in forward, code: tanh_4 = torch.ops.aten.tanh.default(mul_135);  mul_135 = None
        tanh_4: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_131);  mul_131 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1005 in forward, code: add_72 = torch.ops.aten.add.Tensor(tanh_4, 1);  tanh_4 = None
        add_67: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1);  tanh_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1006 in forward, code: mul_137 = torch.ops.aten.mul.Tensor(mul_136, add_72);  mul_136 = add_72 = None
        mul_133: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, add_67);  mul_132 = add_67 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1007 in forward, code: convert_element_type_203 = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        convert_element_type_198: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1009 in forward, code: view_128 = torch.ops.aten.view.default(convert_element_type_199, [2048, 5376]);  convert_element_type_199 = None
        view_128: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_194, [2048, 5376]);  convert_element_type_194 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1008 in forward, code: permute_58 = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        permute_58: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1010 in forward, code: mm_33 = torch.ops.aten.mm.default(view_128, permute_58);  view_128 = permute_58 = None
        mm_33: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_128, permute_58);  view_128 = permute_58 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1011 in forward, code: view_129 = torch.ops.aten.view.default(mm_33, [1, 2048, 21504]);  mm_33 = None
        view_129: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [1, 2048, 21504]);  mm_33 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1012 in forward, code: mul_138 = torch.ops.aten.mul.Tensor(convert_element_type_203, view_129);  convert_element_type_203 = view_129 = None
        mul_134: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, view_129);  convert_element_type_198 = view_129 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1014 in forward, code: view_130 = torch.ops.aten.view.default(mul_138, [2048, 21504]);  mul_138 = None
        view_130: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_134, [2048, 21504]);  mul_134 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1013 in forward, code: permute_59 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        permute_59: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1015 in forward, code: mm_34 = torch.ops.aten.mm.default(view_130, permute_59);  view_130 = permute_59 = None
        mm_34: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_130, permute_59);  view_130 = permute_59 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1016 in forward, code: view_131 = torch.ops.aten.view.default(mm_34, [1, 2048, 5376]);  mm_34 = None
        view_131: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [1, 2048, 5376]);  mm_34 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1017 in forward, code: convert_element_type_208 = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        convert_element_type_203: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1018 in forward, code: pow_69 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_208, 2)
        pow_69: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_203, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1019 in forward, code: mean_34 = torch.ops.aten.mean.dim(pow_69, [-1], True);  pow_69 = None
        mean_34: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_69, [-1], True);  pow_69 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1020 in forward, code: add_73 = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        add_68: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1021 in forward, code: pow_70 = torch.ops.aten.pow.Tensor_Scalar(add_73, -0.5);  add_73 = None
        pow_70: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_68, -0.5);  add_68 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1022 in forward, code: mul_139 = torch.ops.aten.mul.Tensor(convert_element_type_208, pow_70);  convert_element_type_208 = pow_70 = None
        mul_135: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, pow_70);  convert_element_type_203 = pow_70 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1023 in forward, code: convert_element_type_209 = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None
        convert_element_type_204: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1024 in forward, code: mul_140 = torch.ops.aten.mul.Tensor(mul_139, convert_element_type_209);  mul_139 = convert_element_type_209 = None
        mul_136: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, convert_element_type_204);  mul_135 = convert_element_type_204 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1025 in forward, code: convert_element_type_210 = torch.ops.prims.convert_element_type.default(mul_140, torch.bfloat16);  mul_140 = None
        convert_element_type_205: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1026 in forward, code: add_74 = torch.ops.aten.add.Tensor(add_69, convert_element_type_210);  add_69 = convert_element_type_210 = None
        add_69: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_64, convert_element_type_205);  add_64 = convert_element_type_205 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1027 in forward, code: mul_141 = torch.ops.aten.mul.Tensor(add_74, arg74_1);  add_74 = arg74_1 = None
        mul_137: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, arg74_1);  add_69 = arg74_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1028 in forward, code: convert_element_type_211 = torch.ops.prims.convert_element_type.default(mul_141, torch.float32)
        convert_element_type_206: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_137, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1029 in forward, code: pow_71 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_211, 2)
        pow_71: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_206, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1030 in forward, code: mean_35 = torch.ops.aten.mean.dim(pow_71, [-1], True);  pow_71 = None
        mean_35: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_71, [-1], True);  pow_71 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1031 in forward, code: add_75 = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        add_70: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1032 in forward, code: pow_72 = torch.ops.aten.pow.Tensor_Scalar(add_75, -0.5);  add_75 = None
        pow_72: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_70, -0.5);  add_70 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1033 in forward, code: mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_211, pow_72);  convert_element_type_211 = pow_72 = None
        mul_138: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_206, pow_72);  convert_element_type_206 = pow_72 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1034 in forward, code: convert_element_type_212 = torch.ops.prims.convert_element_type.default(arg75_1, torch.float32);  arg75_1 = None
        convert_element_type_207: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg75_1, torch.float32);  arg75_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1035 in forward, code: mul_143 = torch.ops.aten.mul.Tensor(mul_142, convert_element_type_212);  mul_142 = convert_element_type_212 = None
        mul_139: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, convert_element_type_207);  mul_138 = convert_element_type_207 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1036 in forward, code: convert_element_type_213 = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        convert_element_type_208: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_139, torch.bfloat16);  mul_139 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1038 in forward, code: view_132 = torch.ops.aten.view.default(convert_element_type_213, [2048, 5376])
        view_132: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1037 in forward, code: permute_60 = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        permute_60: "bf16[5376, 16384][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1039 in forward, code: mm_35 = torch.ops.aten.mm.default(view_132, permute_60);  view_132 = permute_60 = None
        mm_35: "bf16[2048, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_132, permute_60);  view_132 = permute_60 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1040 in forward, code: view_133 = torch.ops.aten.view.default(mm_35, [1, 2048, 16384]);  mm_35 = None
        view_133: "bf16[1, 2048, 16384][33554432, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [1, 2048, 16384]);  mm_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1041 in forward, code: view_134 = torch.ops.aten.view.default(view_133, [1, 2048, -1, 512]);  view_133 = None
        view_134: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [1, 2048, -1, 512]);  view_133 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1042 in forward, code: convert_element_type_216 = torch.ops.prims.convert_element_type.default(view_134, torch.float32);  view_134 = None
        convert_element_type_211: "f32[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_134, torch.float32);  view_134 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1043 in forward, code: pow_73 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_216, 2)
        pow_73: "f32[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_211, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1044 in forward, code: mean_36 = torch.ops.aten.mean.dim(pow_73, [-1], True);  pow_73 = None
        mean_36: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_73, [-1], True);  pow_73 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1045 in forward, code: add_76 = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        add_71: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1046 in forward, code: pow_74 = torch.ops.aten.pow.Tensor_Scalar(add_76, -0.5);  add_76 = None
        pow_74: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_71, -0.5);  add_71 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1047 in forward, code: mul_144 = torch.ops.aten.mul.Tensor(convert_element_type_216, pow_74);  convert_element_type_216 = pow_74 = None
        mul_140: "f32[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, pow_74);  convert_element_type_211 = pow_74 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1048 in forward, code: convert_element_type_217 = torch.ops.prims.convert_element_type.default(arg77_1, torch.float32);  arg77_1 = None
        convert_element_type_212: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg77_1, torch.float32);  arg77_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1049 in forward, code: mul_145 = torch.ops.aten.mul.Tensor(mul_144, convert_element_type_217);  mul_144 = convert_element_type_217 = None
        mul_141: "f32[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, convert_element_type_212);  mul_140 = convert_element_type_212 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1050 in forward, code: convert_element_type_218 = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None
        convert_element_type_213: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:152 in forward, code: unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_21: "i64[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:153 in forward, code: convert_element_type_3 = torch.ops.prims.convert_element_type.default(unsqueeze_21, torch.float32);  unsqueeze_21 = None
        convert_element_type_3: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_21, torch.float32);  unsqueeze_21 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:154 in forward, code: mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_3, arg4_1);  convert_element_type_3 = arg4_1 = None
        mul_2: "f32[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg4_1);  convert_element_type_3 = arg4_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:155 in forward, code: unsqueeze_22 = torch.ops.aten.unsqueeze.default(mul_4, 2);  mul_4 = None
        unsqueeze_22: "f32[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2, 2);  mul_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:156 in forward, code: expand_3 = torch.ops.aten.expand.default(unsqueeze_22, [1, 2048, 2, 256]);  unsqueeze_22 = None
        expand_3: "f32[1, 2048, 2, 256][524288, 256, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_22, [1, 2048, 2, 256]);  unsqueeze_22 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:157 in forward, code: clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        clone_1: "f32[1, 2048, 2, 256][1048576, 512, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:158 in forward, code: view_1 = torch.ops.aten.view.default(clone_2, [1, 2048, 512]);  clone_2 = None
        view_1: "f32[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [1, 2048, 512]);  clone_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:159 in forward, code: cos_1 = torch.ops.aten.cos.default(view_1)
        cos_1: "f32[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.aten.cos.default(view_1)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:161 in forward, code: convert_element_type_4 = torch.ops.prims.convert_element_type.default(cos_1, torch.bfloat16);  cos_1 = None
        convert_element_type_4: "bf16[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cos_1, torch.bfloat16);  cos_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1051 in forward, code: unsqueeze_53 = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)
        unsqueeze_53: "bf16[1, 2048, 1, 512][1048576, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1053 in forward, code: mul_146 = torch.ops.aten.mul.Tensor(convert_element_type_218, unsqueeze_53);  unsqueeze_53 = None
        mul_142: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_213, unsqueeze_53);  unsqueeze_53 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1055 in forward, code: slice_28 = torch.ops.aten.slice.Tensor(convert_element_type_218, 3, 256, 9223372036854775807);  convert_element_type_218 = None
        slice_28: "bf16[1, 2048, 32, 256][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_213, 3, 256, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1056 in forward, code: neg_10 = torch.ops.aten.neg.default(slice_28);  slice_28 = None
        neg_10: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.neg.default(slice_28);  slice_28 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1054 in forward, code: slice_27 = torch.ops.aten.slice.Tensor(convert_element_type_218, 3, 0, 256)
        slice_27: "bf16[1, 2048, 32, 256][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_213, 3, 0, 256);  convert_element_type_213 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1057 in forward, code: cat_12 = torch.ops.aten.cat.default([neg_10, slice_27], -1);  neg_10 = slice_27 = None
        cat_12: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.cat.default([neg_10, slice_27], -1);  neg_10 = slice_27 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:160 in forward, code: sin_1 = torch.ops.aten.sin.default(view_1);  view_1 = None
        sin_1: "f32[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.aten.sin.default(view_1);  view_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:162 in forward, code: convert_element_type_5 = torch.ops.prims.convert_element_type.default(sin_1, torch.bfloat16);  sin_1 = None
        convert_element_type_5: "bf16[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sin_1, torch.bfloat16);  sin_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1052 in forward, code: unsqueeze_54 = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)
        unsqueeze_54: "bf16[1, 2048, 1, 512][1048576, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1058 in forward, code: mul_147 = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_54);  cat_12 = unsqueeze_54 = None
        mul_143: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_54);  cat_12 = unsqueeze_54 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1059 in forward, code: add_77 = torch.ops.aten.add.Tensor(mul_146, mul_147);  mul_146 = mul_147 = None
        add_72: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1060 in forward, code: permute_61 = torch.ops.aten.permute.default(add_77, [0, 2, 1, 3]);  add_77 = None
        permute_61: "bf16[1, 32, 2048, 512][33554432, 512, 16384, 1]cuda:0" = torch.ops.aten.permute.default(add_72, [0, 2, 1, 3]);  add_72 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1104 in forward, code: convert_element_type_226 = torch.ops.prims.convert_element_type.default(permute_61, torch.float32);  permute_61 = None
        convert_element_type_221: "f32[1, 32, 2048, 512][33554432, 512, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_61, torch.float32);  permute_61 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1107 in forward, code: mul_153 = torch.ops.aten.mul.Scalar(convert_element_type_226, 1.0);  convert_element_type_226 = None
        mul_149: "f32[1, 32, 2048, 512][33554432, 512, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(convert_element_type_221, 1.0);  convert_element_type_221 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1110 in forward, code: expand_36 = torch.ops.aten.expand.default(mul_153, [1, 32, 2048, 512]);  mul_153 = None
        expand_36: "f32[1, 32, 2048, 512][33554432, 512, 16384, 1]cuda:0" = torch.ops.aten.expand.default(mul_149, [1, 32, 2048, 512]);  mul_149 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1111 in forward, code: view_140 = torch.ops.aten.view.default(expand_36, [32, 2048, 512]);  expand_36 = None
        view_140: "f32[32, 2048, 512][512, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(expand_36, [32, 2048, 512]);  expand_36 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1062 in forward, code: view_135 = torch.ops.aten.view.default(convert_element_type_213, [2048, 5376]);  convert_element_type_213 = None
        view_135: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [2048, 5376]);  convert_element_type_208 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1061 in forward, code: permute_62 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        permute_62: "bf16[5376, 2048][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1063 in forward, code: mm_36 = torch.ops.aten.mm.default(view_135, permute_62);  view_135 = permute_62 = None
        mm_36: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_135, permute_62);  view_135 = permute_62 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1064 in forward, code: view_136 = torch.ops.aten.view.default(mm_36, [1, 2048, 2048]);  mm_36 = None
        view_136: "bf16[1, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [1, 2048, 2048]);  mm_36 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1065 in forward, code: view_137 = torch.ops.aten.view.default(view_136, [1, 2048, -1, 512]);  view_136 = None
        view_137: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [1, 2048, -1, 512]);  view_136 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1066 in forward, code: convert_element_type_221 = torch.ops.prims.convert_element_type.default(view_137, torch.float32)
        convert_element_type_216: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1067 in forward, code: pow_75 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_221, 2)
        pow_75: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_216, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1068 in forward, code: mean_37 = torch.ops.aten.mean.dim(pow_75, [-1], True);  pow_75 = None
        mean_37: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_75, [-1], True);  pow_75 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1069 in forward, code: add_78 = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        add_73: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1070 in forward, code: pow_76 = torch.ops.aten.pow.Tensor_Scalar(add_78, -0.5);  add_78 = None
        pow_76: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_73, -0.5);  add_73 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1071 in forward, code: mul_148 = torch.ops.aten.mul.Tensor(convert_element_type_221, pow_76);  convert_element_type_221 = pow_76 = None
        mul_144: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_216, pow_76);  convert_element_type_216 = pow_76 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1072 in forward, code: convert_element_type_222 = torch.ops.prims.convert_element_type.default(arg79_1, torch.float32);  arg79_1 = None
        convert_element_type_217: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg79_1, torch.float32);  arg79_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1073 in forward, code: mul_149 = torch.ops.aten.mul.Tensor(mul_148, convert_element_type_222);  mul_148 = convert_element_type_222 = None
        mul_145: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, convert_element_type_217);  mul_144 = convert_element_type_217 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1074 in forward, code: convert_element_type_223 = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        convert_element_type_218: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1075 in forward, code: unsqueeze_55 = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2);  convert_element_type_4 = None
        unsqueeze_55: "bf16[1, 2048, 1, 512][1048576, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2);  convert_element_type_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1077 in forward, code: mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_223, unsqueeze_55);  unsqueeze_55 = None
        mul_146: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_218, unsqueeze_55);  unsqueeze_55 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1079 in forward, code: slice_30 = torch.ops.aten.slice.Tensor(convert_element_type_223, 3, 256, 9223372036854775807);  convert_element_type_223 = None
        slice_30: "bf16[1, 2048, 4, 256][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_218, 3, 256, 9223372036854775807)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1080 in forward, code: neg_11 = torch.ops.aten.neg.default(slice_30);  slice_30 = None
        neg_11: "bf16[1, 2048, 4, 256][2097152, 1024, 256, 1]cuda:0" = torch.ops.aten.neg.default(slice_30);  slice_30 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1078 in forward, code: slice_29 = torch.ops.aten.slice.Tensor(convert_element_type_223, 3, 0, 256)
        slice_29: "bf16[1, 2048, 4, 256][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_218, 3, 0, 256);  convert_element_type_218 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1081 in forward, code: cat_13 = torch.ops.aten.cat.default([neg_11, slice_29], -1);  neg_11 = slice_29 = None
        cat_13: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.cat.default([neg_11, slice_29], -1);  neg_11 = slice_29 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1076 in forward, code: unsqueeze_56 = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2);  convert_element_type_5 = None
        unsqueeze_56: "bf16[1, 2048, 1, 512][1048576, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2);  convert_element_type_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1082 in forward, code: mul_151 = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_56);  cat_13 = unsqueeze_56 = None
        mul_147: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_56);  cat_13 = unsqueeze_56 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1083 in forward, code: add_79 = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
        add_74: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, mul_147);  mul_146 = mul_147 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1084 in forward, code: permute_63 = torch.ops.aten.permute.default(add_79, [0, 2, 1, 3]);  add_79 = None
        permute_63: "bf16[1, 4, 2048, 512][4194304, 512, 2048, 1]cuda:0" = torch.ops.aten.permute.default(add_74, [0, 2, 1, 3]);  add_74 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1093 in forward, code: unsqueeze_57 = torch.ops.aten.unsqueeze.default(permute_63, 2);  permute_63 = None
        unsqueeze_57: "bf16[1, 4, 1, 2048, 512][4194304, 512, 4194304, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_63, 2);  permute_63 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1094 in forward, code: expand_34 = torch.ops.aten.expand.default(unsqueeze_57, [1, 4, 8, 2048, 512]);  unsqueeze_57 = None
        expand_34: "bf16[1, 4, 8, 2048, 512][4194304, 512, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_57, [1, 4, 8, 2048, 512]);  unsqueeze_57 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1095 in forward, code: clone_19 = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        clone_17: "bf16[1, 4, 8, 2048, 512][33554432, 8388608, 1048576, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1096 in forward, code: view_138 = torch.ops.aten.view.default(clone_19, [1, 32, 2048, 512]);  clone_19 = None
        view_138: "bf16[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1, 32, 2048, 512]);  clone_17 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1105 in forward, code: convert_element_type_227 = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        convert_element_type_222: "f32[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1108 in forward, code: permute_65 = torch.ops.aten.permute.default(convert_element_type_227, [0, 1, 3, 2]);  convert_element_type_227 = None
        permute_65: "f32[1, 32, 512, 2048][33554432, 1048576, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_222, [0, 1, 3, 2]);  convert_element_type_222 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1109 in forward, code: mul_154 = torch.ops.aten.mul.Scalar(permute_65, 1.0);  permute_65 = None
        mul_150: "f32[1, 32, 512, 2048][33554432, 1048576, 1, 512]cuda:0" = torch.ops.aten.mul.Scalar(permute_65, 1.0);  permute_65 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1112 in forward, code: expand_37 = torch.ops.aten.expand.default(mul_154, [1, 32, 512, 2048]);  mul_154 = None
        expand_37: "f32[1, 32, 512, 2048][33554432, 1048576, 1, 512]cuda:0" = torch.ops.aten.expand.default(mul_150, [1, 32, 512, 2048]);  mul_150 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1113 in forward, code: view_141 = torch.ops.aten.view.default(expand_37, [32, 512, 2048]);  expand_37 = None
        view_141: "f32[32, 512, 2048][1048576, 1, 512]cuda:0" = torch.ops.aten.reshape.default(expand_37, [32, 512, 2048]);  expand_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1114 in forward, code: bmm_10 = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
        bmm_10: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1115 in forward, code: view_142 = torch.ops.aten.view.default(bmm_10, [1, 32, 2048, 2048]);  bmm_10 = None
        view_142: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [1, 32, 2048, 2048]);  bmm_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:100 in forward, code: full_default = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:90 in forward, code: iota_4 = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_3: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:97 in forward, code: unsqueeze_7 = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        unsqueeze_7: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_3, 0);  iota_3 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:98 in forward, code: unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_8: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:99 in forward, code: unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_9: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:89 in forward, code: iota_3 = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_2: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:94 in forward, code: unsqueeze_4 = torch.ops.aten.unsqueeze.default(iota_3, 0);  iota_3 = None
        unsqueeze_4: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:95 in forward, code: unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_5: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:96 in forward, code: unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        unsqueeze_6: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:101 in forward, code: le = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)
        le: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:102 in forward, code: bitwise_and = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None
        bitwise_and: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:80 in forward, code: slice_1 = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        slice_1: "i64[1, 1][2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:81 in forward, code: sub = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None
        sub: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:82 in forward, code: cat = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = None
        cat: "i64[1, 2049][2049, 1]cuda:0" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = unsqueeze = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:84 in forward, code: slice_3 = torch.ops.aten.slice.Tensor(cat, -1, 1, 2049);  cat = None
        slice_3: "i64[1, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 1, 2049)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:83 in forward, code: slice_2 = torch.ops.aten.slice.Tensor(cat, -1, 0, 2048)
        slice_2: "i64[1, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 0, 2048);  cat = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:85 in forward, code: sub_1 = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None
        sub_1: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:86 in forward, code: ne = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        ne: "b8[1, 2048][2048, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:87 in forward, code: cumsum = torch.ops.aten.cumsum.default(ne, -1);  ne = None
        cumsum: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:88 in forward, code: iota_1 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_1: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:91 in forward, code: unsqueeze_1 = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_1: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:92 in forward, code: unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_2: "i64[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:93 in forward, code: unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None
        unsqueeze_3: "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:103 in forward, code: index = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:104 in forward, code: index_1 = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        index_1: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:105 in forward, code: eq = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        eq: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:106 in forward, code: bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        bitwise_and_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:107 in forward, code: expand = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 2048, 2048]);  bitwise_and_1 = None
        expand: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 2048, 2048]);  bitwise_and_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1102 in forward, code: full_default_19 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1101 in forward, code: full_default_18 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1103 in forward, code: where_10 = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  expand = full_default_19 = full_default_18 = None
        where_10: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_19, full_default_18);  expand = full_default_19 = full_default_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1116 in forward, code: add_81 = torch.ops.aten.add.Tensor(view_142, where_10);  view_142 = where_10 = None
        add_76: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(view_142, where_10);  view_142 = where_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1122 in forward, code: eq_7 = torch.ops.aten.eq.Scalar(add_81, -inf);  add_81 = None
        eq_7: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Scalar(add_76, -inf)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1123 in forward, code: logical_not_10 = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        logical_not_10: "b8[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1124 in forward, code: any_6 = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        any_6: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1125 in forward, code: logical_not_11 = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        logical_not_11: "b8[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_6);  any_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1126 in forward, code: full_default_20 = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.full.default([1, 32, 2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1117 in forward, code: amax_5 = torch.ops.aten.amax.default(add_81, [-1], True)
        amax_5: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.amax.default(add_76, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1118 in forward, code: sub_10 = torch.ops.aten.sub.Tensor(add_81, amax_5);  amax_5 = None
        sub_10: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_76, amax_5);  add_76 = amax_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1119 in forward, code: exp_5 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        exp_5: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1120 in forward, code: sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        sum_6: "f32[1, 32, 2048, 1][65536, 2048, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1121 in forward, code: div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        div_5: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1127 in forward, code: where_11 = torch.ops.aten.where.self(logical_not_11, full_default_20, div_5);  logical_not_11 = full_default_20 = div_5 = None
        where_11: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(logical_not_11, full_default_20, div_5);  logical_not_11 = full_default_20 = div_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1128 in forward, code: expand_38 = torch.ops.aten.expand.default(where_11, [1, 32, 2048, 2048]);  where_11 = None
        expand_38: "f32[1, 32, 2048, 2048][134217728, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_11, [1, 32, 2048, 2048]);  where_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1129 in forward, code: view_143 = torch.ops.aten.view.default(expand_38, [32, 2048, 2048]);  expand_38 = None
        view_143: "f32[32, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(expand_38, [32, 2048, 2048]);  expand_38 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1085 in forward, code: convert_element_type_224 = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        convert_element_type_219: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1086 in forward, code: pow_77 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_224, 2)
        pow_77: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_219, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1087 in forward, code: mean_38 = torch.ops.aten.mean.dim(pow_77, [-1], True);  pow_77 = None
        mean_38: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_77, [-1], True);  pow_77 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1088 in forward, code: add_80 = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        add_75: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1089 in forward, code: pow_78 = torch.ops.aten.pow.Tensor_Scalar(add_80, -0.5);  add_80 = None
        pow_78: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_75, -0.5);  add_75 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1090 in forward, code: mul_152 = torch.ops.aten.mul.Tensor(convert_element_type_224, pow_78);  convert_element_type_224 = pow_78 = None
        mul_148: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_219, pow_78);  convert_element_type_219 = pow_78 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1091 in forward, code: convert_element_type_225 = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        convert_element_type_220: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_148, torch.bfloat16);  mul_148 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1092 in forward, code: permute_64 = torch.ops.aten.permute.default(convert_element_type_225, [0, 2, 1, 3]);  convert_element_type_225 = None
        permute_64: "bf16[1, 4, 2048, 512][4194304, 512, 2048, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_220, [0, 2, 1, 3]);  convert_element_type_220 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1097 in forward, code: unsqueeze_58 = torch.ops.aten.unsqueeze.default(permute_64, 2);  permute_64 = None
        unsqueeze_58: "bf16[1, 4, 1, 2048, 512][4194304, 512, 4194304, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_64, 2);  permute_64 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1098 in forward, code: expand_35 = torch.ops.aten.expand.default(unsqueeze_58, [1, 4, 8, 2048, 512]);  unsqueeze_58 = None
        expand_35: "bf16[1, 4, 8, 2048, 512][4194304, 512, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_58, [1, 4, 8, 2048, 512]);  unsqueeze_58 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1099 in forward, code: clone_20 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        clone_18: "bf16[1, 4, 8, 2048, 512][33554432, 8388608, 1048576, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1100 in forward, code: view_139 = torch.ops.aten.view.default(clone_20, [1, 32, 2048, 512]);  clone_20 = None
        view_139: "bf16[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1, 32, 2048, 512]);  clone_18 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1106 in forward, code: convert_element_type_228 = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        convert_element_type_223: "f32[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1130 in forward, code: expand_39 = torch.ops.aten.expand.default(convert_element_type_228, [1, 32, 2048, 512]);  convert_element_type_228 = None
        expand_39: "f32[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_223, [1, 32, 2048, 512]);  convert_element_type_223 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1131 in forward, code: view_144 = torch.ops.aten.view.default(expand_39, [32, 2048, 512]);  expand_39 = None
        view_144: "f32[32, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.aten.reshape.default(expand_39, [32, 2048, 512]);  expand_39 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1132 in forward, code: bmm_11 = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None
        bmm_11: "f32[32, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1133 in forward, code: view_145 = torch.ops.aten.view.default(bmm_11, [1, 32, 2048, 512]);  bmm_11 = None
        view_145: "f32[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_11, [1, 32, 2048, 512]);  bmm_11 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1134 in forward, code: convert_element_type_230 = torch.ops.prims.convert_element_type.default(view_145, torch.bfloat16);  view_145 = None
        convert_element_type_224: "bf16[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_145, torch.bfloat16);  view_145 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1135 in forward, code: permute_66 = torch.ops.aten.permute.default(convert_element_type_230, [0, 2, 1, 3]);  convert_element_type_230 = None
        permute_66: "bf16[1, 2048, 32, 512][33554432, 512, 1048576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_224, [0, 2, 1, 3]);  convert_element_type_224 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1136 in forward, code: clone_21 = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        clone_19: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1137 in forward, code: view_146 = torch.ops.aten.view.default(clone_21, [1, 2048, -1]);  clone_21 = None
        view_146: "bf16[1, 2048, 16384][33554432, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [1, 2048, -1]);  clone_19 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1139 in forward, code: view_147 = torch.ops.aten.view.default(view_146, [2048, 16384]);  view_146 = None
        view_147: "bf16[2048, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(view_146, [2048, 16384]);  view_146 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1138 in forward, code: permute_67 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        permute_67: "bf16[16384, 5376][1, 16384]cuda:0" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1140 in forward, code: mm_37 = torch.ops.aten.mm.default(view_147, permute_67);  view_147 = permute_67 = None
        mm_37: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_147, permute_67);  view_147 = permute_67 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1141 in forward, code: view_148 = torch.ops.aten.view.default(mm_37, [1, 2048, 5376]);  mm_37 = None
        view_148: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [1, 2048, 5376]);  mm_37 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1142 in forward, code: convert_element_type_233 = torch.ops.prims.convert_element_type.default(view_148, torch.float32);  view_148 = None
        convert_element_type_227: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_148, torch.float32);  view_148 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1143 in forward, code: pow_79 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2)
        pow_79: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_227, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1144 in forward, code: mean_39 = torch.ops.aten.mean.dim(pow_79, [-1], True);  pow_79 = None
        mean_39: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_79, [-1], True);  pow_79 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1145 in forward, code: add_82 = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        add_77: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1146 in forward, code: pow_80 = torch.ops.aten.pow.Tensor_Scalar(add_82, -0.5);  add_82 = None
        pow_80: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_77, -0.5);  add_77 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1147 in forward, code: mul_155 = torch.ops.aten.mul.Tensor(convert_element_type_233, pow_80);  convert_element_type_233 = pow_80 = None
        mul_151: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_227, pow_80);  convert_element_type_227 = pow_80 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1148 in forward, code: convert_element_type_234 = torch.ops.prims.convert_element_type.default(arg81_1, torch.float32);  arg81_1 = None
        convert_element_type_228: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg81_1, torch.float32);  arg81_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1149 in forward, code: mul_156 = torch.ops.aten.mul.Tensor(mul_155, convert_element_type_234);  mul_155 = convert_element_type_234 = None
        mul_152: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, convert_element_type_228);  mul_151 = convert_element_type_228 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1150 in forward, code: convert_element_type_235 = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None
        convert_element_type_229: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1151 in forward, code: add_83 = torch.ops.aten.add.Tensor(mul_141, convert_element_type_235);  mul_141 = convert_element_type_235 = None
        add_78: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, convert_element_type_229);  mul_137 = convert_element_type_229 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1152 in forward, code: convert_element_type_236 = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        convert_element_type_230: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1153 in forward, code: pow_81 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_236, 2)
        pow_81: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_230, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1154 in forward, code: mean_40 = torch.ops.aten.mean.dim(pow_81, [-1], True);  pow_81 = None
        mean_40: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_81, [-1], True);  pow_81 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1155 in forward, code: add_84 = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        add_79: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1156 in forward, code: pow_82 = torch.ops.aten.pow.Tensor_Scalar(add_84, -0.5);  add_84 = None
        pow_82: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_79, -0.5);  add_79 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1157 in forward, code: mul_157 = torch.ops.aten.mul.Tensor(convert_element_type_236, pow_82);  convert_element_type_236 = pow_82 = None
        mul_153: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, pow_82);  convert_element_type_230 = pow_82 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1158 in forward, code: convert_element_type_237 = torch.ops.prims.convert_element_type.default(arg82_1, torch.float32);  arg82_1 = None
        convert_element_type_231: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg82_1, torch.float32);  arg82_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1159 in forward, code: mul_158 = torch.ops.aten.mul.Tensor(mul_157, convert_element_type_237);  mul_157 = convert_element_type_237 = None
        mul_154: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, convert_element_type_231);  mul_153 = convert_element_type_231 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1160 in forward, code: convert_element_type_238 = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None
        convert_element_type_232: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1162 in forward, code: view_149 = torch.ops.aten.view.default(convert_element_type_238, [2048, 5376])
        view_149: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_232, [2048, 5376])

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1161 in forward, code: permute_68 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        permute_68: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1163 in forward, code: mm_38 = torch.ops.aten.mm.default(view_149, permute_68);  view_149 = permute_68 = None
        mm_38: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_149, permute_68);  view_149 = permute_68 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1164 in forward, code: view_150 = torch.ops.aten.view.default(mm_38, [1, 2048, 21504]);  mm_38 = None
        view_150: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [1, 2048, 21504]);  mm_38 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1165 in forward, code: convert_element_type_241 = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        convert_element_type_235: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1171 in forward, code: mul_163 = torch.ops.aten.mul.Tensor(convert_element_type_241, 0.5);  convert_element_type_241 = None
        mul_159: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.5)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1166 in forward, code: mul_159 = torch.ops.aten.mul.Tensor(convert_element_type_241, convert_element_type_241)
        mul_155: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, convert_element_type_235)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1167 in forward, code: mul_160 = torch.ops.aten.mul.Tensor(mul_159, convert_element_type_241);  mul_159 = None
        mul_156: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, convert_element_type_235);  mul_155 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1168 in forward, code: mul_161 = torch.ops.aten.mul.Tensor(mul_160, 0.044715);  mul_160 = None
        mul_157: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_156, 0.044715);  mul_156 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1169 in forward, code: add_85 = torch.ops.aten.add.Tensor(convert_element_type_241, mul_161);  mul_161 = None
        add_80: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_235, mul_157);  convert_element_type_235 = mul_157 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1170 in forward, code: mul_162 = torch.ops.aten.mul.Tensor(add_85, 0.7978845608028654);  add_85 = None
        mul_158: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1172 in forward, code: tanh_5 = torch.ops.aten.tanh.default(mul_162);  mul_162 = None
        tanh_5: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1173 in forward, code: add_86 = torch.ops.aten.add.Tensor(tanh_5, 1);  tanh_5 = None
        add_81: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1);  tanh_5 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1174 in forward, code: mul_164 = torch.ops.aten.mul.Tensor(mul_163, add_86);  mul_163 = add_86 = None
        mul_160: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, add_81);  mul_159 = add_81 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1175 in forward, code: convert_element_type_242 = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16);  mul_164 = None
        convert_element_type_236: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_160, torch.bfloat16);  mul_160 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1177 in forward, code: view_151 = torch.ops.aten.view.default(convert_element_type_238, [2048, 5376]);  convert_element_type_238 = None
        view_151: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_232, [2048, 5376]);  convert_element_type_232 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1176 in forward, code: permute_69 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        permute_69: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1178 in forward, code: mm_39 = torch.ops.aten.mm.default(view_151, permute_69);  view_151 = permute_69 = None
        mm_39: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_151, permute_69);  view_151 = permute_69 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1179 in forward, code: view_152 = torch.ops.aten.view.default(mm_39, [1, 2048, 21504]);  mm_39 = None
        view_152: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [1, 2048, 21504]);  mm_39 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1180 in forward, code: mul_165 = torch.ops.aten.mul.Tensor(convert_element_type_242, view_152);  convert_element_type_242 = view_152 = None
        mul_161: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, view_152);  convert_element_type_236 = view_152 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1182 in forward, code: view_153 = torch.ops.aten.view.default(mul_165, [2048, 21504]);  mul_165 = None
        view_153: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_161, [2048, 21504]);  mul_161 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1181 in forward, code: permute_70 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        permute_70: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1183 in forward, code: mm_40 = torch.ops.aten.mm.default(view_153, permute_70);  view_153 = permute_70 = None
        mm_40: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_153, permute_70);  view_153 = permute_70 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1184 in forward, code: view_154 = torch.ops.aten.view.default(mm_40, [1, 2048, 5376]);  mm_40 = None
        view_154: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [1, 2048, 5376]);  mm_40 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1185 in forward, code: convert_element_type_247 = torch.ops.prims.convert_element_type.default(view_154, torch.float32);  view_154 = None
        convert_element_type_241: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_154, torch.float32);  view_154 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1186 in forward, code: pow_83 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_247, 2)
        pow_83: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_241, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1187 in forward, code: mean_41 = torch.ops.aten.mean.dim(pow_83, [-1], True);  pow_83 = None
        mean_41: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_83, [-1], True);  pow_83 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1188 in forward, code: add_87 = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        add_82: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1189 in forward, code: pow_84 = torch.ops.aten.pow.Tensor_Scalar(add_87, -0.5);  add_87 = None
        pow_84: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_82, -0.5);  add_82 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1190 in forward, code: mul_166 = torch.ops.aten.mul.Tensor(convert_element_type_247, pow_84);  convert_element_type_247 = pow_84 = None
        mul_162: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_241, pow_84);  convert_element_type_241 = pow_84 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1191 in forward, code: convert_element_type_248 = torch.ops.prims.convert_element_type.default(arg86_1, torch.float32);  arg86_1 = None
        convert_element_type_242: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg86_1, torch.float32);  arg86_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1192 in forward, code: mul_167 = torch.ops.aten.mul.Tensor(mul_166, convert_element_type_248);  mul_166 = convert_element_type_248 = None
        mul_163: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, convert_element_type_242);  mul_162 = convert_element_type_242 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1193 in forward, code: convert_element_type_249 = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16);  mul_167 = None
        convert_element_type_243: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_163, torch.bfloat16);  mul_163 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1194 in forward, code: add_88 = torch.ops.aten.add.Tensor(add_83, convert_element_type_249);  add_83 = convert_element_type_249 = None
        add_83: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_78, convert_element_type_243);  add_78 = convert_element_type_243 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1195 in forward, code: mul_168 = torch.ops.aten.mul.Tensor(add_88, arg87_1);  add_88 = arg87_1 = None
        mul_164: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_83, arg87_1);  add_83 = arg87_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1196 in forward, code: convert_element_type_250 = torch.ops.prims.convert_element_type.default(mul_168, torch.float32);  mul_168 = None
        convert_element_type_244: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_164, torch.float32);  mul_164 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1197 in forward, code: pow_85 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_250, 2)
        pow_85: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_244, 2)

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1198 in forward, code: mean_42 = torch.ops.aten.mean.dim(pow_85, [-1], True);  pow_85 = None
        mean_42: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_85, [-1], True);  pow_85 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1199 in forward, code: add_89 = torch.ops.aten.add.Tensor(mean_42, 1e-06);  mean_42 = None
        add_84: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_42, 1e-06);  mean_42 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1200 in forward, code: pow_86 = torch.ops.aten.pow.Tensor_Scalar(add_89, -0.5);  add_89 = None
        pow_86: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_84, -0.5);  add_84 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1201 in forward, code: mul_169 = torch.ops.aten.mul.Tensor(convert_element_type_250, pow_86);  convert_element_type_250 = pow_86 = None
        mul_165: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_244, pow_86);  convert_element_type_244 = pow_86 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1202 in forward, code: convert_element_type_251 = torch.ops.prims.convert_element_type.default(arg88_1, torch.float32);  arg88_1 = None
        convert_element_type_245: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg88_1, torch.float32);  arg88_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1203 in forward, code: mul_170 = torch.ops.aten.mul.Tensor(mul_169, convert_element_type_251);  mul_169 = convert_element_type_251 = None
        mul_166: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, convert_element_type_245);  mul_165 = convert_element_type_245 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1204 in forward, code: convert_element_type_252 = torch.ops.prims.convert_element_type.default(mul_170, torch.bfloat16);  mul_170 = None
        convert_element_type_246: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1205 in forward, code: slice_31 = torch.ops.aten.slice.Tensor(convert_element_type_252, 1, -1, 9223372036854775807);  convert_element_type_252 = None
        slice_31: "bf16[1, 1, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_246, 1, -1, 9223372036854775807);  convert_element_type_246 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1207 in forward, code: view_155 = torch.ops.aten.view.default(slice_31, [1, 5376]);  slice_31 = None
        view_155: "bf16[1, 5376][11010048, 1]cuda:0" = torch.ops.aten.reshape.default(slice_31, [1, 5376]);  slice_31 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1206 in forward, code: permute_71 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        permute_71: "bf16[5376, 262144][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1208 in forward, code: mm_41 = torch.ops.aten.mm.default(view_155, permute_71);  view_155 = permute_71 = None
        mm_41: "bf16[1, 262144][262144, 1]cuda:0" = torch.ops.aten.mm.default(view_155, permute_71);  view_155 = permute_71 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1209 in forward, code: view_156 = torch.ops.aten.view.default(mm_41, [1, 1, 262144]);  mm_41 = None
        view_156: "bf16[1, 1, 262144][262144, 262144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [1, 1, 262144]);  mm_41 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1210 in forward, code: div_6 = torch.ops.aten.div.Tensor(view_156, 30.0);  view_156 = None
        div_6: "bf16[1, 1, 262144][262144, 262144, 1]cuda:0" = torch.ops.aten.div.Tensor(view_156, 30.0);  view_156 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1211 in forward, code: tanh_6 = torch.ops.aten.tanh.default(div_6);  div_6 = None
        tanh_6: "bf16[1, 1, 262144][262144, 262144, 1]cuda:0" = torch.ops.aten.tanh.default(div_6);  div_6 = None

        # File: /tmp/gemma4-artifacts/math/tlparse-report/-_0_0_0/fx_graph_runnable_11.txt:1212 in forward, code: mul_171 = torch.ops.aten.mul.Tensor(tanh_6, 30.0);  tanh_6 = None
        mul_167: "bf16[1, 1, 262144][262144, 262144, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_6, 30.0);  tanh_6 = None
        return (mul_167,)
