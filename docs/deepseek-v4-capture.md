# DeepSeek V4 Flash inference capture

The workload `hf/infer/deepseek-ai/DeepSeek-V4-Flash-Base_layers1-4_b1_s512_bf16_grouped_mm`
captures four representative decoder layers from DeepSeek V4 Flash Base. It
adds one saved FX graph, 28 new canonical patterns and shape points, and 54
fusible-region occurrences. These counts describe the captured slice, not
the complete 43-layer model.

## Workload and coverage

| Setting | Value |
| --- | --- |
| Config | `deepseek-ai/DeepSeek-V4-Flash-Base` |
| Config revision | `8855555deef230a27a21a8d6f294b7b7497759b6` |
| Original config SHA-256 | `90c592a2dc666e9352a9bc28387414486f074351c7f57a1028bd528e0d0243bc` |
| HF class | `DeepseekV4ForCausalLM` |
| Parameter initialization | Random config initialization, seed 0; no checkpoint weights |
| Mode and inputs | Eval/no-grad, batch 1, sequence 512, synthetic token IDs |
| Precision | BF16; model-declared strict FP32 modules and parameters retained in FP32 |
| Attention / experts | Native eager attention / `grouped_mm` |
| Cache | Disabled; stateless prefill |
| Output | All 512 language-model logits; vocabulary 129280 |
| Runtime constant folding | Disabled |
| Dynamo settings | `fullgraph=False`, `dynamic=False` |

Original zero-based layer indices are selected before allocating weights:

| Original index | Attention | Routing |
| --- | --- | --- |
| 1 | Sliding window | Hash |
| 2 | Compressed sparse (CSA) | Hash |
| 3 | Heavily compressed (HCA) | Learned top-k |
| 4 | Compressed sparse (CSA) | Learned top-k |

This includes every attention/routing combination in the published Flash
schedule and the learned-routing HCA/CSA cycle. Widths, vocabulary, head
geometry, expert counts and dimensions, compression rates, and YaRN settings
are unchanged. The embedding, hyper-connections, shared experts, final norm,
and language-model head are included. Hash tables contain synthetic legal
assignments to six distinct experts per vocabulary entry.

The graph contains eight grouped GEMMs, four routing sorts and histograms,
CSA indexer top-k and scatter operations, and sliding/CSA/HCA attention.
Ordinary and grouped GEMMs remain in the saved full graph even though the
canonical corpus partitions fusible regions separately from external calls.
At sequence 512, CSA has 128 compressed entries, so its configured top-k of
512 selects all available entries. Longer-context sparse-selection behavior
and cached decoding are not covered. FP8 checkpoint execution is not covered.

## Generation and validation

Capture used PyTorch `2.16.0a0+gitb8d6389` and local Transformers revision
`6a74bb9094493a513037f0c9812d5a97d76f4f5e` on an H200 MIG 3g.71gb device.
The source workload compiled and executed as one graph with zero graph
breaks and finite BF16 logits of shape `[1, 512, 129280]`.

The producer sequence was source `torch.compile` with `TORCH_TRACE`, tlparse
export, native exported `load_args(InputReader(...))`, and runnable replay
through `install_capture_hook` and `temporary_capture_for_merge`. No generated
graphs, canonical repros, manifests, or shape metadata were edited by hand.
Exported tensor payloads were omitted; replay synthesizes data through the
native loader, so replay does not establish equality with source outputs.

The reproduction bundle is retained outside the repository at
`/workspaces/pytorch-devcontainers/specs/deepseekv4/`:

- `README.md` and `commands.sh`: workload description and exact generation,
  replay, and validation commands.
- `original-config.json` and `effective-config.json`: original checkpoint
  config and the resolved configuration used for the selected layers.
- `capture_source.py`: constructs and compiles the selected HF workload.
- `capture_export.py`: replays the exported runnable and invokes the existing
  capture/merge APIs for staging and final integration.
- `inputs/fx_graph_runnable.py`: an unedited copy of the exported Inductor-input
  runnable, including its native input loader. No tensor payload files are
  required.

Compiler caches, trace exports, raw capture indices, operation/shape census,
environment metadata, and detailed validation reports are stored under
`~/.cache/deepseekv4-extraction/`, rather than in the reproduction bundle or
the repository.

The reproduction commands invoke the shared `scripts/validate_eager.py` and
`scripts/validate_corpus_invariants.py` directly; no separate model-specific
validator is required. The capture hook performs graph roundtrip validation.

Validation requires zero dropped regions, successful graph roundtrips,
execution of the saved full graph and all affected canonical shape points,
and the shared `scripts/validate_eager.py --all-shapes` and
`scripts/validate_corpus_invariants.py` checks. Numerical equivalence, oracle
coverage, and performance measurement are separate work.

The baseline corpus invariant check already failed because 14 existing
canonical directories lacked `repro.py`. The addition is validated in an
isolated staging corpus. The final corpus check reports the same 14 missing
repros; its sampled eager validation passes. All 28 added shape points pass
the shared isolated eager validator.
