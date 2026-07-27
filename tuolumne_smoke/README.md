# BLT Tuolumne smoke tests

These are the small ROCm tests used to validate this BLT checkout on one
Tuolumne node. Run them from the repository root after activating the
environment:

```bash
conda_activate "$WRKSPC/tuolumne_conda_291_643_blt"
cd /p/vast1/kirchenb/hlm-root/blt
```

Submit them through `/p/vast1/kirchenb/llnl-tools/launch_tuo.py`. Use one Flux
task per GPU, do not nest `torchrun`, and pass `--pass_run_name=False`. Keep
the build settings on every target invocation:

```bash
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942'
```

The useful target commands are:

```bash
# Four ranks / four GPUs: ROCm, RCCL all-reduce, and xFormers CK backward.
PYTHONDONTWRITEBYTECODE=1 python -u tuolumne_smoke/runtime_rocm.py

# Four ranks / four GPUs: stress the primitives implicated by asynchronous
# native-generation tracebacks.
PYTHONDONTWRITEBYTECODE=1 python -u \
  tuolumne_smoke/native_generation_primitives.py --iterations 50

# One GPU: official native BLT plus its nested entropy checkpoint.
HF_HUB_CACHE=/p/vast1/kirchenb/hlm-root/hf-cache/hub \
PYTHONPATH="$PWD:$PYTHONPATH" \
python -u tuolumne_smoke/native_inference.py \
  --cache-dir /p/vast1/kirchenb/hlm-root/hf-cache/hub \
  --entropy-from-blt-repo --max-gen-len 4 \
  --prompt 'The capital of France is'

# One GPU: converted Hugging Face Transformers checkpoint.
HF_HUB_CACHE=/p/vast1/kirchenb/hlm-root/hf-cache/hub \
python -u tuolumne_smoke/transformers_inference.py \
  --cache-dir /p/vast1/kirchenb/hlm-root/hf-cache/hub \
  --max-new-tokens 4 --prompt 'The capital of France is'

# One GPU: sequential likelihood scoring and entropy-patch inspection.
HF_HUB_CACHE=/p/vast1/kirchenb/hlm-root/hf-cache/hub \
PYTHONPATH="$PWD:$PYTHONPATH" \
python -u tuolumne_smoke/pretrained_entropy_eval.py \
  --cache-dir /p/vast1/kirchenb/hlm-root/hf-cache/hub

# One GPU: three tiny training steps and a distributed checkpoint.
python -u -m bytelatent.train \
  config=tuolumne_smoke/configs/tiny_blt_space.yaml

# Four ranks / four GPUs: the same fixture with full-shard FSDP.
python -u -m bytelatent.train \
  config=tuolumne_smoke/configs/tiny_blt_space.yaml \
  name=blt-tiny-space-smoke-4gpu \
  dump_dir=/p/vast1/kirchenb/hlm-root/tuo-runs/blt-tiny-space-train-4gpu \
  distributed.fsdp_type=full_shard \
  distributed.dp_shard=4 distributed.dp_replicate=1

# One GPU: strict load and forward pass after checkpoint consolidation.
PYTHONPATH="$PWD:$PYTHONPATH" \
python -u tuolumne_smoke/consolidated_checkpoint_forward.py \
  --checkpoint-dir PATH_TO_CHECKPOINT_STEP
```

For interactive pretrained exploration, open
`tuolumne_smoke/pretrained_blt_exploration.ipynb` with a kernel from
`$WRKSPC/tuolumne_conda_291_643_blt` inside a compute-node allocation. The
notebook loads `facebook/blt-1b` and its nested entropy model once on GPU 0,
then provides reusable cells for patch inspection, greedy or sampled
generation, variable-length batched generation, and teacher-forced BPB
scoring. Its single-process RCCL setup is self-contained and does not require
`torchrun`.

`run_pretrained_blt_notebook.py` is a validation driver that executes the
notebook's actual code cells sequentially in one shared namespace. Job
`f3NkQb4gwC8X` ran all eight cells on an MI300A: the one-time load completed,
Unicode patch inspection returned 28 patches, and the active 16-byte example
generated `" Paris. The capi"` with an 8.931 GiB peak allocation.

The larger `facebook/blt-7b` checkpoint also fits comfortably on one MI300A.
Cold-cache job `f3NkXrAkXPao` downloaded and loaded its 10,553,570,560
parameters in 276.16 seconds, then generated 16 bytes in 43.15 seconds with a
20.133 GiB peak allocation. Warm-cache job `f3NkbDpqn1W7`, using 24 CPU cores,
loaded in 166.29 seconds and generated the same 16 bytes in 27.00 seconds.
Both completed all notebook cells cleanly. Select it by changing `BLT_REPO`
in the loading cell to `facebook/blt-7b`; its files are now present in the
shared Hugging Face cache used by the notebook.

The four JSONL shards are deliberately tiny fixtures, not training data for a
meaningful model. The successful tests used raw JSONL, space patching, bf16,
xFormers local/global attention, FlexAttention cross-attention, optimizer
steps, exact resume, FSDP checkpointing, consolidation, and strict reload.

On this ROCm stack, native single-prompt, batched, and replicated multi-rank
generation now work with eager BlockMask metadata construction. The converted
`itazap/blt-1b-hf` Transformers path is also available as an independent
pretrained generation route.

## Testing debrief

The main issues fell into three groups: BLT correctness/compatibility bugs
fixed in this branch, ROCm/environment mismatches handled by the installer,
and native-generation faults traced to a deprecated PyTorch compilation path.

| Issue | What happened | Status |
| --- | --- | --- |
| CUDA-specific xFormers assumptions | BLT unconditionally referenced Flash/CUTLASS operators absent from the ROCm xFormers wheel, so importing the distributed code could fail before execution. | Fixed by registering only operators exposed by the installed backend and recognizing the ROCm CK operator in `bytelatent/distributed.py`. |
| Raw JSONL checkpoint resume | Training saved a valid checkpoint, but exact continuation reconstructed the input dataset as Arrow regardless of the configured format. Model and optimizer state loaded before iterator restoration failed. | Fixed by preserving `self.file_format` in `bytelatent/data/iterators/arrow_iterator.py`. Exact continuation from step 3 to step 4 then succeeded. |
| Upstream CUDA-oriented environment | The upstream xFormers source pin targets CUDA and was unsuitable for this ROCm stack. Several packages imported by data and Hugging Face paths were missing from upstream requirements. | `install_tuolumne_291_643_blt.sh` uses the ROCm xFormers wheel and adds `jsonlines`, `safetensors`, `hf_xet`, compatible Hub/Transformers packages, and `accelerate`. |
| Native batched generation | Generating several differently sized prompts together caused a ROCm GPU memory-access fault. The original asynchronous traceback misleadingly stopped at `tokens_to_seqlen`; kernel serialization located the fault in Triton's compiled `create_block_mask` metadata kernel for decoder cross-attention. | Fixed by using PyTorch's supported eager BlockMask metadata builder. The mask and FlexAttention computation are unchanged. |
| Native replicated multi-rank generation | With one prompt assigned to each independent rank, three ranks generated successfully while rank 0 appeared to fault during polynomial hashing. Hash stress tests passed on every GPU, while eager BlockMask construction made all four ranks reliable. | Fixed by the same BlockMask change. The old hash frame was delayed asynchronous fault reporting from an earlier compiled cross-mask kernel. |
| Standalone entropy checkpoint access | `facebook/blt-entropy` returned HTTP 403 during the initial test even after access was requested. | Bypassed by loading the entropy checkpoint embedded in `facebook/blt-1b`. |
| Large checkpoint downloads | Downloads could appear hung and leave partial cache content without the Xet transfer backend. | The partial cache was cleaned and `hf_xet` installed; subsequent cached model loading succeeded. |
| Standalone script imports | Running a script by absolute path sets Python's import root to the script directory and produced `ModuleNotFoundError: bytelatent`. | Commands set `PYTHONPATH="$PWD:$PYTHONPATH"` when the driver imports this checkout. |
| Misleading launcher completion | Generated `launch.sh` prints its final completion message after `flux run`, so that message alone does not prove the inner command succeeded. | The launcher remains unchanged. Inspect explicit success markers and tracebacks in the Flux log. |
| Long first-step compilation | The first FlexAttention execution took roughly 30–35 seconds and initially resembled a hang; later tiny-training steps took tens of milliseconds. | Expected compilation warm-up rather than a failure. The shared TorchInductor cache helps subsequent runs. |

The core training path is healthy: single-GPU training, four-GPU FSDP, RCCL,
backward passes, optimizer updates, checkpointing, exact resume,
consolidation, strict reload, entropy-patched likelihood evaluation, and
native pretrained generation all completed successfully.

## Native generation diagnosis

PyTorch 2.9's `create_block_mask` defines `_compile=False` as its supported
default. Its source marks `_compile=True` as a deprecated internal workaround
whose original limitation has been removed. Upstream BLT explicitly requested
that deprecated path for every encoder and decoder cross-attention mask.

With `AMD_SERIALIZE_KERNEL=3`, the variable-prompt failure was attributed to
the generated Triton metadata kernel for a batch-4 decoder mask with
`Q_LEN=24` and `KV_LEN=44`. The original non-serialized stacks stopped later
at whichever operation next synchronized the GPU, explaining the inconsistent
`tokens_to_seqlen` and polynomial-hash frames.

The following checks separate the primitives from the failing integration:

- `f3Nk1xT1KqGB`: all four GPUs passed 50 iterations of sequence-length
  extraction, official hash widths 3–8 against CPU results, and CK xFormers
  block-diagonal local attention at batch sizes 1 and 4.
- `f3Nk6PD23fN7`: the original compiled BlockMask path reproduced the fault on
  a different node and identified the exact Triton `create_block_mask` kernel.
- `f3Nk8tjRrrZu`: eager BlockMask construction generated all four variable
  prompts successfully in one batch.
- `f3NkB9JUUECf`: four independent ranks generated their assigned prompts
  successfully with the same path.
- `f3NkE9xGf2co`: the final source default, without a diagnostic flag,
  reproduced the successful four-prompt batch.
- `f3NkGcLzYEjZ`: a one-step tiny training regression completed its forward
  pass, backward pass, optimizer update, and clean teardown with the final
  source default.

`BLT_COMPILE_BLOCK_MASK=1` restores upstream's legacy metadata compilation
path for controlled comparisons. It should not be enabled for batched native
generation on this ROCm/PyTorch stack.
