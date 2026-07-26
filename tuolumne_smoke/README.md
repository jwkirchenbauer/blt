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

The four JSONL shards are deliberately tiny fixtures, not training data for a
meaningful model. The successful tests used raw JSONL, space patching, bf16,
xFormers local/global attention, FlexAttention cross-attention, optimizer
steps, exact resume, FSDP checkpointing, consolidation, and strict reload.

On this ROCm stack, native single-prompt generation and sequential scoring
work. Native batched generation, and one replicated multi-rank generation
case, encountered GPU memory-access faults in dynamic generation helpers.
The converted `itazap/blt-1b-hf` Transformers path is the reliable pretrained
generation route while that native issue remains unresolved.
