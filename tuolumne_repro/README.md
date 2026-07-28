# BLT reproduction on Tuolumne

This directory contains the reproducible data, training, and evaluation
workflow for a scratch reproduction of the released BLT "1B" model. The
published name refers approximately to the 1.28B-parameter global latent
Transformer; the complete released model has 4,533,879,808 parameters because
its byte-group hash embeddings contain 3,072,012,288 parameters.

The goal is to minimize the scientific difference from the released model.
When the paper, released checkpoint arguments, and repository template
disagree, the released checkpoint arguments are the canonical recipe. Every
intentional or unavoidable difference is recorded below.

## Fixed paths

Repository-controlled specifications and programs live here:

```text
/p/vast1/kirchenb/hlm-root/blt/tuolumne_repro
```

Dataset artifacts, including bounded pilot artifacts, live on the volume that
will eventually hold the full corpus:

```text
$VAST_PT/datasets/blt
```

The Python environment is:

```bash
conda_activate "$WRKSPC/tuolumne_conda_291_643_blt"
```

GPU programs must run through
`/p/vast1/kirchenb/llnl-tools/launch_tuo.py`, with one Flux rank per GPU and
the ROCm build settings:

```bash
MAX_JOBS=48 PYTORCH_ROCM_ARCH=gfx942 GPU_ARCHS=gfx942
```

Multi-node launches must also explicitly pass the known-good Tuolumne RCCL
plugin rather than relying on the launcher's older default:

```bash
--rccl_installdir \
  /collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.4.1/install/lib
```

Use these launcher resource presets. In every GPU case, `tasks_per_node=4`
means one process per MI300A and `cpus_per_task=24`; do not replace this with
an internal `torchrun`.

| Workload | QoS | Nodes | Tasks/node | GPUs/node | CPUs/task |
| --- | --- | ---: | ---: | ---: | ---: |
| Download, scan, shuffle, or audit | `pbatch` | 1 | 1 | 1 | 96 |
| Entropy-model training | `pbatch` | 2 | 4 | 4 | 24 |
| Official or scratch entropy scoring | `pbatch` | 16 | 4 | 4 | 24 |
| Rejected main-model memory candidate | `pbatch` | 16 | 4 | 4 | 24 |
| Primary main-model topology candidate | `pbatch` | 32 | 4 | 4 | 24 |
| Released-topology fallback | `pbatch` | 64 | 4 | 4 | 24 |

Short bounded versions use `pdebug`; its one-hour limit is not appropriate
for production preparation or training.

Do not use Flux `afterok` as the sole correctness gate with the currently
generated `launch.sh`. Its final `echo` masks a nonzero `flux run` status, so
the allocation can be reported as completed after a target failure. This
workflow inspects the target artifact and log before submitting the next
stage. If dependency chaining is later required, adjust that run's generated
`launch.sh` to propagate the `flux run` status; no change to the shared
launcher is required.

The upstream trainer logs its complete environment. Prefix production
training invocations with
`env -u HF_HUB_TOKEN -u HUGGING_FACE_HUB_TOKEN`; all checkpoints and data in
this workflow are local, so those credentials are unnecessary. The bounded
`train_rehearsal.py` wrapper performs this removal itself.

## Frozen code and checkpoint inputs

The reproduction work is based on BLT repository commit
`82d5b55915a72934f865d6bd9e80c562fd7379ed`. Its upstream foundation before
the Tuolumne smoke and notebook commits is
`9774ed4fcc78313f9f218295f3d7e4decdadf2ae`. Record the eventual commit
containing this directory in each production run manifest as well.

The released `facebook/blt-1b` checkpoint is pinned to Hugging Face revision
`8134b32f0b1d25d1248c30e8c7bdfd442d3bb380`:

| Artifact | SHA-256 |
| --- | --- |
| `model.safetensors` | `3738bef71ca45c1ac58783c93c6eee8da6feefc0672113bcccd0a44cfd3cfe1f` |
| `config.json` | `e05024b547dc6765c1cd69ccf1b8cfaca51667145f56bd7493a372c2538f178b` |
| `train_args.json` | `3868dabcf0fd23c31bc7d3fc198c59f6e924d4f6bcd749d6f234361e6e235f94` |
| Entropy `consolidated.pth` | `69357751c37f17caeb21bcf29d4292f4c927cedcb31aefbc73472bd570cd5428` |
| Entropy `params.json` | `98ae063abc6b0183f6534f477ebc72b5d9f28f012a66c99370bc9392443fc716` |

## Canonical released recipe

### Entropy model

- 99,512,064 parameters.
- 14 Transformer layers, width 768, 12 heads.
- Byte vocabulary of 260, context length 8192, and local causal attention
  with a 512-byte sliding window.
- BF16 full-shard FSDP on 8 data-parallel ranks.
- Batch 8 per rank, sequence length 8192, no gradient accumulation.
- 100,000 optimizer updates, or 52,428,800,000 byte positions.
- AdamW: learning rate `4e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`,
  weight decay `0.1`, gradient clip `10`, 500 warmup steps, and cosine decay
  to a minimum ratio of `0.1`.

The paper's statement that this model has width 512 is inconsistent with both
the released 99.5M-parameter checkpoint and the repository configuration.
Width 768 is therefore used.

### Main BLT model

- Local encoder: 1 layer, width 1024, 16 heads.
- Global latent Transformer: 25 layers, width 2048, 16 heads.
- Local decoder: 9 layers, width 1024, 16 heads.
- Cross-attention: 16 heads, `k=2`, after the last encoder layer and before
  every decoder layer, with max-pooling initialization.
- One 500,002-entry hash table for each byte n-gram size from 3 through 8.
- Global entropy patching without monotonicity or newline resets.
- Target average patch size 4.5. The threshold is calibrated on the selected
  corpus and entropy checkpoint rather than copied from the released model.
- 4096 patches per sequence and at most 24,576 encoder byte positions.
- AdamW: learning rate `4e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`,
  weight decay `0.1`, gradient clip `1`, 2000 warmup steps, and cosine decay
  to a minimum ratio of `0.01`.
- BF16 full-shard FSDP, selective activation checkpointing, and no tensor
  parallelism.

The released global batch contains:

```text
256 ranks * 4 sequences/rank * 4096 patches/sequence
    = 4,194,304 patches/update
```

At the target 4.5 bytes per patch this is approximately 18,874,368 bytes per
update. The 240,000-update released run therefore corresponds to about 4.53T
bytes. The initial 100B trial uses 5,299 complete updates, corresponding to
100,015,603,712 byte positions at exactly 4.5 bytes per patch before encoder
truncation. Its final byte accounting will use observed non-padding positions.

## Training topology gates

The 16-node candidate was tested first, but batch 16 exhausted node-wide
MI300A memory during the first forward pass even with 64-way full-shard FSDP.
The next Tuolumne candidate is therefore 32 nodes, with the released 64-node
placement retained as a concurrently queued fallback:

| Setting | Released / fallback | Rejected candidate | Primary candidate |
| --- | ---: | ---: | ---: |
| Nodes | 64 | 16 | 32 |
| GPUs / data-parallel ranks | 256 | 64 | 128 |
| Sequences per rank | 4 | 16 | 8 |
| Patches per sequence | 4096 | 4096 | 4096 |
| Global patches per update | 4,194,304 | 4,194,304 | 4,194,304 |
| Gradient accumulation | 1 | 1 | 1 |

This preserves the mathematical global batch and optimizer cadence. It changes
only where examples and FSDP shards are placed. If the 128-rank collective
gate fails for a reason other than scheduling, the remaining exact fallback is
64 nodes / 256 GPUs / batch 4 per rank, which matches the released placement.
The checked configurations are `configs/blt_1b_32n_100b.yaml` and
`configs/blt_1b_64n_100b.yaml`, respectively.

Do not introduce gradient accumulation as an expedient. The current BLT loop
clips gradients after every microbatch and does not suppress intermediate
FSDP synchronization, so accumulation is not equivalent to the released
update when clipping activates.

## Public data specification

The primary public substitute is
`mlfoundations/dclm-baseline-1.0`, pinned to revision:

```text
a3b142c183aebe5af344955ae20836eb34dcf69b
```

The pinned repository contains 27,838 Zstandard-compressed JSONL files and
7,196,105,155,016 compressed bytes. The dataset card describes approximately
4T tokenizer tokens and 3B documents. DCLM is selected because it is included
in BLT-1T, is directly supported by the BLT repository, and is named in the
released entropy-model template.

Preparation makes no text-level filtering or normalization beyond the
upstream DCLM release:

1. Select source files using a seeded hash order rather than filesystem order.
2. Decompress the selected files in the recorded manifest order.
3. Insert one LF delimiter only where an archive lacks a final LF and another
   archive follows.
4. Preserve each upstream JSONL record byte-for-byte; metadata fields are not
   stripped and IDs are not injected before shuffling.
5. Shuffle complete records with the repository's pinned `terashuf`
   implementation, seed, and memory setting.
6. Round-robin the shuffled stream into the requested number of chunks with
   GNU `split -n r/N`.
7. Follow the repository convention by reserving the first 10,000 records from
   every chunk for validation.
8. Record documents, complete JSONL bytes, UTF-8 `text` bytes, and BLT
   positions including BOS/EOS in sidecar manifests.

Preserving complete input records matters because `terashuf` divides its input
by bytes before shuffling each in-memory run. Removing metadata or adding a
`sample_id` would change run boundaries and therefore the seeded permutation.
Content identity is instead audited with a multiset digest of complete JSONL
records, so duplicate records remain countable without modifying the corpus.

The inter-file delimiter is necessary because DCLM archives omit their final
LF. The stock BLT command uses `xargs zstdcat`, which directly joins those
streams and therefore concatenates the last JSON object of one archive with
the first object of the next. The bounded 10-file rehearsal observed exactly
nine such boundaries: 1,171,806 valid source records became 1,171,797 output
lines containing nine invalid JSON records. Inserting a delimiter at only
those boundaries restores the intended records without altering the bytes of
any record. The number of inserted delimiters is recorded in each
materialization manifest and the complete-record multiset audit remains the
authoritative gate.

### Why retain `terashuf`

The primary pipeline uses the same external-memory shuffle family as the BLT
repository. At pinned commit
`29a65ed74808925266a5dcf4ffccd29552dad0e0`, `terashuf`:

- randomly permutes each memory-sized run;
- writes those runs to temporary files when the input exceeds memory; and
- emits records by sampling a run in proportion to its number of remaining
  records.

This is a fair permutation of input lines and handles duplicate lines
independently. A hash-based shard assignment would be deterministic and
parallel, but it would be a different data-ordering algorithm and is therefore
not the reproduction default.

The opaque part is that the output is not a function of `SEED` alone. It also
depends on the byte-exact input stream, `MEMORY`, and the C++ standard-library
implementation used by `std::shuffle` and `uniform_int_distribution`. The
stock BLT command leaves source order to an unsorted `find` and clones the
current `terashuf` branch. This reproduction instead freezes:

```text
terashuf commit: 29a65ed74808925266a5dcf4ffccd29552dad0e0
terashuf.cc SHA-256: 05b08bfed31765b6f4f3430efbc06442459292a2d8f6f68612d4053026ca678c
seed: 42
production memory: 192 GiB
source order: the checked source-file manifest
Tuolumne compiler: GCC 13.3.1
```

The executable checksum, compiler version, environment, and the checksums of
all output chunks are captured for every materialization. Small tests on
Tuolumne confirmed that fixed input, seed, memory, executable, and environment
produce byte-identical output; changing input order, seed, or memory changes
the output. They also confirmed that taking 10,000 records from every
round-robin chunk selects exactly the first `10,000 * N` records of the
globally shuffled stream, although the validation-file ordering is grouped by
chunk.

`terashuf` is single-process and has no checkpoint/restart format. Those are
operational limitations, not reasons to change the scientific ordering for
the 100B trial. A deterministic hash shuffle remains only a documented
fallback if a measured one-node rehearsal shows that `terashuf` throughput,
temporary-space use, or restart risk makes the run impractical.

The first training materialization has 64 raw JSONL chunks so that the BLT
loader cannot silently discard chunks during 64-rank entropy scoring or the
original main-model topology test.
The loader keeps only the first `world_size` chunks when it sees more chunks
than ranks. Therefore, lossless 8- and 4-chunk views are derived for
released-topology entropy training and one-node rehearsals. These views
round-robin merge the canonical chunks to reconstruct the post-validation
global shuffled stream and round-robin split it again; they do not reshuffle
or change record bytes.

No expanded 128- or 256-chunk view is needed for main-model training.
`find_and_sanitize_chunks` accepts a world size that is a multiple of the
chunk count, and `distribute_data_to_rank` assigns
`world_size / n_chunks` strided Arrow workers to each chunk. Thus 128 ranks
consume the same 64 JSONL/Arrow pairs with two disjoint record streams per
chunk, and 256 ranks use four. This is the repository's native lossless
partitioning path and avoids a scientifically unnecessary reshard and entropy
rescoring pass.

The full-scale-compatible artifact layout is:

```text
$VAST_PT/datasets/blt/
  manifests/
  sources/
    dclm-baseline-1.0/
      a3b142c183aebe5af344955ae20836eb34dcf69b/
  prepared/
    dclm-100b-v1/
      dclm_baseline_1.0/
        dclm_baseline_1.0.chunk.00.jsonl
        ...
        dclm_baseline_1.0.chunk.63.jsonl
      validation/
      views/
        world_0008/
          dclm_baseline_1.0/
        world_0004/
          dclm_baseline_1.0/
  entropy/
    dclm-100b-v1/
      official/
        dclm_baseline_1.0/
          transformer_100m/
      scratch/
        dclm_baseline_1.0/
          transformer_100m/
```

The entropy directory name is derived by BLT from the raw filename prefix,
not from the enclosing source directory. The layout above deliberately makes
both names `dclm_baseline_1.0`.

CPU preparation is designed for one 96-core node. Bounded entropy-scoring
rehearsals use its four MI300A GPUs. The measured scorer is file-parallel but
serial within each file, so the 100B official and scratch scoring passes use
16 nodes / 64 independent GPU ranks, one canonical chunk per rank. This
changes only placement and wall time; each chunk still runs the repository's
unchanged serial scorer.

## Data command sequence

Run these as targets under the Tuolumne launcher. The inventory and pinned
tool build are already complete. The combined fail-fast implementation of
this sequence is `prepare_100b.sh`.

```bash
# Freeze upstream inventory and build the exact shuffle implementation.
python -u tuolumne_repro/data_pipeline.py inventory
python -u tuolumne_repro/data_pipeline.py build-terashuf

# Initial download estimate, then exact whole-file-prefix selection.
python -u tuolumne_repro/data_pipeline.py download --source-files 184
python -u tuolumne_repro/data_pipeline.py scan-sources \
  --source-files 184 --workers 96 --target-text-bytes 110000000000

# N is recommended_source_files from the scan. Download more and rescan if
# the initial estimate does not reach the target.
python -u tuolumne_repro/data_pipeline.py prepare \
  --materialization dclm-100b-v1 --source-files N \
  --n-chunks 64 --validation-per-chunk 10000 --memory-gib 192
python -u tuolumne_repro/data_pipeline.py audit \
  --materialization dclm-100b-v1 --workers 96
python -u tuolumne_repro/data_pipeline.py derive-view \
  --materialization dclm-100b-v1 --target-chunks 8 --audit-workers 96
python -u tuolumne_repro/data_pipeline.py derive-view \
  --materialization dclm-100b-v1 --target-chunks 4 --audit-workers 96
```

The final two views solve a loader/topology issue; they are not additional
shuffles. For GPU scoring, launch one rank per GPU:

```bash
python -u tuolumne_repro/score_entropy.py \
  --input-dir "$VAST_PT/datasets/blt/prepared/dclm-100b-v1/dclm_baseline_1.0" \
  --output-root "$VAST_PT/datasets/blt/entropy/dclm-100b-v1/official"
```

With 64 ranks, each rank owns exactly one canonical chunk. Re-run with the
consolidated scratch entropy checkpoint and the `scratch` output root for the
primary reproduction view. Validate each view and derive its threshold before
main-model training:

```bash
python -u tuolumne_repro/validate_entropies.py \
  --input-dir "$VAST_PT/datasets/blt/prepared/dclm-100b-v1/dclm_baseline_1.0" \
  --entropy-dir "$VAST_PT/datasets/blt/entropy/dclm-100b-v1/official/dclm_baseline_1.0/transformer_100m" \
  --output "$VAST_PT/datasets/blt/entropy/dclm-100b-v1/official/validation_all_64_world128.json" \
  --capacity-batch-size 8 --capacity-seq-len 4096 \
  --capacity-buffer-size 512 --capacity-steps 5299 \
  --capacity-workers-per-chunk 2
```

Repeat for the scratch entropy root. The validator computes one global
threshold but also retains each chunk's patch count and independently audits
both strided rank streams in every chunk.

If the exact released 64-node topology is needed, reuse the same input and
entropy directories and change only the capacity options:

```bash
--capacity-batch-size 4 --capacity-workers-per-chunk 4
```

This audits 256 disjoint rank streams. No data reshard or entropy rescoring is
performed.

Before either main-model control, run the machine-readable recipe and artifact
gate against that control's validation report:

```bash
python -u tuolumne_repro/check_preflight.py \
  --validation "$VAST_PT/datasets/blt/entropy/dclm-100b-v1/scratch/validation_all_64_world128.json" \
  --require-production-data \
  --require-capacity \
  --require-clean-repo \
  --output "$VAST_PT/datasets/blt/entropy/dclm-100b-v1/scratch/preflight.json"
```

The checker expands both YAML configurations through BLT's Pydantic parser,
compares them field-by-field with the pinned released main and entropy
training arguments, injects the selected validation threshold into both
required main-model fields, verifies the exact global batches and checkpoint
cadence, and requires audited production data and 5,299-update capacity in
all 128 strided streams. The current pilot-mode report is
`tuo-runs/blt-repro-1b-preflight-v1/preflight-pilot.json`; every required
recipe check passes, while pilot capacity and the not-yet-complete production
paths are correctly reported as non-required failures.

## Staged experiment

### Stage 0: freeze and audit

- Pin the code commit, DCLM revision, released BLT and entropy checkpoint
  revisions, seeds, exact arguments, and environment.
- Generate and checksum the complete upstream file inventory.
- Keep a machine-readable record of every selected source file and output
  shard.
- Pin and build `terashuf`, then record its source, binary, compiler, `SEED`,
  `MEMORY`, `TMPDIR`, and ordered input manifest.

Gate: rerunning inventory and selection produces identical manifests.

### Stage 1: bounded data-pipeline rehearsal

- Download a small, deterministically selected set of DCLM source files into
  `$VAST_PT/datasets/blt`.
- Exercise pinned `terashuf`, 64-way round-robin sharding, repository-equivalent
  validation selection, byte accounting, checksums, resume behavior, and
  cleanup.
- Derive topology-matched 8- and 4-chunk views from the canonical post-validation
  stream and verify them with complete-record multiset digests.
- Measure compressed input, normalized output, and Arrow expansion ratios.

Gate: complete-record multiset digests prove no document loss or duplication;
reruns have stable checksums; and output paths match the eventual 100B layout.

### Stage 2: entropy-model reproduction

- Run a short exact-shape entropy-training test.
- Train a 1--5B-byte convergence and throughput rehearsal.
- If healthy, train the 768-wide entropy model for the released 52.43B byte
  positions on the same DCLM distribution used for the main model.
- Consolidate the final distributed checkpoint with
  `python -m bytelatent.checkpoint consolidate <checkpoint-step-dir>` and use
  that immutable consolidated directory for scratch-entropy scoring.
- Evaluate the scratch and official entropy models on the same frozen
  validation data.

Gate: stable training, finite loss and gradients, acceptable held-out BPB,
successful model/optimizer/scheduler restart, quantified iterator-resume
semantics, and sufficient throughput for the full entropy run.

### Stage 3: entropy preprocessing and threshold calibration

- Score representative data with both official and scratch entropy models.
- Retain the repository's serial per-document scorer; parallelize only by
  assigning independent canonical chunks to GPU ranks.
- Calibrate separate global thresholds that produce 4.5 mean bytes per patch.
- Measure patch-size percentiles, 24,576-byte truncation frequency, packing
  efficiency, and rank-to-rank byte balance.
- Verify rank-local patch capacity after calibration. Aggregate corpus size is
  insufficient because every data-parallel rank loops its own chunk
  independently.

Gate: average patch size is within 1% of 4.5 and truncation or rank imbalance
does not dominate training. Every rank must also cover the requested update
count without repeating its chunk.

### Stage 4: materialize the 100B trial

- Download an initial 184-file estimate, scan exact text bytes in selection
  order, and extend the download if necessary.
- Select the shortest whole-file prefix reaching 110B UTF-8 bytes, then
  prepare it in 64 chunks on one CPU node. The 10B margin covers the
  repository-equivalent validation removal.
- Require the audited post-validation training chunks to contain at least
  100B unique text bytes before any training starts.
- Require every calibrated chunk to contain enough complete 512-sequence
  shuffle buffers for all 5,299 optimizer updates. At batch 8, one buffer
  supplies 64 updates; 83 complete buffers require 174,063,616 patches per
  strided rank stream. This conservative buffer-aware gate prevents the first
  buffer that crosses a shard boundary from silently mixing repeated
  documents.
- If any rank fails that capacity gate, extend the frozen whole-file prefix
  and create a new immutable materialization. Do not compensate with an extra
  epoch or by allowing only the short ranks to repeat.
- Produce official-entropy and scratch-entropy Arrow views.
- Validate loader state, exact and approximate resume behavior, and sustained
  batch delivery.
- Keep raw source archives until Arrow checksums and replay tests pass; then
  reclaim redundant normalized data if required.

Gate: a representative loader benchmark keeps GPU data stalls below the
compute time and accounts for every selected document and byte.

### Stage 5: 100B main-model experiment

Run the controls in this order:

1. Official released BLT-1B evaluated on the frozen public validation suite.
2. Scratch BLT using the official entropy model with a DCLM-calibrated
   threshold.
3. Scratch BLT using the scratch entropy model with its independently
   calibrated threshold. This is the primary scratch reproduction.
4. The released fixed threshold `1.335442066192627` is a diagnostic for
   distribution shift, not the primary setting.

Use 32 nodes, 128 ranks, batch 8, and no accumulation if the final collective
gate passes. Compare training curves and held-out BPB across
Wikipedia, DCLM/Common-Crawl, GitHub, and the frozen training distribution.
The paper's matching 100B cross-attention row is context, not a strict
acceptance target, because its corpus and validation samples are unavailable.
That row reports BPB `0.823` on Wikipedia, `0.871` on Common Crawl, `0.443`
on GitHub, and `0.846` on its training distribution.

### Stage 6: full 4.53T-byte reproduction

Proceed to 240,000 updates only after the 100B controls isolate data,
entropy-model, threshold, topology, and implementation effects. Reassess
storage using measured compression and Arrow expansion before materializing
the complete corpus. Because a run of this duration will require restarts,
either serialize the `SequenceIterator` shuffle buffer exactly and validate
continuous/restarted batch parity, or register the resulting data-order
change as a scientifically meaningful implementation difference.

## Scientific difference register

| Difference | Status | Treatment |
| --- | --- | --- |
| Original Llama 2 / BLT-1T mixture is unavailable | Unavoidable, major | Use pinned DCLM and never describe results as an exact data replication |
| Original data filtering, order, and validation samples are unavailable | Unavoidable, major | Freeze public deterministic replacements and evaluate the official checkpoint on them |
| Entropy width is 512 in prose but 768 in released artifacts | Resolved | Follow released checkpoint: width 768 |
| Paper says cosine decay to zero; released main config uses minimum ratio 0.01 | Resolved | Follow released checkpoint for checkpoint reproduction |
| Paper's final 8B uses monotonicity/newline resets | Not applicable | Reproduce released 1B global thresholding without them |
| 128 ranks instead of 256 | Intentional, bounded | Preserve exact global patches/update with batch 8 and no accumulation; 64-rank/batch-16 was rejected by the measured memory gate |
| Pinned `terashuf` instead of an unpinned clone plus unsorted `find` | Reproducibility hardening, no algorithm change | Pin source/compiler/seed/memory and feed the exact source manifest order |
| LF insertion at DCLM archive boundaries | Correctness repair to stock public-data command | Add a delimiter only between files when the preceding archive lacks LF; record the count and require exact record-multiset parity |
| Hash-based shuffle | Contingency only, scientifically meaningful | Do not use unless measured `terashuf` gates fail; retain as an explicit ablation if used |
| Topology-matched derived chunk views | Systems-only correction for loader behavior | Repartition the same post-validation global stream without reshuffling; audit record multiset and checksums |
| Trainer's sticky `saved` flag suppresses later/final checkpoints | Correctness repair, no model-math change | Reset "saved this iteration" at each loop iteration; verify periodic, final, and restart checkpoints |
| `SequenceIterator` checkpoint omits its live shuffle buffer | Upstream limitation, scientifically relevant only after a restart | Prefer uninterrupted 100B jobs; record every restart and observed divergence; require an exact-state implementation or an explicit difference before the multi-allocation 4.53T run |
| 64-rank RCCL `INFO` plus recompile logs overloaded a nested Flux content store | Systems-only logging correction | Use `NCCL_DEBUG=WARN` and disable recompile logging for production; retain detailed logging only for targeted diagnosis |
| Batched entropy preprocessing | Deferred optimization, excluded from primary reproduction | Keep the exact serial scorer; revisit only as an explicit post-reproduction systems experiment |
| Arrow compression | Proposed lossless storage optimization | Use only after reader and checksum parity tests |

## Current validation status

Prior smoke testing established that ROCm, xFormers self-attention,
FlexAttention cross-attention, BF16 full-shard FSDP, optimizer updates, and
checkpoint machinery work on Tuolumne. A two-step exact released-shape test
instantiated all 4,533,879,808 parameters and reached approximately 35% of a
128 GiB MI300A at batch 4 per rank on four-way FSDP.

The first bounded DCLM rehearsal now also establishes:

| Measurement | Result |
| --- | ---: |
| Selected compressed source | 227,985,308 bytes |
| Complete JSONL records | 93,147 |
| UTF-8 text | 529,169,745 bytes |
| Complete JSONL stream | 685,922,949 bytes |
| Pinned `terashuf` + 64-way split + validation extraction | 11.78 s |
| Input/output multiset audit | Exact match in 3.55 s |
| Canonical training records after pilot validation | 86,747 |
| Canonical chunk text-byte balance | 5.98% CV; 1.259 max/min |
| Lossless 64-to-8 chunk derivation and read-back audit | 8.98 s |
| Lossless 64-to-4 chunk derivation and read-back audit | 3.65 s |

The subsequent multi-file entropy rehearsal is the test that exposed the
missing archive-boundary delimiters described above. Its failed `v1` artifact
is retained under `datasets/blt/failed/`; it is not a loader-visible training
root. The repaired immutable `dclm-entropy-rehearsal-1b-v2` result is:

| Measurement | Result |
| --- | ---: |
| Selected source archives | 10 |
| Selected compressed bytes | 2,932,084,691 |
| Pre-validation records | 1,171,806 |
| Pre-validation UTF-8 text bytes | 6,790,860,403 |
| Missing inter-file LF delimiters inserted | 9 |
| Shuffle, split, and validation extraction | 113.58 s |
| Validation records removed | 640,000 |
| Training records retained | 531,806 |
| Training UTF-8 text bytes retained | 3,074,249,866 |
| Canonical training-chunk text balance | 2.16% CV; 1.099 max/min |
| Source/output complete-record multiset audit | Exact; zero invalid JSON |
| Multiset audit time | 55.89 s |
| Lossless 64-to-8 view and embedded audit | Exact in 24.03 s |

The pilot used 100 validation records per chunk so that a single source
archive retained enough training data. Production uses the repository's
10,000 records per chunk. Its canonical shards contain only about 1,355
training documents apiece, so the measured text-byte imbalance is a
small-sample upper bound; the full materialization must still pass the same
balance audit before scoring.

Linear scaling of the original one-file pass gave a roughly 41-minute estimate
for 110B pre-validation text bytes. The more representative 10-file,
192-GiB-memory rehearsal projects 30.7 minutes from its 113.58-second runtime.
The pilot's complete-JSON/text ratio projects to about 143 GB of shuffle
input, below the frozen 192 GiB `terashuf` run size, so the initial full
preparation should remain a single shuffle run rather than an external merge.
Filesystem throughput can make the real time longer; allocate two hours and
retain the measured runtime instead of changing the shuffle algorithm or its
memory setting.

Four MI300A GPUs then scored one canonical chunk each with the released
99.5M-parameter entropy model. Every rank completed 1,356 documents in
40.1--42.4 seconds. The four files contained 30,606,954 text bytes and
produced 92,307,424 Arrow bytes. Validation proved exact ID, text, record
order, and entropy-length agreement for all 5,424 documents.

The scorer was then resumed across all 64 canonical chunks. Each of four
ranks processed its 15 remaining files in 465--478 seconds while reusing one
loaded model. The final view contains 86,747 records, 493,604,274 text bytes,
493,777,768 BLT positions, and 1,488,614,568 Arrow bytes. Full read-back
validation proved exact ID, text, record order, schema, finite entropy, and
entropy-length parity for every record.

Across the complete pilot, the released threshold `1.335442066192627`
produces 4.1722 bytes per patch. A threshold of `1.421875` produces 4.4908
bytes per patch, 0.20% from the 4.5 target; the residual is caused by 666,787
stored FP16 entropy values tied at that threshold. Final thresholds will be
recalibrated independently on the full official- and scratch-entropy views.

The initial cold measurement and the longer cached-model pass project a range
of approximately 1.7--2.4 hours for 100B text bytes on 64 GPUs, assuming file
balance and linear rank scaling. This step uses the repository's serial
per-document scorer independently on each GPU. No batched-document scoring
optimization is part of the primary recipe. Arrow size projects to
approximately 302 GB per entropy checkpoint, which is small relative to the
available dataset volume.

The pilot ratio estimates that the first 184 files in the frozen hash order
contain 110.32B text bytes. This is only a download-planning estimate. Before
shuffling, `scan-sources` measures exact text bytes in each downloaded archive
and freezes the shortest whole-file prefix that reaches 110B. The margin is
necessary because the stock convention removes 640,000 validation records
from a 64-chunk corpus. The post-shuffle audit is the authoritative gate that
at least 100B unique training bytes remain.

The checked-in entropy configuration has zero architecture or optimizer
field differences from the released nested entropy checkpoint parameters. The
main-model configuration likewise has zero architecture differences other
than its deliberately unset threshold, and zero optimizer differences from
`facebook/blt-1b`. Its intentional runtime changes are the 100B step budget,
paths, 128-rank/batch-8 placement, and disabled asynchronous evaluation.

The exact entropy architecture first completed a two-update training rehearsal
on four MI300A GPUs using the lossless 4-chunk raw-JSON view. Batch 16 per
rank preserves the released global batch of 64 sequences and 524,288 byte
positions per update while making four-way FSDP a conservative memory test.
Both updates had finite loss and gradients, no allocation retries or OOMs,
and approximately 51.0 GiB maximum active memory per GPU. The first update
took 17.32 seconds, including first-use overhead; the immediately warm update
took 0.666 seconds with 0.013 seconds attributed to data loading. These two
steps establish execution and shape parity, not convergence or sustained
throughput.

The subsequent released-topology rehearsal ran 1,908 updates on two nodes and
eight MI300A GPUs, consuming 1,000,341,504 byte positions without repeating
the bounded 8-chunk view. It completed successfully in 14.27 minutes including
model/data startup and one step-1000 checkpoint. Across logged steps 10--1900,
the median update time was 0.3391 seconds, median per-rank throughput was
188,030 byte positions/s, and median data wait was 0.0074 seconds. Peak active
and reserved memory were 25.72 and 27.76 GiB per GPU, with zero allocator
retries or OOMs. Interval BPB fell from 7.4375 at step 10 to 1.4297 at step
1900; this short curve is a convergence/throughput check, not a claim about
the released model's final quality.

The measured update time projects the released 100,000-update entropy recipe
to approximately 9.42 hours of training, plus about ten minutes for startup
and 100 periodic checkpoints. A 12-hour `pbatch` request is therefore the
measured starting allocation; retain a larger margin for the first production
attempt if queue policy makes requeue undesirable.

This sustained run also exposed an upstream checkpoint bookkeeping bug.
After saving at step 1000, `saved` remained true for the rest of training, so
the final-save block incorrectly skipped step 1908. The one-line
systems-correctness repair in `bytelatent/train.py` resets that flag at the
start of every loop iteration. A restart from the complete step-1000
checkpoint then completed steps 1001--1908 in 7.86 minutes and wrote a
complete step-1908 checkpoint: eight distributed state files, eight rank
train-state files all reporting step 1908, metadata, and parameters. The
repository consolidator produced a 1,194,302,761-byte `consolidated.pth`.
`load_entropy_model` loaded all 99,512,064 BF16 parameters from it on CPU,
which verifies the checkpoint format consumed by scratch-entropy scoring.

The unchanged GPU scorer then loaded this consolidated scratch checkpoint on
four ranks and processed one audited pilot chunk per rank in 39.2--42.8
seconds. Read-back validation matched all 5,424 IDs, texts, row positions, and
entropy lengths. On these four chunks, the 1B-byte rehearsal checkpoint
calibrates to threshold `1.765625` and 4.5079 bytes per patch; applying the
released fixed threshold gives only 3.2166 bytes per patch. This is expected
for a deliberately undertrained entropy model, but it proves that the scratch
checkpoint is a working scorer input and that official and scratch thresholds
must be calibrated independently.

The restart restores model, optimizer, scheduler, global step, learning rate,
and token count, but it does not reproduce the uninterrupted batch stream.
The repository itself marks the cause with
`TODO: need to also persist the current shuffle buffer` in
`SequenceIterator.get_state()`: file row and NumPy RNG state are serialized,
whereas the live 512-sequence permutation, unconsumed sequences, and any
partial next buffer are generator locals. At step 1000, each rank had consumed
8000 sequences, or 320 positions into its current 512-sequence buffer, so as
many as 192 complete buffered sequences per rank plus partial spill could be
skipped when rebuilding. Logged post-restart losses therefore differ from the
uninterrupted run even though LR and total-token fields agree exactly.

This limitation is not patched opportunistically because doing so requires a
state-format and iterator-lifecycle refactor, and the released main-model
configuration explicitly uses approximate asynchronous persistence. The
planned 100B entropy and main runs should use single uninterrupted `pbatch`
allocations when possible. Any actual restart must be recorded; the final
4.53T stage is gated on implementing and validating exact shuffle-buffer
serialization or explicitly accepting this data-order difference.

A corresponding four-way-FSDP main-model control passed at batch 8 per rank.
Its two updates had finite loss and gradients, no allocator retries or OOMs,
67.70 GiB maximum active memory and 80.04 GiB maximum reserved memory per
GPU. The first update took 40.32 seconds after a 122.20-second initial data
wait; the warm update took 4.83 seconds with a 0.007-second data wait. At 128
ranks, batch 8 preserves the released 4,194,304 patches per optimizer update
exactly, so 32 nodes is already a viable unchanged-math fallback.

The topology tests were deliberately incremental. The first successful
main-model test used the released batch 4 and approximately 35% of one MI300A.
Batch 8 then passed with the memory figures above. Batch 16 never passed on
the small topology: it immediately exhausted node-wide unified memory.
Because four-way FSDP holds sixteen times the model-state shard of 64-way
FSDP, that failure could not by itself distinguish model-state pressure from
batch-scaled activation and mask pressure. The direct 64-rank test below was
therefore necessary, but its result shows that describing 16 nodes as a
candidate—not a preferred topology—was the appropriate posture.

The failure site explains why extra FSDP ranks were insufficient. FSDP shards
parameters, gradients, and optimizer state, but not rank-local activations or
the masks built from each local batch. At the configured maxima,
`cross_attn_mask` first constructs a dense Boolean precursor with shape
`[batch, 24576, 4096 * 2]`: 3,221,225,472 elements, or 3.0 GiB before
FlexAttention block metadata, at batch 16 on every rank. Both encoder and
decoder cross-attention build this orientation, and ordinary decoder
activations also scale with the local batch. Halving to batch 8 directly
reduces this unsharded pressure; moving from four-way to 64-way FSDP does not.

Each tiny pilot shard contains fewer than the loader's
`512 * 4096`-patch shuffle-buffer target, so `LoopingIterator` must revisit
the shard while constructing the first buffer. The 122-second cold data time
is therefore a pilot artifact, not a production throughput estimate. The
near-zero warm data time proves that the unchanged asynchronous prefetch path
can feed these bounded updates once initialized.

The enhanced entropy audit quantifies this effect at the calibrated threshold.
Whole shards contain 1,520,461--1,988,463 patches. More importantly for the
selected topology, the validator reproduces BLT's two strided Arrow workers
per chunk: the resulting 128 rank streams contain 702,376--1,079,218 patches,
with mean 859,013.48, 8.56% CV, and 1.537 max/min ratio. All are below the
2,097,152 patches required for one complete shuffle buffer, so the pilot has
zero buffer-safe optimizer updates before repetition even though nominal
rank-stream capacity is 21--32 updates. This does not invalidate the bounded
systems tests; it does prohibit using their loss curve as a data-unique
training result. Production requires 83 complete buffers, or 174,063,616
patches, in every one of the 128 streams.

The analogous batch-16 test did not reach an optimizer update on four-way
FSDP. All ranks completed model construction and production asynchronous
loader startup, after which Flux reported a node memory-cgroup OOM and a
52.51 GiB host-memory peak. A separate in-allocation probe measured
512,038,322,176 bytes of host/cgroup capacity and four 128 GiB MI300A
devices. On MI300A, CPU and accelerator allocations consume the same
node-wide HBM pool; the failure was therefore not evidence of an artificial
52 GiB scheduler cap or an individual PyTorch GPU OOM.

Because four-way sharding leaves much larger model-state shards than the
candidate production topology, batch 16 was also tested directly on 16 nodes
and 64 ranks. Every rank constructed the complete 4,533,879,808-parameter
model, optimizer, FSDP mesh, production asynchronous loader, and its mapping
to one of the 64 canonical entropy-scored chunks. No rank reached an optimizer
update. During the first forward pass, several nodes reported memory-cgroup
OOM kills with measured host peaks of 52.58--54.36 GiB; a surviving rank ended
in `MemoryError` while `create_block_mask` was constructing the decoder
cross-attention mask in `bytelatent/model/blt.py`. This reproduces the
four-rank batch-16 failure after removing the confounding large debug log and
decisively rejects the 16-node placement under the unchanged implementation.

The primary 32-node/128-rank/batch-8 candidate leaves the global batch,
sequence shape, optimizer cadence, data records, stored entropy values, and
gradient-accumulation count unchanged. BLT assigns two strided workers to
each existing canonical Arrow chunk, so this fallback requires no data
reshard. Its bounded 128-rank collective test is queued as Flux job
`f3NoWX2xK1hH`.

The released 64-node/256-rank/batch-4 fallback is independently queued as
Flux job `f3NoqY9Bkvij`. It uses the same two-update, no-checkpoint rehearsal
and the same canonical pilot data, but assigns four strided workers to each
of the 64 Arrow chunks. The corresponding world-256 audit contains 256 rank
streams with 335,732--600,807 patches, mean 429,506.74, 11.89% CV, and 1.790
max/min ratio. The pilot is intentionally too small for a complete shuffle
buffer, but the topology-aware preflight passes every required recipe check;
its report is
`tuo-runs/blt-repro-1b-preflight-v1/preflight-pilot-world256.json`.

No 100B model-training run is authorized until at least one bounded topology
gate reaches both optimizer steps and exits cleanly. If both pass, prefer the
32-node candidate for schedulability while recording its topology difference;
the 64-node result provides the exact released placement comparison.

The full data-preparation driver was submitted independently as one-node,
six-hour `pbatch` job `f3NohQp6stes` (the unstarted 12-hour request
`f3Nof4C2UZxb` was canceled to improve backfill). It downloads and
SHA-verifies a 200-file planning envelope, freezes the shortest prefix
reaching 110B pre-validation text bytes, materializes only that prefix with
pinned `terashuf`, requires exact record-multiset parity and at least 100B
post-validation training text, and then derives audited 8- and 4-chunk
views. At the time of this note the job is pending; no long model-training
run has started.
