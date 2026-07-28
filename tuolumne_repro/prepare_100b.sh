#!/bin/bash
# Fail-fast driver for the immutable DCLM 100B materialization.
#
# Run on one complete Tuolumne node from the BLT conda environment. Each
# underlying command writes its own checksummed manifest. An interrupted run
# preserves its incomplete staging directory for diagnosis and never replaces
# a completed materialization.

set -euo pipefail

repo_root=/p/vast1/kirchenb/hlm-root/blt
data_root=/p/vast1/pretrain/datasets/blt
materialization=dclm-100b-v1
planning_source_files=200
target_prevalidation_text_bytes=110000000000
minimum_training_text_bytes=100000000000
revision=a3b142c183aebe5af344955ae20836eb34dcf69b
scan_path="$data_root/manifests/dclm_baseline_1.0/$revision/source_scan_first_00200.json"
materialization_root="$data_root/prepared/$materialization"
audit_path="$materialization_root/audit.json"

cd "$repo_root"

if [[ -e "$materialization_root" ]]; then
    echo "Refusing to overwrite existing materialization: $materialization_root" >&2
    exit 2
fi

echo "Downloading and verifying the first $planning_source_files frozen sources"
python -u tuolumne_repro/data_pipeline.py download \
    --source-files "$planning_source_files"

echo "Scanning exact decompressed records and text bytes"
python -u tuolumne_repro/data_pipeline.py scan-sources \
    --source-files "$planning_source_files" \
    --workers 96 \
    --target-text-bytes "$target_prevalidation_text_bytes"

selected_source_files=$(
    jq -er \
        --argjson maximum "$planning_source_files" \
        '.recommended_source_files
         | select(type == "number" and . >= 1 and . <= $maximum)' \
        "$scan_path"
)
scanned_text_bytes=$(jq -er '.scanned_utf8_text_bytes' "$scan_path")
if (( scanned_text_bytes < target_prevalidation_text_bytes )); then
    echo "Planning envelope did not reach the text target: $scanned_text_bytes" >&2
    exit 3
fi

echo "Selected shortest frozen prefix: $selected_source_files source files"
python -u tuolumne_repro/data_pipeline.py prepare \
    --materialization "$materialization" \
    --source-files "$selected_source_files" \
    --n-chunks 64 \
    --validation-per-chunk 10000 \
    --memory-gib 192

echo "Auditing complete-record multiset identity"
python -u tuolumne_repro/data_pipeline.py audit \
    --materialization "$materialization" \
    --workers 96

jq -e '.comparison.passed == true' "$audit_path" >/dev/null
training_text_bytes=$(
    jq -er \
        '[.output.files[]
          | select(.path | contains("/dclm_baseline_1.0/"))
          | .utf8_text_bytes] | add' \
        "$audit_path"
)
if (( training_text_bytes < minimum_training_text_bytes )); then
    echo "Post-validation training text is too small: $training_text_bytes" >&2
    exit 4
fi

echo "Deriving lossless entropy-training and one-node views"
python -u tuolumne_repro/data_pipeline.py derive-view \
    --materialization "$materialization" \
    --target-chunks 8 \
    --audit-workers 96
python -u tuolumne_repro/data_pipeline.py derive-view \
    --materialization "$materialization" \
    --target-chunks 4 \
    --audit-workers 96

jq -e '.audit.passed == true' \
    "$materialization_root/views/world_0008/view.json" >/dev/null
jq -e '.audit.passed == true' \
    "$materialization_root/views/world_0004/view.json" >/dev/null

echo "DCLM 100B materialization passed all preparation gates"
jq -n \
    --arg materialization "$materialization" \
    --argjson planning_source_files "$planning_source_files" \
    --argjson selected_source_files "$selected_source_files" \
    --argjson scanned_text_bytes "$scanned_text_bytes" \
    --argjson training_text_bytes "$training_text_bytes" \
    '{
      materialization: $materialization,
      planning_source_files: $planning_source_files,
      selected_source_files: $selected_source_files,
      scanned_text_bytes: $scanned_text_bytes,
      post_validation_training_text_bytes: $training_text_bytes,
      record_multiset_audit: "passed",
      world_0008_view_audit: "passed",
      world_0004_view_audit: "passed"
    }'
