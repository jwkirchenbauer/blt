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
stage="${1:-all}"

cd "$repo_root"

case "$stage" in
    all | download-scan | materialize | audit-views) ;;
    *)
        echo \
            "Usage: $0 [all|download-scan|materialize|audit-views]" \
            >&2
        exit 2
        ;;
esac

load_selection() {
    if [[ ! -f "$scan_path" ]]; then
        echo "Missing completed source scan: $scan_path" >&2
        exit 3
    fi
    selected_source_files=$(
        jq -er \
            --argjson maximum "$planning_source_files" \
            '.recommended_source_files
             | select(type == "number" and . >= 1 and . <= $maximum)' \
            "$scan_path"
    )
    scanned_text_bytes=$(jq -er '.scanned_utf8_text_bytes' "$scan_path")
    if (( scanned_text_bytes < target_prevalidation_text_bytes )); then
        echo \
            "Planning envelope did not reach the text target: $scanned_text_bytes" \
            >&2
        exit 3
    fi
    echo "Selected shortest frozen prefix: $selected_source_files source files"
}

download_and_scan() {
    echo \
        "Downloading and verifying the first $planning_source_files frozen sources"
    python -u tuolumne_repro/data_pipeline.py download \
        --source-files "$planning_source_files"

    echo "Scanning exact decompressed records and text bytes"
    python -u tuolumne_repro/data_pipeline.py scan-sources \
        --source-files "$planning_source_files" \
        --workers 96 \
        --target-text-bytes "$target_prevalidation_text_bytes"
    load_selection
}

materialize() {
    load_selection
    if [[ -e "$materialization_root" ]]; then
        jq -e \
            --arg materialization "$materialization" \
            --argjson source_files "$selected_source_files" \
            '.materialization == $materialization
             and .dataset.source_files == $source_files
             and .split.n_chunks == 64
             and .split.validation.records_per_chunk_requested == 10000
             and .shuffle.implementation == "terashuf"
             and .shuffle.seed == 42
             and .shuffle.memory_gib == 192' \
            "$materialization_root/materialization.json" >/dev/null
        echo "Using existing immutable materialization: $materialization_root"
        return
    fi

    python -u tuolumne_repro/data_pipeline.py prepare \
        --materialization "$materialization" \
        --source-files "$selected_source_files" \
        --n-chunks 64 \
        --validation-per-chunk 10000 \
        --memory-gib 192
}

ensure_view() {
    local target_chunks="$1"
    local view_name
    view_name=$(printf 'world_%04d' "$target_chunks")
    local view_root="$materialization_root/views/$view_name"
    local view_manifest="$view_root/view.json"

    if [[ -f "$view_manifest" ]]; then
        jq -e \
            --argjson target_chunks "$target_chunks" \
            '.target_chunks == $target_chunks and .audit.passed == true' \
            "$view_manifest" >/dev/null
        echo "Using existing audited view: $view_root"
        return
    fi
    if [[ -e "$view_root" ]]; then
        echo "Refusing incomplete or unverified view: $view_root" >&2
        exit 5
    fi

    python -u tuolumne_repro/data_pipeline.py derive-view \
        --materialization "$materialization" \
        --target-chunks "$target_chunks" \
        --audit-workers 96
    jq -e '.audit.passed == true' "$view_manifest" >/dev/null
}

audit_and_derive_views() {
    load_selection
    if [[ ! -f "$materialization_root/materialization.json" ]]; then
        echo "Missing completed materialization: $materialization_root" >&2
        exit 4
    fi

    if [[ -f "$audit_path" ]] && \
        jq -e '.comparison.passed == true' "$audit_path" >/dev/null
    then
        echo "Using existing successful multiset audit: $audit_path"
    else
        echo "Auditing complete-record multiset identity"
        python -u tuolumne_repro/data_pipeline.py audit \
            --materialization "$materialization" \
            --workers 96
    fi

    jq -e '.comparison.passed == true' "$audit_path" >/dev/null
    training_text_bytes=$(
        jq -er \
            '[.output.files[]
              | select(.path | contains("/dclm_baseline_1.0/"))
              | .utf8_text_bytes] | add' \
            "$audit_path"
    )
    if (( training_text_bytes < minimum_training_text_bytes )); then
        echo \
            "Post-validation training text is too small: $training_text_bytes" \
            >&2
        exit 4
    fi

    echo "Deriving lossless entropy-training and one-node views"
    ensure_view 8
    ensure_view 4

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
}

case "$stage" in
    all)
        download_and_scan
        materialize
        audit_and_derive_views
        ;;
    download-scan)
        download_and_scan
        ;;
    materialize)
        materialize
        ;;
    audit-views)
        audit_and_derive_views
        ;;
esac
