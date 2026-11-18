#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../MemBase"
export DATASET=mobilemem WAIT_FOR_COMPLETION=1
for method in rag long_context mem0; do
  export METHOD="$method"
  for stage in construction search evaluation; do
    if [[ "${RESUME_RAG_EVALUATION:-0}" == "1" && "$method" == "rag" && "$stage" != "evaluation" ]]; then continue; fi
    echo "Starting $method / $stage"
    bash "examples/trace_memory_lifecycle_with_membase/run_traced_${stage}.sh"
  done
done
