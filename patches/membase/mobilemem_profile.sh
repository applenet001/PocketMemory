# MobileMem smoke20 profile. Sourced after the scripts enter the MemBase directory.
if [[ "$DATASET" == "mobilemem" ]]; then
    case "$METHOD" in rag|long_context|mem0) ;; *) echo "Unsupported MobileMem method: $METHOD" >&2; exit 1;; esac
    config_path="../config/mobilemem-qwen/${METHOD}_traced_smoke20.json"
    dataset_path="../data/text/smoke20/mobilemem_data.json"
    if [[ -z "${PYTHON:-}" ]]; then
        if [[ -x ../pocketmemory/bin/python ]]; then
            PYTHON=../pocketmemory/bin/python
        else
            PYTHON=../pocketmemory/Scripts/python.exe
        fi
    fi
    sample_size=$("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1],encoding="utf-8"))))' "$dataset_path")
    start_idx=0
    end_idx="$sample_size"
    top_k=30
    [[ "$METHOD" != "long_context" ]] || top_k=1
    qa_model="deepseek-flash"
    judge_model="deepseek-flash"
    qa_batch_size=1
    judge_batch_size=1
    api_config_path="../outputs/private/deepseek-api.json"
    export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 NO_PROXY=localhost,127.0.0.1,::1
    export no_proxy="$NO_PROXY"
    export MEMORY_METHOD="$METHOD"
    export DEEPSEEK_USAGE_LOG="../outputs/mobilemem-qwen/traced-smoke20/${METHOD}/deepseek_usage.jsonl"
    # Mem0 needs an OpenAI-compatible credential as well as the evaluation API config.
    export OPENAI_API_KEY
    OPENAI_API_KEY=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["api_keys"][0])' "$api_config_path")
fi
PYTHON="${PYTHON:-python}"
