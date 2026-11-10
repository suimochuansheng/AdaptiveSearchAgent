#!/usr/bin/env bash
# =============================================================================
# run_ragas_baseline.sh — 一键跑 RAGAS 基准线（换模型后的对比基线）
#
# 用法:
#   ./evals/run_baseline.sh                 # 默认：v3 + bge-m3 + BGE 精排 + eval_v3 专用库
#   ./evals/run_baseline.sh --dry-run       # 只做预检，不调 LLM、不花钱
#   ./evals/run_baseline.sh --smoke         # 2 题冒烟（验证链路，几分钟）
#   RERANKER=ms-marco ./evals/run_baseline.sh  # 只取 RRF，不做 BGE 精排
#   DATASET=evals/datasets/qa_dataset_v4.yaml ./evals/run_baseline.sh  # 24 题优化版（v3 冻结）
#   TAG=my_baseline ./evals/run_baseline.sh # 自定义归档目录名
#   ./evals/run_baseline.sh --tag my_baseline  # 同上，作为命令行参数（推荐，不易写错）
#
# 常用环境变量:
#   MODEL=bge-m3                  仅 bge-m3（nomic 已下线）
#   DATASET=<yaml 路径>           默认 evals/datasets/qa_dataset_v3.yaml
#   KB_ID=eval_v3                 黄金集专用库（6 篇，171 片）
#   TOP_K=3                       默认 3；脚本对 bge 会放大到 3×TOP_K，网关 hybrid 再×3 返回
#   MAX_WORKERS=2                 Ragas 并发（默认 2：4 并发下 RAG 偶发空检索）
#   TIMEOUT=120                   单任务超时（秒）
#   RERANKER=ms-marco|bge         默认 bge（BGE 精排；首次会加载 2.2G 模型）
#   EVAL_JUDGE_MODEL=<model>      裁判模型（默认取 .env 的 ALIYUN_QWEN_MODEL_NAME）
#   TAG=<归档名>                  默认 时间戳_model_reranker_dataset；实际目录为 <TAG>_<YYYYmmdd-HHMM>
#   INGEST_URL=http://localhost:8100   评测脚本调用的 RAG 网关（必须是已启动的 ingest_dev）
#
# 前置条件：RAG 网关必须已经在跑，脚本不会自动兜底：
#   cd /home/ubhuazhu/dev_pros && docker compose up -d ingest_dev
#   curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8100/health   # 期望 200
# 若网关不可用，脚本直接报错退出（不再启动 WSL 原生网关——开发环境统一走 ingest_dev）。
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL="${MODEL:-bge-m3}"
DATASET="${DATASET:-evals/datasets/qa_dataset_v3.yaml}"
KB_ID="${KB_ID:-eval_v3}"
TOP_K="${TOP_K:-3}"
# 默认 2 而不是 4：4 并发下 RAG 子服务会偶发返回空结果（空检索 1 次），
# 而基线判据要求"空检索 0"，否则基线本身不可复现。
MAX_WORKERS="${MAX_WORKERS:-2}"
TIMEOUT="${TIMEOUT:-120}"
RERANKER="${RERANKER:-bge}"
TAG="${TAG:-}"
INGEST_URL="${INGEST_URL:-http://localhost:8100}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
REPORT_DIR="${EVAL_REPORT_DIR:-$PROJECT_DIR/../文档/项目/AdaptiveSearchAgent/评测报告}"

DRY_RUN=0
SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke)   SMOKE=1; shift ;;
    --tag)
      [ $# -ge 2 ] || { echo "--tag 需要一个名字，如 --tag baseline_v2" >&2; exit 2; }
      TAG="$2"; shift 2 ;;
    --tag=*) TAG="${1#*=}"; shift ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（可用 --dry-run / --smoke / --tag <名字> / --help）" >&2; exit 2 ;;
  esac
done

if [ "$SMOKE" = "1" ]; then
  DATASET="evals/datasets/qa_dataset_smoke.yaml"
  MAX_WORKERS=2
  TAG="${TAG:-smoke_$(date +%Y%m%d_%H%M%S)}"
fi
[ -n "$TAG" ] || TAG="$(date +%Y%m%d_%H%M%S)_${MODEL}_${RERANKER}_ollama_$(basename "$DATASET" .yaml)"
TAGDIR="$REPORT_DIR/${TAG}_$(date +%Y%m%d-%H%M)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── 输出 ─────────────────────────────────────────────────────────────────────
c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_cyn=$'\033[36m'; c_rst=$'\033[0m'
log()  { printf '%s[%s]%s %s\n' "$c_cyn" "$(date +%H:%M:%S)" "$c_rst" "$*"; }
warn() { printf '%s[WARN]%s %s\n' "$c_ylw" "$c_rst" "$*" >&2; }
die()  { printf '%s[FAIL]%s %s\n' "$c_red" "$c_rst" "$*" >&2; exit 1; }

# ── 预检工具 ─────────────────────────────────────────────────────────────────
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"; }

# probe_search <url>：0=正常 2=空结果 3=HTTP错误 4=连不上
probe_search() {
  python3 - "$1" "$KB_ID" "$MODEL" <<'PY'
import json, sys, urllib.error, urllib.request
base = sys.argv[1].rstrip("/") + "/api/v1/search"
payload = {
    "query": "PostgreSQL 的写入吞吐量是多少？",
    "top_k": 1,
    "kb_id": sys.argv[2],
    "model": sys.argv[3],
    "search_mode": "hybrid",
    "skip_reranker": True,
}
req = urllib.request.Request(
    base, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.load(resp).get("result", "")
    sys.exit(0 if result.strip() else 2)
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", "ignore")[-500:]
    print(f"HTTP {e.code}: {body}", file=sys.stderr)
    sys.exit(3)
except Exception as e:  # noqa: BLE001
    print(str(e), file=sys.stderr)
    sys.exit(4)
PY
}

ollama_has() {
  python3 - "$OLLAMA_URL" "$1" <<'PY'
import json, sys, urllib.request
url, model = sys.argv[1].rstrip("/") + "/api/tags", sys.argv[2]
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        names = [m.get("name", "") for m in json.load(r).get("models", [])]
except Exception as e:  # noqa: BLE001
    print(f"无法访问 Ollama {url}: {e}", file=sys.stderr)
    sys.exit(4)
sys.exit(0 if any(n == model or n.split(":")[0] == model for n in names) else 3)
PY
}

env_val() {  # env_val KEY：从 .env 读取（不打印敏感值）
  [ -f "$PROJECT_DIR/.env" ] || return 0
  grep -E "^${1}=" "$PROJECT_DIR/.env" | tail -1 | cut -d= -f2- | sed -E 's/[[:space:]]+#.*$//' || true
}

# 裁判名唯一来源：EVAL_JUDGE_MODEL（环境变量 > .env）→ 回退 ALIYUN_QWEN_MODEL_NAME。
# 与 config.py 的 Settings.EVAL_JUDGE_MODEL 默认值保持一致。
judge_model() {
  local m="${EVAL_JUDGE_MODEL:-$(env_val EVAL_JUDGE_MODEL)}"
  [ -n "$m" ] || m="$(env_val ALIYUN_QWEN_MODEL_NAME)"
  printf '%s' "$m"
}

# ── 1. 基础预检 ──────────────────────────────────────────────────────────────
log "项目目录: $PROJECT_DIR"
need_cmd poetry
need_cmd python3
need_cmd curl

[ -f "$DATASET" ] || die "黄金集不存在: $DATASET（DATASET 需相对项目根或绝对路径）"
[ -f "$PROJECT_DIR/evals/rag_quality.py" ] || die "缺少 evals/rag_quality.py"
[ -d "$REPORT_DIR" ] || mkdir -p "$REPORT_DIR"
[ -w "$REPORT_DIR" ] || die "报告目录不可写: $REPORT_DIR"

case "$MODEL" in bge-m3) ;; *) die "MODEL 只能是 bge-m3（nomic-embed 已下线，当前: $MODEL）" ;; esac
case "$RERANKER" in ms-marco|bge) ;; *) die "RERANKER 只能是 ms-marco 或 bge（当前: $RERANKER）" ;; esac

OLLAMA_EMBED="bge-m3"

# 脚本内 retrieval_top_k = TOP_K（ms-marco）或 3×TOP_K（bge）；
# 网关 hybrid 再取 3 倍候选并全部返回（skip_reranker=True），所以最终上下文切片数约 3×retrieval_top_k。
if [ "$RERANKER" = "bge" ]; then RETRIEVAL_K=$((TOP_K * 3)); else RETRIEVAL_K="$TOP_K"; fi
CTX_MAX=$((RETRIEVAL_K * 3))

log "检查 Poetry 环境（ragas / datasets / torch）..."
poetry run python -c "import ragas, datasets, torch" >/dev/null 2>&1 \
  || die "Poetry 环境缺依赖，请先 cd $PROJECT_DIR && poetry install"

QUESTION_COUNT="$(poetry run python - "$DATASET" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    data = yaml.safe_load(f) or []
print(len(data))
PY
)"
[ "$QUESTION_COUNT" -gt 0 ] || die "黄金集为空: $DATASET"

log "黄金集: $DATASET（${QUESTION_COUNT} 题）"
log "模型: $MODEL (Ollama: $OLLAMA_EMBED) | KB: $KB_ID | TOP_K: $TOP_K | Reranker: $RERANKER"
log "答案生成: 本地 $(env_val OLLAMA_MODEL_NAME) | Ragas 裁判: $(judge_model)"

# ── 2. 关键告警（不阻断，但请先读）───────────────────────────────────────────
if [ "$RERANKER" = "bge" ]; then
  warn "RERANKER=bge：候选 retrieval_top_k=${RETRIEVAL_K}，网关 hybrid 最多返回 3×=${CTX_MAX} 片，"
  warn "再用 BAAI/bge-reranker-v2-m3 精排回 TOP_K=${TOP_K} 片（首次加载 2.2G 模型，CPU 约 10~30s）。"
  warn "注意：2026-10-12 的旧 bge 基线因解析 bug 实际没精排；本期是修复后的新基线序列，不与之直接横比。"
fi
case "$DATASET" in
  *qa_dataset_v2.yaml)
    warn "v2 黄金集含已知无效题 q015（HITL 在 test_rag_spec.pdf 中不存在），且 20 题全压在单文档；"
    warn "建议默认 v3（evals/datasets/qa_dataset_v3.yaml）。本次仍按你指定的 v2 执行。"
    ;;
esac

# 生成器固定为本地 Ollama（被测 RAG 系统），裁判固定为 Qwen API
log "生成器: 本地 $(env_val OLLAMA_MODEL_NAME) | 裁判: $(judge_model)"

if [ -z "$(env_val ALIYUN_QWEN_API_KEY)" ]; then
  die ".env 里 ALIYUN_QWEN_API_KEY 为空，Ragas 无法调用裁判 LLM"
fi

log "检查 Ollama embedding 模型: $OLLAMA_EMBED ..."
if ollama_has "$OLLAMA_EMBED"; then
  log "Ollama OK: $OLLAMA_URL ($OLLAMA_EMBED)"
else
  warn "Ollama 没有模型 $OLLAMA_EMBED，先执行: ollama pull $OLLAMA_EMBED"
fi

mkdir -p "$TAGDIR"

log "探测 RAG 网关: $INGEST_URL ..."
PROBE_RC=0
probe_search "$INGEST_URL" || PROBE_RC=$?
if [ "$PROBE_RC" -eq 0 ]; then
  log "RAG 网关检索正常: $INGEST_URL"
else
  if [ "$PROBE_RC" -eq 3 ]; then
    warn "RAG 网关返回 500（常见原因：容器内 OLLAMA_BASE_URL=host.docker.internal:11435，"
    warn "而 ollama_relay 没起来，容器到不了宿主机的 11434）"
  fi
  die "RAG 网关不可用: $INGEST_URL（probe rc=$PROBE_RC）

基线必须走 ingest_dev（容器里的 RAG 路由转发器）。脚本已不再自动启动
WSL 原生网关兜底，因为开发环境统一走容器这条路。请先起容器再重跑：

  cd /home/ubhuazhu/dev_pros && docker compose up -d ingest_dev
  curl -s -o /dev/null -w '%{http_code}\\n' http://localhost:8100/health   # 期望 200
"
fi

# ── 4. 试运行：只打印解析后的配置 ────────────────────────────────────────────
if [ "$DRY_RUN" = "1" ]; then
  cat <<EOF

${c_grn}== DRY RUN 预检通过，未调用任何 LLM ==${c_rst}
  黄金集      : $DATASET ($QUESTION_COUNT 题)
  Embedding   : $MODEL ($OLLAMA_EMBED)
  Reranker    : $RERANKER
  KB_ID       : $KB_ID
  TOP_K       : $TOP_K（候选 retrieval_top_k=$RETRIEVAL_K → 网关最多 3×=$CTX_MAX 片 → BGE 精排回 $TOP_K 片）
  答案生成    : 本地 $(env_val OLLAMA_MODEL_NAME)
  裁判        : $(judge_model)
  并发 workers: $MAX_WORKERS
  RAG 子服务  : $INGEST_URL
  报告归档    : $TAGDIR
  执行命令    : INGEST_SERVICE_URL=$INGEST_URL poetry run python evals/rag_quality.py \\
                  --model $MODEL --dataset $DATASET --kb_id $KB_ID --top_k $TOP_K \\
                  --max_workers $MAX_WORKERS --timeout $TIMEOUT --reranker $RERANKER
EOF
  rmdir "$TAGDIR" 2>/dev/null || true
  exit 0
fi

# ── 5. 归档旧报告，避免被同名覆盖 ────────────────────────────────────────────
if [ "$RERANKER" = "ms-marco" ]; then SUFFIX="${MODEL}_ollama"; else SUFFIX="${MODEL}_${RERANKER}_ollama"; fi
PREV_DIR="$REPORT_DIR/_prev_${TIMESTAMP}"
for ext in json csv; do
  if [ -f "$REPORT_DIR/ragas_evaluation_${SUFFIX}.${ext}" ]; then
    mkdir -p "$PREV_DIR"
    cp "$REPORT_DIR/ragas_evaluation_${SUFFIX}.${ext}" "$PREV_DIR/"
  fi
done
[ -d "$PREV_DIR" ] && log "旧报告已备份: $PREV_DIR"

# ── 6. 跑评测 ────────────────────────────────────────────────────────────────
log "开始评测（${QUESTION_COUNT} 题 × 3 指标；预计 $((QUESTION_COUNT * 25 / 60 + 3))~$((QUESTION_COUNT * 40 / 60 + 5)) 分钟，请勿中断）..."
RUN_LOG="$TAGDIR/run.log"
set +e
INGEST_SERVICE_URL="$INGEST_URL" poetry run python evals/rag_quality.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --kb_id "$KB_ID" \
  --top_k "$TOP_K" \
  --max_workers "$MAX_WORKERS" \
  --timeout "$TIMEOUT" \
  --reranker "$RERANKER" 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then
  die "评测失败（exit=$RC），日志见 $RUN_LOG"
fi

# ── 7. 归档本次报告 + 元信息 ─────────────────────────────────────────────────
JSON_SRC="$REPORT_DIR/ragas_evaluation_${SUFFIX}.json"
CSV_SRC="$REPORT_DIR/ragas_evaluation_${SUFFIX}.csv"
[ -f "$JSON_SRC" ] || die "未找到预期报告: $JSON_SRC（脚本输出名可能变了）"
cp "$JSON_SRC" "$TAGDIR/"
[ -f "$CSV_SRC" ] && cp "$CSV_SRC" "$TAGDIR/"

# smoke 结果只进归档目录，不覆盖根目录的正式基线
if [ "$SMOKE" = "1" ]; then
  if [ -n "${PREV_DIR:-}" ] && [ -d "$PREV_DIR" ]; then
    cp "$PREV_DIR"/ragas_evaluation_${SUFFIX}.* "$REPORT_DIR"/ 2>/dev/null || true
    log "smoke 报告已归档到 $TAGDIR，根目录恢复为上一份正式基线"
  else
    rm -f "$REPORT_DIR/ragas_evaluation_${SUFFIX}.json" "$REPORT_DIR/ragas_evaluation_${SUFFIX}.csv"
    log "smoke 报告已归档到 $TAGDIR，根目录未保留 smoke 结果"
  fi
fi

GIT_REV="$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "$PROJECT_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
DATASET_SHA="$(sha256sum "$DATASET" | cut -d' ' -f1)"
SCRIPT_SHA="$(sha256sum "$PROJECT_DIR/evals/rag_quality.py" | cut -d' ' -f1)"
cat > "$TAGDIR/run_meta.txt" <<EOF
时间          : $(date -Iseconds)
项目 commit   : $GIT_REV (dirty files: $GIT_DIRTY)
黄金集        : $DATASET ($QUESTION_COUNT 题, sha256=$DATASET_SHA)
评测脚本      : evals/rag_quality.py (sha256=$SCRIPT_SHA)
Embedding     : $MODEL / $OLLAMA_EMBED
Reranker      : $RERANKER
KB_ID         : $KB_ID
TOP_K         : $TOP_K（候选 retrieval_top_k=$RETRIEVAL_K，网关最多 3×=$CTX_MAX 片；BGE 精排回 $TOP_K 片）
裁判 LLM      : $(judge_model)（EVAL_JUDGE_MODEL，默认取 ALIYUN_QWEN_MODEL_NAME）
生成 LLM      : 本地 $(env_val OLLAMA_MODEL_NAME)
项目 Agent LLM: $(env_val OLLAMA_MODEL_NAME)
RAG 子服务    : $INGEST_URL
并发 workers  : $MAX_WORKERS
报告前缀      : ragas_evaluation_${SUFFIX}
EOF

# 控制台再打一遍核心指标
# smoke 的根目录报告会被恢复/删除，汇总改读归档副本
SUMMARY_JSON="$JSON_SRC"
[ -f "$TAGDIR/ragas_evaluation_${SUFFIX}.json" ] && SUMMARY_JSON="$TAGDIR/ragas_evaluation_${SUFFIX}.json"
poetry run python - "$SUMMARY_JSON" "$KB_ID" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("\n" + "=" * 64)
print(f"基准线汇总: {d.get('model_evaluated')} | reranker={d.get('reranker')} | "
      f"题数={d.get('total_questions')} | KB={sys.argv[2]}")
for k, v in d.get("metrics", {}).items():
    print(f"  {k:18s} mean={v['mean']:.4f}  median={v['median']:.4f}  min={v['min']:.4f}")
print(f"  文档 top-1 归因: {d.get('doc_attribution_accuracy')}  "
      f"| 检索异常: {d.get('retrieval_errors')} 次")
print("=" * 64)
PY

log "完成。报告目录: $TAGDIR"
log "  - $(basename "$SUMMARY_JSON")"
log "  - run.log / run_meta.txt"
