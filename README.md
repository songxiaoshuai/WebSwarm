# WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide Web Search

## 🔍 Overview

**WebSwarm is a recursive multi agent orchestration framework for complex web search tasks that require both deep reasoning and broad information coverage.** Rather than relying on a fixed decomposition plan, WebSwarm progressively builds a delegation tree as new evidence is discovered. Each search node combines a local objective with one of four search modes, including `atom`, `deep`, `wide`, and `entity_collect`. A node can solve its objective directly or create specialized child nodes, while returned results guide further expansion, revision, and aggregation. WebSwarm also uses lightweight web structure probing to align task decomposition with how information is organized online. For batches of similar subtasks, it extracts reusable process experience from a small number of scout nodes and shares it with the remaining sibling nodes. 

![alt text](Figs/method_overall.png)

## 🚀 Quick Start

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Create and configure a `.env` file in the project root with the required service URLs and API keys. Then select `MODEL`, `BENCHMARK`, `TASK_IDS`, and other settings at the top of `experiment_config.py`, and run:

```bash
python3 run_main.py
```

To use a local search/fetch service, set `SEARCH_ENGINE` to `"local"` in `experiment_config.py`, configure `LOCAL_ENGINE_BASE_URL` in `.env`, and run:

```bash
python3 run_main.py
```

## 🗂️ Repository Structure

```text
.
├── experiment_config.py       # Experiment configuration: loads .env and builds WebSwarm/tool configs
├── run_main.py                # Experiment entry point; search/fetch engines are selected in experiment_config.py
├── webswarm/                  # Core WebSwarm agent implementation
├── tool_env/                  # Environments for the search, fetch_url, and submit_answer tools
├── task_manager/              # Benchmark loading and reward evaluation
├── llm_infer/                 # OpenAI-compatible and Claude-compatible LLM wrappers
├── depoly_bc_plus_local_corpus/  # Local BrowseComp-Plus corpus retrieval service (backend for local mode)
├── cache/                     # Serper/Jina caches generated on demand; ignored by .gitignore by default
└── result_debug/              # Default log output directory; ignored by .gitignore by default
```

## ⚙️ Core Workflow

`WebSwarmAgent` connects `TaskManager`, `ToolEnv`, and the root agent. A single task runs approximately as follows:

1. `TaskManager` loads the task and evaluator based on `benchmark`, `benchmark_version`, and `task_id`.
2. `RootNode`, acting as the root agent, reads the original task.
3. The root agent delegates self-contained subtasks through `solve_subtask(task, verb)`.
4. Verb Agents use the tool environment to search, read web pages, and organize their findings.
5. The root agent synthesizes the subtask results and submits the final answer through `submit_answer(answer)`.
6. `TaskManager` computes the reward, and the runner writes the trajectory, answer, reward, and tool statistics to the log.

### Agent Modules

`webswarm/` contains the core agent implementation:

- `webswarm_agent.py`: Top-level runner for a single task; connects `ToolEnv`, `TaskManager`, and the root agent.
- `root_node.py`: Root agent responsible for delegation through `solve_subtask(task, verb)` and submission through `submit_answer(answer)`.
- `verbs/atom_agent/`: Attribute lookup for known entities, short-chain multi-hop retrieval, and small-scale fact verification.
- `verbs/deep_agent/`: Deep retrieval for unknown target entities using constraint intersection and a propose-then-verify strategy.
- `verbs/wide_agent/`: Fan-out data collection that can delegate homogeneous subtasks concurrently or expand them recursively.
- `verbs/entity_collect_agent/`: Entity-set enumeration using multi-strategy sampling and split-verify-merge validation.
- `guidance/`: Runtime guidance signals, including `web_probing` and `subtask_experience`.

Top-level WebSwarm runtime parameters are built and validated by `experiment_config.build_webswarm_config`, then passed to `WebSwarmAgent` as a dictionary.

### Tool Environment

`tool_env/` always loads the following tools:

| Tool | Input | Description |
| --- | --- | --- |
| `search` | `query`; optional `date_range` | Searches a single query. Uses Serper by default; in local mode, calls `{LOCAL_ENGINE_BASE_URL}/search`. |
| `fetch_url` | `url`, `goal` | Reads a single URL and uses WebSummaryTool to extract goal-relevant information from the page. Uses Jina by default; in local mode, calls `{LOCAL_ENGINE_BASE_URL}/document`. |
| `submit_answer` | `answer` | Submits the final answer and terminates the current task. Empty answers are rejected and must be retried. |

### LLM Abstraction

`llm_infer/` is the unified entry point for LLM calls. It currently supports `openai` and `claude`.

## 🔐 Environment Variables

The project loads environment variables from `.env` in the project root:

| Variable | Purpose |
| --- | --- |
| `LLM_BASE_URL` | LLM service URL for the main agent, Verb Agents, and page summarization. |
| `LLM_API_KEY` | LLM service API key for the main agent, Verb Agents, and page summarization. |
| `JUDGE_MODEL_BASE_URL` | Service URL for the benchmark LLM judge. |
| `JUDGE_MODEL_API_KEY` | Service API key for the benchmark LLM judge. |
| `SERPER_BASE_URL` | Service URL for the default Serper engine used by `search`. |
| `SERPER_API_KEY` | API key for the default Serper engine used by `search`. |
| `JINA_BASE_URL` | Service URL for the default Jina Reader engine used by `fetch_url`. |
| `JINA_API_KEY` | API key for the default Jina Reader engine used by `fetch_url`. |
| `LOCAL_ENGINE_BASE_URL` | Root URL of the local search/fetch service. Required only when `SEARCH_ENGINE="local"` in `experiment_config.py`. |

## 🛠️ Runtime Configuration

Experiment settings are defined at the top of `experiment_config.py`:

```python
# experiment_config.py
MODEL = "<model-name>"
PROVIDER = "openai"

JUDGE_MODEL_NAME = "<judge-model-name>"
JUDGE_MODEL_PROVIDER = "claude"

BENCHMARK = "browsecomp_plus"
BENCHMARK_VERSION = "bc_all"
TASK_IDS = ["bc_en_1"]   # None means all cases for the selected benchmark/version

SEARCH_ENGINE = "web"    # "web" (Serper/Jina) or "local" (self-hosted retrieval service)
ENABLE_CACHE = True      # Enable or disable search/fetch caching independently
```

Common settings:

- `MODEL` / `PROVIDER`: Main model name and provider, passed to all agents.
- `JUDGE_MODEL_NAME` / `JUDGE_MODEL_PROVIDER`: Model name and provider for the benchmark judge.
- `BENCHMARK` / `BENCHMARK_VERSION` / `TASK_IDS`: Dataset, version, and tasks to run.
- `webswarm_config.max_steps`: Shared step budget for the root and leaf agents.
- `webswarm_config.prompt_version`: Root prompt version.
- `webswarm_config.web_probing`: Whether to run Web Probing before the first-layer wide task begins and record the corresponding guidance event.
- `webswarm_config.subtask_experience`: Whether to run scout subtasks first, extract transferable experience, and inject it into subsequent atom subtasks.
- `SEARCH_ENGINE="web"`: Uses Serper for search and Jina Reader for page retrieval. `SERPER_BASE_URL`, `SERPER_API_KEY`, `JINA_BASE_URL`, and `JINA_API_KEY` must be configured.
- `SEARCH_ENGINE="local"`: Uses the `/search` and `/document` services under `LOCAL_ENGINE_BASE_URL`. Serper and Jina keys are not required.
- `enable_cache`: An independent hyperparameter controlled by `ENABLE_CACHE`; it is no longer derived from `SEARCH_ENGINE`. When enabled, cache files are written to `cache/serper_cache/` and `cache/jina_cache/`. In general, enable it for web mode and disable it for local mode.
- `max_workers`: Number of benchmark tasks the runner executes concurrently.
- `save_every_n`: Number of completed tasks between automatic saves.

Prompt versions are subject to explicit constraints:

- When `benchmark="gisa"`, `prompt_version` must be `"gisa"`.
- For all other benchmarks, `prompt_version` must be `"general"`.

## 🖥️ Deploying the BrowseComp-Plus Retrieval Service Locally

`depoly_bc_plus_local_corpus/` provides a local corpus retrieval service for BrowseComp-Plus. When `SEARCH_ENGINE="local"`, it replaces Serper and Jina by performing dense retrieval over a fixed corpus and exposing two HTTP endpoints: `POST /search` and `POST /document` (port 8080 by default). These correspond to the `search` and `fetch_url` tools in the tool environment. This setup isolates the effect of the retriever and enables fair comparisons between deep-research agents.

The repository contains only the service code and empty data directories; the index files must be downloaded separately. Once prepared, the main files should be organized as follows:

```text
depoly_bc_plus_local_corpus/
├── searcher/
│   ├── http_server.py         # Flask/waitress service exposing /search and /document
│   └── searchers/             # Registrations for bm25/faiss/reasonir/custom; the steps below use faiss
├── indexes/
│   └── qwen3-embedding-8b/    # Downloaded prebuilt vector-index shards
│       ├── corpus.shard1_of_4.pkl
│       ├── corpus.shard2_of_4.pkl
│       ├── corpus.shard3_of_4.pkl
│       └── corpus.shard4_of_4.pkl
└── data/                      # Optional local corpus directory
```

### Download the Index

Download the Qwen3-Embedding-8B index from Hugging Face and place it in the `indexes/qwen3-embedding-8b/` directory shown above:

- `depoly_bc_plus_local_corpus/indexes/qwen3-embedding-8b/` ← [Tevatron/browsecomp-plus-indexes · qwen3-embedding-8b](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes/tree/main/qwen3-embedding-8b)

By default, the service loads [Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus) through Hugging Face Datasets. The corpus is downloaded and cached on the first startup.

When running dense retrieval on an NVIDIA GPU with the current dependency configuration, install FlashAttention 2 after installing PyTorch if needed:

```bash
pip install flash-attn --no-build-isolation
```

### Start the Service

```bash
cd depoly_bc_plus_local_corpus
python3 searcher/http_server.py \
  --searcher-type faiss \
  --index-path 'indexes/qwen3-embedding-8b/corpus.*.pkl' \
  --model-name 'Qwen/Qwen3-Embedding-8B' \
  --dataset-name 'Tevatron/browsecomp-plus-corpus' \
  --pooling eos \
  --normalize \
  --get-document \
  --snippet-max-tokens -1 \
  --port 8080
```

`--get-document` also enables the `/document` endpoint. When snippet token truncation is enabled, the current `http_server.py` uses a hard-coded local tokenizer path, so the example above disables truncation with `--snippet-max-tokens -1`. The service uses waitress by default; its concurrency and connection limits can be adjusted with `--max-concurrency` and `--connection-limit`.

After the service starts, set `SEARCH_ENGINE = "local"` in `experiment_config.py` and add the following to `.env` in the project root:

```text
LOCAL_ENGINE_BASE_URL=http://localhost:8080
```

Then run `python3 run_main.py`. The `search` tool will call `{LOCAL_ENGINE_BASE_URL}/search`, and `fetch_url` will call `{LOCAL_ENGINE_BASE_URL}/document`; no Serper or Jina keys are required.

## 📊 Benchmarks

`TaskManager` currently supports the following benchmark/version combinations:

| Benchmark | Version | Primary Data File | Gold Answer |
| --- | --- | --- | --- |
| `browsecomp_plus` | `bc_all` | `task_manager/benchmark/browsecomp_plus/data/bc.jsonl` | `answer` field in the JSONL file |
| `browsecomp_plus` | `all` | `task_manager/benchmark/browsecomp_plus/data/bc_plus.jsonl` | `answer` field in the JSONL file |
| `browsecomp_plus` | `plus_subset` | `task_manager/benchmark/browsecomp_plus/data/bc_plus_subset.jsonl` | `answer` field in the JSONL file |
| `widesearch` | `all` | `task_manager/benchmark/widesearch/data/widesearch.json` | `task_manager/benchmark/widesearch/data/widesearch_gold/*.csv` |
| `widesearch` | `en_subset` | `task_manager/benchmark/widesearch/data/widesearch_en_subset.json` | `task_manager/benchmark/widesearch/data/widesearch_gold/*.csv` |
| `deepwidesearch` | `all` | `task_manager/benchmark/deepwidesearch/data/deepwidesearch.jsonl` | `task_manager/benchmark/deepwidesearch/data/gold_answer/*.csv` |
| `deepwidesearch` | `en_subset` | `task_manager/benchmark/deepwidesearch/data/deepwidesearch_en_subset.jsonl` | `task_manager/benchmark/deepwidesearch/data/gold_answer/*.csv` |
| `gisa` | `all` | `task_manager/benchmark/gisa/data/question.jsonl` | `task_manager/benchmark/gisa/data/answers/{id}.csv` |

Evaluation methods:

- `browsecomp_plus`: An LLM judge determines whether the final answer matches the gold answer.
- `widesearch`: The model's output table is evaluated against the gold CSV using table metrics. An LLM judge aligns columns and values when necessary.
- `deepwidesearch`: The current runner skips the original entity-accuracy gate by default and reuses the WideSearch table-evaluation logic.
- `gisa`: Uses local rule-based evaluation without an LLM judge. Different task types use the corresponding item, set, list, or table primary metric.

## 📝 Logging

Results are written to `result_debug/` by default. The top-level log structure is approximately:

```json
{
  "run_info": {
    "task_config": {},
    "task_ids": [],
    "agent_name": "webswarm",
    "webswarm_config": {},
    "env_config": {},
    "judge_model_name": "",
    "judge_model_provider": "",
    "max_workers": 0,
    "avg_reward_score": null,
    "tool_counts": {}
  },
  "results": {
    "<task_id>": {}
  }
}
```

For each task, `results` stores agent messages, the trajectory, subtask results, the final answer, reward information, and per-task tool statistics. Failed tasks include `error` and `traceback` fields for debugging.

## 🔄 Resuming a Run

At the bottom of `run_main.py`, set `resume_from` to an existing result file:

```python
resume_from = "result_debug/<existing-result>.json"
```

The current `resume_from` behavior is **rerun and merge**, not automatic skipping of completed tasks:

- Tasks in the current `TASK_IDS` that already exist in the old log are rerun, and their old results are overwritten.
- New tasks in the current run are appended to `results`.
- Tasks in the old log that are not selected by the current `TASK_IDS` are retained.
- `TASK_IDS = None` first expands to all tasks for the current benchmark/version, so every task is rerun.

After the run finishes, the program recalculates the success and failure counts, average reward, and tool statistics across all merged results.
