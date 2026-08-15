# HotpotQA 全局 GraphRAG 运行手册

本文档面向第一次运行项目的用户。所有命令都在远程 Linux 服务器的项目根目录执行。

## 1. 安装依赖

```bash
cd /data/11rankRAG/code
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

服务器若需要特定 CUDA 版 PyTorch，请先按服务器 CUDA 版本安装 PyTorch，再执行最后一条命令。

## 2. 检查输入

```bash
test -f data/hotpot_train.json && echo "训练问题：存在" || echo "训练问题：缺失"
test -f data/hotpot_train_paragraphs.json && echo "全局段落：存在" || echo "全局段落：缺失"
test -f outputs/kg_train_gliner_rebel/kg_extractions.jsonl && echo "完整 KG：存在" || echo "完整 KG：缺失"
wc -l outputs/kg_train_gliner_rebel/kg_extractions.jsonl
```

完整 GLiNER + REBEL 抽取文件应为 `483696` 行。已有 66,635 行旧 KG 不再参与训练集全局配置，可以保留但无需再次处理。全局 `ParagraphCorpus` 会按稳定段落 ID 去重，所以 corpus 数量可能小于 483,696，这是正常现象。

## 3. 一次性构建离线资产

```bash
python prepare_global_retrieval.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage all
```

这条命令依次构建去重 ParagraphCorpus、全部 paragraph embedding、FAISS index 和全局 KG index。输出：

```text
outputs/hotpotqa_global_assets/train/
├── corpus.jsonl
├── paragraph_embeddings.npy
├── paragraphs.faiss
├── global_graph.sqlite
└── manifest.json
```

`global_graph.sqlite` 明确包含 `paragraph_to_entities`、`entity_to_paragraphs` 和 `entity_relation_adjacency` 三张表。query retrieval 只读缓存，不会重新编码 corpus 或重建 KG。

后台运行：

```bash
mkdir -p logs
nohup python prepare_global_retrieval.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage all \
  > logs/prepare_global_retrieval.log 2>&1 &
echo $!
tail -f logs/prepare_global_retrieval.log
```

只有需要覆盖全部资产时才加 `--force`：

```bash
python prepare_global_retrieval.py --config configs/hotpotqa_train_fullkg.yaml --stage all --force
```

## 4. 检查资产

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/hotpotqa_global_assets/train/manifest.json")
m = json.loads(path.read_text(encoding="utf-8"))
print("corpus 段落数 =", m["corpus"]["paragraph_count"])
print("去重数量 =", m["corpus"]["duplicates_removed"])
print("embedding 维度 =", m["semantic_index"]["embedding_dim"])
print("FAISS 向量数 =", m["semantic_index"]["faiss_ntotal"])
print("KG 映射记录数 =", m["global_graph"]["mapped_kg_records"])
print("段落-实体边数 =", m["global_graph"]["paragraph_entity_edges"])
print("实体关系边数 =", m["global_graph"]["entity_relation_edges"])
PY
```

FAISS 向量数必须等于 corpus 段落数。`mapped_kg_records` 若明显偏小，应检查 KG 是否由当前训练段落文件生成。

## 5. 完整运行全局 GraphRAG

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage graphrag \
  --force
```

每个 query 执行：

```text
query embedding
  -> FAISS semantic Top500
  -> 高相关 seed paragraph/entity
  -> 最多 2-hop KG expansion
  -> 合并并去重候选池
  -> GraphRAG semantic/graph/rag score
  -> 真正 Top100
```

`supporting_facts` 和 positive IDs 不参与候选生成，也不会被强制加入候选池；它们只用于 label 和 evaluation。

输出与检查命令：

```bash
wc -l outputs/hotpotqa/hotpot_train_global_graphrag/graphrag.jsonl
python -m json.tool outputs/hotpotqa/hotpot_train_global_graphrag/graphrag_retrieval_stats.json
```

完整训练问题应输出 90,447 行。统计包含平均 candidate pool 大小、候选池 gold recall、Recall@100、平均 graph expansion candidate 数和检索时间。

## 6. 先运行 10 条 smoke

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage graphrag \
  --limit 10 \
  --force
```

该命令会覆盖正式 `graphrag.jsonl`，之后完整运行时必须再次使用 `--force`。

## 7. 参数说明

```yaml
global_retrieval:
  semantic_top_k: 500
  seed_paragraph_k: 20
  graph_hops: 2
  max_graph_candidates: 1000

retrieval:
  top_k: 100
```

`HybridCandidateGenerator` 只构造候选池，`retrieval.top_k` 才控制 GraphRAG 最终输出数量。代码强制 `graph_hops <= 2`。

## 8. 测试

```bash
python -m pytest -q
```

测试覆盖 Neural 多 worker shard 分配、no-positive validation 指标、validation-only inference，以及全局 corpus/KG/混合检索契约。
