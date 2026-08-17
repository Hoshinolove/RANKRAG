# RankRAG：GraphRAG + Neural + LLM 级联排序

当前 HotpotQA 正式流程：

```text
全局 ParagraphCorpus
  -> 离线 embedding + FAISS
  -> 全局 paragraph/entity/relation KG
  -> Hybrid Candidate Pool
  -> GraphRAG Top100
  -> Candidate Set Transformer Top20
  -> LLM Top10
```

## 当前数据

| 文件 | 规模 | 用途 |
| --- | ---: | --- |
| `data/hotpot_train.json` | 90,447 个 query | Neural train/validation 来源 |
| `data/hotpot_train_paragraphs.json` | 483,696 条段落记录 | 全局候选 corpus 输入 |
| `outputs/kg_train_gliner_rebel/kg_extractions.jsonl` | 应为 483,696 行 | GLiNER + REBEL 完整训练段落 KG |
| `data/hotpot_dev_subset.json` | 7,405 个 query | 旧开发子集和本地基线 |
| `data/kg_extractions.jsonl` | 66,635 行 | 旧开发段落 KG，可保留但正式训练配置不使用 |

`ParagraphCorpus` 使用 `SHA1(title + "\n" + text)` 作为稳定 `paragraph_id` 并按 ID 去重。HotpotQA 每题自带的 context 只提供 ground truth/evaluation 信息，不再是 GraphRAG 候选全集。

## 全局候选检索

正式配置 `configs/hotpotqa_train_fullkg.yaml` 使用：

```text
query -> FAISS semantic Top500
      -> seed paragraph/entity
      -> 最多 2-hop graph expansion
      -> 合并和去重
      -> GraphRAG scorer
      -> Top100
```

positive IDs 和 supporting facts 不会强制注入候选池。候选生成器不负责最终 Top100 排名；最终排名仍由 GraphRAG scorer 根据 `semantic_score`、`graph_score`、`rag_score` 和 evidence paths 完成。

所有全局资产只离线构建一次：

```bash
python prepare_global_retrieval.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage all
```

详细中文命令见 [RUN_GLOBAL_GRAPHRAG_CN.md](E:/11rankRAG/code/RUN_GLOBAL_GRAPHRAG_CN.md)。

## Neural Ranker

正式 Neural 模型是 Candidate Set Transformer：

```text
[B,K,D] -> projection -> TransformerEncoder -> scores -> listwise loss
```

GraphRAG JSONL 先一次性转换为 Tensor shards。训练期间不做文本 embedding、图遍历或路径搜索。多 worker 使用同一个 epoch 的统一 shard permutation，再按 worker ID 划分 shard。

validation 中 GraphRAG Top-K 没有 positive 的 query 会进入指标分母并贡献 0。Neural inference 默认且标准输出只处理 validation；显式 `--split train` 时写入 `neural.train.jsonl`，不会覆盖或混入 `neural.jsonl`。

详细中文命令见 [RUN_TRAIN_NEURAL_CN.md](E:/11rankRAG/code/RUN_TRAIN_NEURAL_CN.md)。

## 最短运行命令

```bash
# 安装
python -m pip install -r requirements.txt

# 离线全局资产
python prepare_global_retrieval.py --config configs/hotpotqa_train_fullkg.yaml --stage all

# 全局 GraphRAG Top100；中断后重复本命令可从已完成 shard 恢复
python run_pipeline.py --config configs/hotpotqa_train_fullkg.yaml --stage graphrag

# 一次性 Tensor preprocessing
python prepare_ranker_dataset.py --config configs/hotpotqa_train_fullkg.yaml

# Tensor Neural 训练
python train.py --config configs/hotpotqa_train_fullkg.yaml

# 只推理 validation
python run_pipeline.py --config configs/hotpotqa_train_fullkg.yaml --stage neural --split validation --force

# 评估 validation
python evaluate.py --config configs/hotpotqa_train_fullkg.yaml --stage neural

# LLM Top20 -> Top10
python run_pipeline.py --config configs/hotpotqa_train_fullkg.yaml --stage llm --force
```

## 输出

```text
outputs/hotpotqa/hotpot_train_global_graphrag/
├── graphrag.jsonl
├── graphrag_retrieval_stats.json
├── graphrag_shards/         # 原子分片和恢复进度
├── ranker_dataset/
├── ranker_checkpoints/
├── neural.jsonl
├── neural.train.jsonl       # 仅显式训练集诊断时生成
├── llm.jsonl
└── metrics.json
```

`graphrag_retrieval_stats.json` 包含平均候选池大小、GraphRAG 前 gold recall、Recall@100、平均图扩展候选数，以及 query embedding、semantic search、graph expansion、evidence serialization 和总 query 耗时。

## 测试

```bash
python -m pytest -q
```

本项目代码位于 `src/rankrag/`。`LightRAG-main/`、`graphrag-main/` 和 `hotpot-master/` 是上游参考仓库，不属于本次实现，也不应当作为清理对象删除。
