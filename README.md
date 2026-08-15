# RankRAG 通用 GraphRAG 排序框架

RankRAG 是一个面向推荐与排序研究的通用三级级联框架：

```text
Query + Candidate + Graph -> GraphRAG -> Neural Ranker -> LLM Reranker
```

当前第一阶段只接入 HotpotQA。目录中的 `hotpot-master/` 和 `graphrag-main/` 是上游参考仓库，本项目没有修改它们；本项目代码位于 `src/rankrag/`。

## 当前数据

| 文件 | 规模 | 用途 |
| --- | ---: | --- |
| `data/hotpot_dev_subset.json` | 7,405 条问题 | 当前默认实验的开发子集；本文件中的样本均为 hard，每题最多 10 个候选段落 |
| `data/hotpot_train.json` | 90,447 条问题 | Neural Ranker 训练数据，可通过流式 Adapter 读取 |
| `data/hotpot_paragraphs.json` | 66,635 个段落 | 当前 KG 抽取和 GraphRAG 基线覆盖的段落库 |
| `data/hotpot_train_paragraphs.json` | 483,696 条记录（481,959 个唯一标题） | 训练段落库；与 KG 标题存在部分重叠，但没有完整 KG 抽取 |
| `data/kg_extractions.jsonl` | 66,635 行（约 63,445 个首实体标题） | 已有段落的实体和关系抽取，按需加载到局部图 |
| `rag_storage/graph_chunk_entity_relation.graphml` | 约 177 MB | 已有 LightRAG 图，可作为后续图存储来源 |
| `rag_storage/vdb_*.json` | GB 级 | 已有 LightRAG 实体、关系、段落向量存储，当前基线不整体加载 |

因此，当前 `7405` 指的是开发子集问题数，不是全部 HotpotQA 数据。训练段落唯一标题与现有 KG 的标题交集约为 33,009 个，约覆盖 6.85%，不能视为训练集已经完成 KG 抽取。训练集仍可以使用 `configs/hotpotqa_train.yaml` 完整运行 GraphRAG；已有 KG 的段落优先使用实体关系，没有抽取的训练段落使用确定性的文本词项图兜底。该兜底保证流程完整，但后续补齐训练集 KG 后图质量会更高。

## 训练集完整 KG 抽取

现有 66,635 条 KG 可以搁置。训练段落的完整 KG 使用 GLiNER 做实体识别、REBEL 做关系和三元组抽取。远程 GPU 机器安装额外依赖：

```bash
python -m pip install -r requirements-kg.txt
```

单机运行（从项目根目录执行）：

```bash
python scripts/extract_kg_gliner_rebel.py \
  --config configs/kg_train_gliner_rebel.yaml
```

推荐多 GPU 分片运行。下面以 8 个 worker 为例，每个 worker 使用一张 GPU，`--shard-index` 从 0 到 7：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 1
```

其余分片使用相同命令，仅修改 `CUDA_VISIBLE_DEVICES` 和 `--shard-index`。脚本支持断点续跑，默认会跳过已经写入的记录。分片完成后合并并校验：

```bash
python scripts/merge_kg_jsonl.py \
  --input-dir outputs/kg_train_gliner_rebel \
  --output outputs/kg_train_gliner_rebel/kg_extractions.jsonl \
  --expected-count 483696
```

合并成功后，使用新 KG 运行训练集 GraphRAG：

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage graphrag \
  --force
```

抽取结果每行包含 `id`、`title`、`entities` 和 `relationships`，字段兼容当前 GraphBuilder。抽取脚本会把段落标题强制作为第一个 document entity，确保标题索引稳定；GLiNER 未识别但被 REBEL 用于关系的主体和客体也会补入实体列表。

面向初学者的逐步复制命令见 [RUN_TRAIN_KG_CN.md](E:/11rankRAG/code/RUN_TRAIN_KG_CN.md)。

Neural Ranker 的训练、checkpoint、推理和评估命令见 [RUN_TRAIN_NEURAL_CN.md](E:/11rankRAG/code/RUN_TRAIN_NEURAL_CN.md)。

## 统一数据接口

`HotpotQAAdapter` 把原始问题转换为统一的 `RecommendationInstance`：

- `query`：问题 ID 和问题文本
- `candidates`：段落标题作为 `candidate_id`，段落正文作为 `text`
- `positive_ids`：HotpotQA supporting facts 对应的证据段落标题

GraphRAG、Neural 和 LLM 三个阶段都使用 JSONL 缓存。每行包含 `query_id`、`query_text`、`positive_ids`、`stage` 和有序的 `candidates`，候选 ID 可以在三个阶段之间追踪。

## 运行方式

默认使用确定性的 CPU 哈希向量，不需要 GPU、模型下载或联网 API。所有路径和关键参数由 YAML 控制。

```bash
# 运行完整三级流水线；已有缓存会自动复用
python run_pipeline.py --config configs/hotpotqa.yaml

# 只运行 GraphRAG 阶段
python run_pipeline.py --config configs/hotpotqa.yaml --stage graphrag

# 完整训练集 GraphRAG（90,447 条问题，只生成 GraphRAG 缓存）
python run_pipeline.py --config configs/hotpotqa_train.yaml --stage graphrag --force

# 训练 Neural Ranker。需要先存在 graphrag.jsonl
python train.py --config configs/hotpotqa.yaml

# 使用新的 Neural 配置重新生成 Top-20
python run_pipeline.py --config configs/hotpotqa.yaml --stage neural --force

# 只运行 LLM Top-20 -> Top-10
python run_pipeline.py --config configs/hotpotqa.yaml --stage llm --force

# 评估已有缓存
python evaluate.py --config configs/hotpotqa.yaml
```

本地小样本验证可以使用 `--limit 10 --force`。如果之前生成过截断缓存，之后运行完整数据时也必须加 `--force`。

输出目录为 `outputs/hotpotqa/<experiment>/`，包括：

```text
graphrag.jsonl       GraphRAG Top-100（候选不足时取实际数量）
neural.jsonl         Neural Ranker Top-20
llm.jsonl            LLM Reranker Top-10
metrics.json         Recall、NDCG、MRR、Hit@K
config.yaml          实验配置快照
llm_cache/           LLM response 缓存
ranker.pt            可选的 Neural Ranker checkpoint
```

如果要使用训练好的模型，把 `ranker.checkpoint` 设置为 `train.py` 输出的 checkpoint 路径，并使用新的 `output.experiment` 保存实验结果。要接入 Qwen、DeepSeek 或 OpenAI-compatible 服务，把 `llm.provider` 改为 `openai_compatible`，配置 `base_url`、`model`，并导出配置中指定的 API key 环境变量。程序不会读取 `data/api_keys.txt`。

## 当前开发子集基线

使用 7,405 条开发子集问题、hashing embedding、未训练 MLP 和离线 passthrough LLM 跑通后的指标为：

```text
Recall@5   0.6496
Recall@10  1.0000
NDCG@5     0.5478
NDCG@10    0.6859
MRR        0.6460
Hit@5      0.8929
```

当前 passthrough provider 只保留 Neural 顺序，用于验证接口和缓存，不代表真实 LLM 的重排效果。

## 扩展边界

增加新数据集时，只需要实现 `DatasetAdapter` 和对应的 `GraphBuilder`。检索、评分、Neural 模型、LLM provider、评估和缓存只依赖通用模型。`GraphStore`、`TextEmbedder`、`RankerModel`、`CandidateInteraction` 和 `LLMProvider` 都是后续替换 Neo4j、FAISS、SentenceTransformer、GNN、候选交互网络或其他 LLM API 的接口。

运行测试：

```bash
python -m pytest
```
