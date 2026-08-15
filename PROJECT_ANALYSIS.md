# 当前项目结构与数据分析

## 现有目录

### 可复用数据

- `data/hotpot_dev_subset.json`：7,405 条 HotpotQA 开发子集问题。本文件中的样本均为 `hard`；每个问题最多有 10 个段落候选，平均约 9.95 个。
- `data/hotpot_train.json`：90,447 条训练问题。文件较大，项目 Adapter 采用顶层 JSON 数组流式解析，不会一次性读入内存。
- `data/hotpot_paragraphs.json`：66,635 个段落。它与 `data/kg_extractions.jsonl` 的记录数一致，是目前最完整的“段落 + KG 抽取”数据。
- `data/hotpot_train_paragraphs.json`：483,696 条训练段落记录，去重后 481,959 个标题。它可以用于后续训练或候选池实验，但当前目录没有与之对应的完整 KG 抽取结果。
- `data/kg_extractions.jsonl`：66,635 行，按首实体名去重后约 63,445 个标题，每行包含段落实体和关系。与训练段落标题的交集为 33,009 个，约覆盖训练段落唯一标题的 6.85%。运行 GraphRAG 时先收集当前候选标题，再流式扫描该文件，只把相关记录建立到局部图中。

### 已有上游代码和存储

- `hotpot-master/`：2019 年 HotpotQA 问答模型代码，输出假设是 QA 答案，不是通用段落排序结果，因此没有直接嵌入 RankRAG 核心流程。
- `graphrag-main/`：Microsoft GraphRAG 完整上游仓库。它适合参考和后续替换底层能力，本项目没有对其进行大规模重构。
- `rag_storage/graph_chunk_entity_relation.graphml`：已有 LightRAG NetworkX 图，约 177 MB。第一版按题目建立局部图，避免每个 query 都加载整张图。
- `rag_storage/vdb_entities.json`、`vdb_relationships.json`、`vdb_chunks.json`：已有压缩向量存储，体积达到 GB 级。第一版使用离线 hashing embedder，避免把这些 JSON 向量整体载入本地内存；后续可封装为 FAISS 或其他 VectorStore。

### 原始敏感文件

`data/api_keys.txt` 存在于数据目录，但 RankRAG 不读取它。真实 LLM provider 只从配置指定的环境变量读取 API key，避免把密钥耦合到数据处理流程。

## 新增代码结构

```text
src/rankrag/
  models.py             通用 Query/Candidate/Graph/Ranking 数据模型
  data/                 DatasetAdapter、HotpotQA Adapter、流式 JSON 解析
  graph/                GraphStore、NetworkX 实现、HotpotQA GraphBuilder
  graphrag/             语义检索、图扩展、证据路径、可解释评分
  ranker/               RankerModel、MLP、特征、训练、候选交互扩展点
  llm/                  Provider、Prompt、JSON 解析、缓存、Reranker
  evaluation/           Recall、NDCG、MRR、Hit@K
  pipeline/             GraphRAG -> Neural -> LLM 三级流水线
```

根目录的 `run_pipeline.py`、`train.py` 和 `evaluate.py` 是远程 Linux 环境可直接调用的标准入口。配置位于 `configs/hotpotqa.yaml`，不依赖 IDE 或本地绝对路径。

## 当前数据流

```text
hotpot_dev_subset.json (7,405 queries)
        |
        v
Unified HotpotQA Adapter
        |
        v
局部 NetworkX Graph + kg_extractions.jsonl
        |
        v
graphrag.jsonl  ->  neural.jsonl  ->  llm.jsonl
 Top-100             Top-20            Top-10
```

如果某题的候选数少于 100、20 或 10，代码使用实际候选数量，不会人为复制候选。

## 当前限制与后续方向

当前预处理 KG 没有覆盖完整训练段落库，标题级核对显示约 6.85% 的训练唯一标题与已有 KG 重叠。训练集完整运行使用 `configs/hotpotqa_train.yaml`：已有 KG 的段落使用实体关系，没有 KG 的训练段落使用确定性的文本词项图兜底，因此可以完整生成 GraphRAG 结果；补齐训练集 KG 后可替换该兜底以提升图质量。

当前默认 embedding 是可复现的 CPU hashing baseline；远程服务器可以切换到 SentenceTransformer/BGE/E5。当前 LLM 默认是 offline passthrough provider，真实 API provider 已实现但需要用户配置 endpoint 和环境变量。

## 训练集完整 KG 方案

训练集 KG 抽取脚本为 `scripts/extract_kg_gliner_rebel.py`，配置为 `configs/kg_train_gliner_rebel.yaml`。它使用 GLiNER 识别实体、REBEL 生成关系三元组，支持：

- 顶层 JSON 流式读取，不把 483,696 条段落一次性载入内存。
- `--num-shards` / `--shard-index` 分片并行。
- 每条 JSONL 记录即时 flush，异常退出后可 `--resume` 继续。
- `scripts/merge_kg_jsonl.py` 合并时检查重复 ID、字段完整性和总数量。

完整 KG 合并后，`configs/hotpotqa_train_fullkg.yaml` 将 `lexical_fallback` 关闭，只使用新生成的训练集 KG。
