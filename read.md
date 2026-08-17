我要实现一个**通用 GraphRAG 推荐/排序研究框架**。请先完成可扩展的工程架构和 HotpotQA 版本，不要一次性实现所有数据集。

# 1. 研究目标

把不同任务统一抽象成：

Query + Candidate + Graph → Relevance Score → Candidate Ranking

这里“推荐”采用广义定义：

* HotpotQA：Query 是 question，Candidate 是 paragraph，目标是推荐/排序相关 evidence paragraphs。
* Amazon：Query 是 user preference/history，Candidate 是 product。
* Yelp：Query 是 user preference/history，Candidate 是 business。
* 后续还可能扩展 MIND、2WikiMultiHopQA、MuSiQue 等。

因此不要把代码写死为 QA 系统或商品推荐系统。

---

# 2. 核心 Pipeline

统一采用三级级联排序：

```text
Raw Candidate Pool
        ↓
GraphRAG Retrieval / Coarse Ranking
        ↓
Top-100
        ↓
Neural Ranker
        ↓
Top-20
        ↓
LLM Reranker
        ↓
Top-10
```

三个模块职责必须严格分离。

## Stage 1：GraphRAG

GraphRAG 负责高召回候选检索和粗排。

输入：

* query
* candidates
* graph

输出 Top-100，每个 candidate 至少保存：

* candidate_id
* text
* semantic_score
* graph_score
* rag_score
* evidence_nodes
* evidence_edges
* paths
* rank

GraphRAG 不能只返回无序候选。

GraphRAG score 可以先采用简单可解释版本：

rag_score =
alpha * semantic_score

* beta * graph_score

GraphRAG 模块需要从一开始设计成可扩展框架，不要只实现临时 baseline。当前版本实现一个完整、清晰、可替换的 GraphRAG pipeline，包括：

graph construction
entity/node retrieval
multi-hop graph expansion
candidate scoring
evidence path extraction
candidate ranking output

当前实现的算法可以采用基础方法，但代码接口必须支持后续替换更复杂的方法。

后续我会在 GraphRAG retrieval、graph reasoning、candidate scoring 等模块继续进行算法创新，因此不要把 GraphRAG 写成不可修改的固定逻辑。

---

# 3. GraphRAG 基础实现

暂时不要完整复刻 Microsoft GraphRAG。

实现面向 ranking/recommendation 的轻量 GraphRAG：

```text
Query
 ↓
Query Embedding
 ↓
Semantic Seed Retrieval
 ↓
Relevant Graph Nodes
 ↓
1~N Hop Graph Expansion
 ↓
Evidence Paths
 ↓
Candidate Graph Features
 ↓
Graph + Semantic Scoring
 ↓
Top-100
```

建议：

* PyTorch
* NetworkX：第一版图结构
* FAISS：向量检索
* SentenceTransformers/BGE/E5：embedding
* 后续允许替换 Neo4j 等

所有组件必须接口化，不能和具体数据集绑定。

---

# 4. Stage 2：Neural Ranker

输入 GraphRAG Top-100。

Neural Ranker 不只是重新计算文本 cosine。

每个 candidate 至少允许使用：

```text
query embedding
candidate embedding
semantic score
graph score
GraphRAG score
graph features
evidence/path representation
```

预留 Candidate-Candidate Interaction 模块，因为后续可能研究：

“候选集合内部关系是否有助于 ranking”。

Neural Ranker 输出：

* candidate_id
* neural_score
* neural_rank
* 保留 GraphRAG 原始信息

然后截取 Top-20。
Neural Ranker 从开始就需要设计成完整模块，而不是临时实现。

要求：

独立模型接口
独立训练流程
独立 evaluation
支持后续替换不同模型结构

当前可以实现一个基础可训练版本作为 baseline，例如：

MLP Ranker
Transformer-based Ranker
Graph-aware Ranker

但代码结构必须支持后续加入：

GNN
Candidate Interaction Network
Cross Attention
Graph Neural Ranking Model

Neural Ranker 输入统一包含：

query representation
candidate representation
GraphRAG score
graph evidence features
path features

输出：

ranking score
candidate ranking
intermediate representation

---

# 5. Stage 3：LLM Reranker

LLM 只处理 Neural Ranker 的 Top-20。

输入包含：

```text
Query

Candidate 1:
text
graph evidence
rag score
neural score

Candidate 2:
...
```

LLM 的主要任务是根据 query、candidate 和 evidence 对 Top-20 做最终 reranking。

输出 Top-10。

要求：

* LLM provider 独立封装。
* 可以替换 Qwen / Gemini / DeepSeek / OpenAI 等 API。
* 使用结构化 JSON 输出。
* API response 必须缓存。
* 相同 query + candidates + model + prompt version 不允许重复调用 API。
* LLM 模块和 Neural Ranker 完全解耦。

---

# 6. 统一数据接口

设计 Dataset Adapter。

所有数据集转换为统一结构，例如：

```python
@dataclass
class Query:
    query_id: str
    text: str
    user_id: str | None = None

@dataclass
class Candidate:
    candidate_id: str
    text: str
    metadata: dict

@dataclass
class RecommendationInstance:
    query: Query
    candidates: list[Candidate]
    positive_ids: list[str]
```

Graph 也统一：

```python
@dataclass
class Node:
    node_id: str
    node_type: str
    text: str
    metadata: dict

@dataclass
class Edge:
    source: str
    relation: str
    target: str
    metadata: dict
```

后续任何数据集只增加 Adapter，不修改核心 Pipeline。

---

# 7. 第一阶段只实现 HotpotQA

当前先不要实现 Amazon/Yelp。

先用 HotpotQA 把整个系统跑通。

定义：

```text
Query = HotpotQA question
Candidate = paragraph
Positive = supporting evidence paragraph
```

第一阶段需要实现：

```text
HotpotQA
 ↓
Unified Dataset Adapter
 ↓
Graph Construction
 ↓
GraphRAG
 ↓
Top-100
 ↓
Neural Ranker
 ↓
Top-20
 ↓
LLM Reranker
 ↓
Top-10
 ↓
Evaluation
```

如果某个实例原始 candidate 数量少于阶段要求，不要人为重复候选：

```python
k = min(configured_k, len(candidates))
```

---

# 8. Evaluation

Evaluation 模块第一版统一实现以下指标：

Recall@5
Recall@10
NDCG@5
NDCG@10
MRR
Hit@K


Evaluation 模块需要支持不同阶段结果评估：

GraphRAG:

Top-K candidate retrieval evaluation

Neural Ranker:

intermediate ranking evaluation

LLM:

final ranking evaluation

需要支持后续直接比较：

```text
Dense Retrieval
GraphRAG
GraphRAG + Neural
GraphRAG + LLM
GraphRAG + Neural + LLM
```

---

# 9. 所有中间结果必须缓存

目录类似：

```text
outputs/
  hotpotqa/
    experiment_x/
      graphrag.jsonl
      neural.jsonl
      llm.jsonl
      metrics.json
      config.yaml
```

GraphRAG 结果计算一次后，可以独立训练 Neural Ranker。

Neural Top-20 保存后，可以独立运行不同 LLM。

不要让修改 LLM 后重新运行 GraphRAG。

每条结果保留 query_id 和 candidate_id，保证三个阶段可以追踪同一个 candidate。

---

# 10. 配置化

所有关键参数放 YAML，不要硬编码：

```yaml
retrieval:
  top_k: 100
  hops: 2
  semantic_weight: 0.5
  graph_weight: 0.5

ranker:
  top_k: 20
  model: mlp
  hidden_dim: 256

llm:
  top_k: 10
  model: null
  prompt_version: v1

evaluation:
  ks: [5, 10, 20, 100]
```

后续我要进行：

* GraphRAG Top-K sensitivity
* hop sensitivity
* Neural Top-K sensitivity
* LLM Top-K sensitivity
* 不同 Neural Ranker
* 不同 LLM
* 消融实验

所以配置必须方便批量实验。

---

# 11. 工程结构

建议：

```text
src/
  data/
    base.py
    hotpotqa.py

  graph/
    builder.py
    store.py
    retriever.py

  graphrag/
    retriever.py
    scorer.py
    evidence.py

  ranker/
    base.py
    mlp.py
    interaction.py
    loss.py

  llm/
    base.py
    client.py
    prompt.py
    reranker.py
    cache.py

  evaluation/
    metrics.py
    evaluator.py

  pipeline/
    recommender.py

configs/
scripts/
outputs/
tests/
```

请保持模块低耦合。

---

# 12. 后续扩展

HotpotQA 跑通以后，我会继续增加：

```text
2WikiMultiHopQA
MuSiQue
Amazon
Yelp
MIND
```

统一接口：

```text
Query + Candidate + Graph
          ↓
GraphRAG Top-100
          ↓
Neural Ranker Top-20
          ↓
LLM Top-10
```

所以现在的设计必须保证增加新数据集时：

**只增加 Dataset Adapter 和对应 Graph Builder，不修改 GraphRAG / Neural / LLM / Evaluation 的核心接口。**

---



---

# 14. 实现顺序

不要一次生成整个项目。按照以下顺序实施，每一步先检查现有代码再修改：

1. 检查当前 repository 和已有 HotpotQA 代码。
2. 给出当前代码结构分析。
3. 设计 Unified Dataset Interface。
4. 将已有 HotpotQA 接入该接口，尽量复用已有代码。
5. 实现/整理 Graph 数据接口。
6. 实现 GraphRAG Top-100，并保存结果。
7. 实现最简单 Neural Ranker baseline，Top-100 → Top-20。
8. 实现统一 evaluation。
9. 确认前两阶段能够独立运行和复现。
10. 最后再实现 LLM Reranker Top-20 → Top-10。
11. 增加测试、配置和实验日志。

每完成一个阶段，请告诉我：

* 修改了哪些文件
* 数据输入/输出格式
* 如何运行
* 当前指标
* 下一阶段准备做什么



重要 
data和rag是旧代码的一些结果 能用则用 不好用就不用

开发环境：

本地电脑使用 Codex 编写代码
远程服务器负责运行训练和实验

因此：

不要假设本地环境拥有 GPU。
不要绑定本地路径。
所有路径使用配置文件或相对路径。
所有实验必须支持命令行运行。
训练、推理、评估脚本必须可以在远程 Linux 环境直接执行。

需要提供：

python train.py --config xxx.yaml


python evaluate.py --config xxx.yaml


python run_pipeline.py --config xxx.yaml

等标准运行方式。

所有依赖写入：

requirements.txt

或：

environment.yml

不要依赖 IDE 配置。
