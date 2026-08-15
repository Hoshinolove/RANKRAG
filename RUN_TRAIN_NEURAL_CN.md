# Candidate Set Transformer 训练与推理手册

本文档从已经生成全局 GraphRAG 结果开始。完整 GraphRAG 命令见 `RUN_GLOBAL_GRAPHRAG_CN.md`。

## 1. 检查 GraphRAG

```bash
cd /data/11rankRAG/code
source .venv/bin/activate
test -f outputs/hotpotqa/hotpot_train_global_graphrag/graphrag.jsonl \
  && echo "GraphRAG：存在" || echo "GraphRAG：缺失"
wc -l outputs/hotpotqa/hotpot_train_global_graphrag/graphrag.jsonl
```

完整训练问题结果应为 90,447 行。

## 2. 一次性生成 Tensor 分片

```bash
python prepare_ranker_dataset.py --config configs/hotpotqa_train_fullkg.yaml
```

输出结构：

```text
ranker_dataset/
├── manifest.json
├── train/
│   ├── shard-00000.pt
│   └── shard-00000.jsonl
└── validation/
    ├── shard-00000.pt
    └── shard-00000.jsonl
```

`.pt` 保存 `[query, candidate, feature]` Tensor、label 和 mask；`.jsonl` sidecar 保存 query ID 与 candidate ID。全局配置会按 paragraph ID 直接复用离线 corpus embedding，不会再次编码 Top100 段落。训练 100 epoch 时只读取 Tensor，不执行 embedding、图遍历或 JSON 特征构建。

检查 split：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/hotpotqa/hotpot_train_global_graphrag/ranker_dataset/manifest.json")
m = json.loads(path.read_text(encoding="utf-8"))
print("train =", m["splits"]["train"]["count"])
print("validation =", m["splits"]["validation"]["count"])
print("GraphRAG Top100 无 positive =", m["statistics"]["queries_without_positive_in_top_k"])
PY
```

`split_seed: 13` 表示用固定种子 13 根据 query ID 做可复现的 train/validation 划分。它不是第 13 条数据，也不是训练轮数。相同数据、query ID 和 seed 会得到相同 split。

## 3. 训练 Neural Ranker

```bash
python train.py --config configs/hotpotqa_train_fullkg.yaml
```

正式模型保持不变：

```text
[B,K,D]
 -> Linear projection
 -> 3 层 TransformerEncoder（8 heads）
 -> candidate scores
 -> listwise ranking loss
```

默认使用 100 epochs、batch size 256、CUDA BF16 和 8 个 DataLoader worker。所有 worker 在同一 epoch 使用统一 shard permutation，再按 `worker_id::num_workers` 分配；worker 内样本可以独立打乱。

后台训练：

```bash
mkdir -p logs
nohup python train.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  > logs/neural_train.log 2>&1 &
echo $!
tail -f logs/neural_train.log
```

输出：

```text
outputs/hotpotqa/hotpot_train_global_graphrag/ranker_checkpoints/best.pt
outputs/hotpotqa/hotpot_train_global_graphrag/ranker_checkpoints/last.pt
outputs/hotpotqa/hotpot_train_global_graphrag/ranker_checkpoints/training_log.jsonl
```

`best.pt` 按 validation NDCG@10 选择。GraphRAG Top-K 没有 positive 的 validation query 不会被跳过：Recall、NDCG、Hit、MRR 均贡献 0，并进入 query 总数。

## 4. 只对 validation 推理

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage neural \
  --split validation \
  --force
```

标准输出：

```text
outputs/hotpotqa/hotpot_train_global_graphrag/neural.jsonl
```

该文件只包含 validation query，绝不混入 train query。配置默认也是：

```yaml
ranker:
  inference_split: validation
```

只有明确做训练集诊断时才运行：

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage neural \
  --split train \
  --force
```

训练集结果写入 `neural.train.jsonl`，不会覆盖 validation 的 `neural.jsonl`。推理支持 manifest 中存在的任意 split 名称，因此以后增加独立 `test` 或 `dev` split 时无需修改 Neural 核心接口。

## 5. 评估 validation

```bash
python evaluate.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage neural
```

指标为 Recall@5、Recall@10、NDCG@5、NDCG@10、MRR、Hit@5、Hit@10，写入：

```text
outputs/hotpotqa/hotpot_train_global_graphrag/metrics.json
```

## 6. LLM Top20 到 Top10

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage llm \
  --force
```

LLM 默认读取 validation 的 `neural.jsonl`，GraphRAG、Neural 和 LLM 核心输出接口没有改变。

## 7. 显存不足

先把 batch size 改为 128 或 64：

```yaml
training:
  batch_size: 128
```

仍不足时再缩小模型：

```yaml
ranker:
  hidden_dim: 128
  num_heads: 4
  num_layers: 2
  feedforward_dim: 512
```

修改模型结构后必须重新训练，但无需重新运行 GraphRAG，也无需重新生成 Tensor。

## 8. 最短可复制流程

```bash
# 1. 全局段落、embedding、FAISS 和 KG 索引，只构建一次
python prepare_global_retrieval.py --config configs/hotpotqa_train_fullkg.yaml --stage all

# 2. 真正全局 GraphRAG Top100
python run_pipeline.py --config configs/hotpotqa_train_fullkg.yaml --stage graphrag --force

# 3. GraphRAG 转 Tensor，只执行一次
python prepare_ranker_dataset.py --config configs/hotpotqa_train_fullkg.yaml

# 4. 训练 Candidate Set Transformer
python train.py --config configs/hotpotqa_train_fullkg.yaml

# 5. 只推理 validation
python run_pipeline.py --config configs/hotpotqa_train_fullkg.yaml --stage neural --split validation --force

# 6. 评估 validation
python evaluate.py --config configs/hotpotqa_train_fullkg.yaml --stage neural
```
