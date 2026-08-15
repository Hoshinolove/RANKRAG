# Neural Ranker 训练与推理操作手册

这份手册假设你已经完成训练集完整 KG 和 GraphRAG。它只负责 Neural Ranker，不需要你修改 Python 代码。

当前 Neural Ranker 是一个可训练的 PyTorch MLP baseline，训练流程是：

```text
GraphRAG graphrag.jsonl
        -> 读取 query、candidate、语义分、图分、路径特征
        -> MLP + pointwise BCE loss
        -> 保存 ranker.pt
        -> Neural Ranker 重新排序
        -> 输出 neural.jsonl
        -> 评估 Recall / NDCG / MRR / Hit@K
```

当前版本已经具备独立模型接口、独立训练入口、checkpoint 保存/加载和独立评估。它是研究 baseline，不是最终的 Transformer/GNN 模型：当前没有验证集 early stopping，默认使用 pointwise BCE，并在训练集缓存上训练和评估。后续可以在不改变流水线接口的情况下替换模型和 loss。

## 0. 进入项目目录

```bash
# 替换成你的真实项目路径
cd /data/11rankRAG/code

# 如果使用了虚拟环境，先激活
source .venv/bin/activate
```

## 1. 检查 GraphRAG 是否完成

Neural Ranker 不能直接读取原始段落，必须先读取 GraphRAG 缓存：

```bash
# 检查 GraphRAG 输出文件
test -f outputs/hotpotqa/hotpot_train_fullkg_graphrag/graphrag.jsonl \
  && echo "GraphRAG file: OK" \
  || echo "GraphRAG file: MISSING"

# 检查问题数量，完整训练集预期为 90,447 行
wc -l outputs/hotpotqa/hotpot_train_fullkg_graphrag/graphrag.jsonl
```

如果文件不存在，先回到 [RUN_TRAIN_KG_CN.md](E:/11rankRAG/code/RUN_TRAIN_KG_CN.md)，完成 KG 合并和 GraphRAG。

## 2. 确认配置

本手册使用：

```text
configs/hotpotqa_train_fullkg.yaml
```

默认训练参数是：

```yaml
ranker:
  model: mlp
  hidden_dim: 256
  top_k: 20

training:
  epochs: 3
  learning_rate: 0.001
  seed: 13
```

第一次运行不需要修改配置。

## 3. 训练 Neural Ranker

```bash
# 使用 GraphRAG 缓存训练 MLP Ranker
python train.py \
  --config configs/hotpotqa_train_fullkg.yaml
```

训练完成后应该看到类似输出：

```text
{
  "checkpoint": "outputs/hotpotqa/hotpot_train_fullkg_graphrag/ranker.pt",
  "mean_training_loss": 0.8,
  "updates": 271341.0,
  "epochs": 3.0
}
```

实际 loss 和 updates 会因数据、GPU、配置而不同。

checkpoint 文件是：

```text
outputs/hotpotqa/hotpot_train_fullkg_graphrag/ranker.pt
```

## 4. 使用训练好的模型生成 Neural 结果

不需要手工把 checkpoint 路径写入 YAML。程序会自动发现当前实验目录下的 `ranker.pt`：

```bash
# 读取 graphrag.jsonl，加载刚才训练的 ranker.pt，输出 neural.jsonl
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage neural \
  --force
```

输出文件：

```text
outputs/hotpotqa/hotpot_train_fullkg_graphrag/neural.jsonl
```

每个 candidate 会保留：

- `candidate_id`
- 原始 `rag_score`
- `semantic_score`
- `graph_score`
- `neural_score`
- `neural_rank`
- 图证据和路径
- 中间表示 `intermediate_representation`

## 5. 评估 Neural Ranker

```bash
# 只评估 Neural 阶段
python evaluate.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage neural
```

也可以同时查看 GraphRAG 和 Neural：

```bash
python evaluate.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage all
```

指标文件：

```text
outputs/hotpotqa/hotpot_train_fullkg_graphrag/metrics.json
```

## 6. 继续运行 LLM Reranker（可选）

Neural 结果完成后，才可以运行 LLM 阶段：

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage llm \
  --force
```

默认配置使用 offline passthrough provider，不会调用真实 API。如果要使用真实 LLM，需要在 YAML 中配置 `openai_compatible` provider 和 API key 环境变量。

## 7. 修改训练参数

编辑配置：

```bash
nano configs/hotpotqa_train_fullkg.yaml
```

例如训练 5 轮：

```yaml
training:
  epochs: 5
  learning_rate: 0.001
  seed: 13
```

修改后重新训练：

```bash
python train.py --config configs/hotpotqa_train_fullkg.yaml
python run_pipeline.py --config configs/hotpotqa_train_fullkg.yaml --stage neural --force
python evaluate.py --config configs/hotpotqa_train_fullkg.yaml --stage neural
```

## 8. 常见错误

### 找不到 GraphRAG 文件

错误类似：

```text
Missing cached GraphRAG results
```

说明还没有完成 GraphRAG，先执行：

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage graphrag \
  --force
```

### Neural 结果没有变化

确认你在训练后使用了 `--force`：

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage neural \
  --force
```

不加 `--force` 时，如果 `neural.jsonl` 已存在，程序会直接复用旧文件。

### 显存不足

当前默认 Neural Ranker 使用 CPU：

```yaml
ranker:
  device: cpu
```

如果远程机器需要 GPU，可改为：

```yaml
ranker:
  device: cuda
```

但当前 MLP 特征维度不大，CPU 也可以运行；训练数据很多时主要瓶颈是 JSONL 读取和文本 embedding。

### 为什么 Top-20 结果少于 20 条

HotpotQA 每个问题原始候选通常只有 2 到 10 个。代码会使用：

```python
k = min(configured_k, len(candidates))
```

因此不会复制候选，结果少于 20 是正确行为。

## 9. 完整命令顺序

如果 KG 和 GraphRAG 已经完成，直接按下面三条执行：

```bash
# 训练
python train.py --config configs/hotpotqa_train_fullkg.yaml

# 推理并生成 Top-20
python run_pipeline.py --config configs/hotpotqa_train_fullkg.yaml --stage neural --force

# 评估
python evaluate.py --config configs/hotpotqa_train_fullkg.yaml --stage neural
```
