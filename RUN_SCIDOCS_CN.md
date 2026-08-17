# SCIDOCS：full-corpus GraphRAG → Neural Ranker

本实现使用两套严格隔离的协议：

- `configs/scidocs_development.yaml`：original SciDocs `recomm/train.csv` 训练 Neural，`cite/val.qrel` 选择 `best.pt`。
- `configs/scidocs_beir_test.yaml`：BEIR-SCIDOCS 的 1,000 条 query 只做最终测试，只加载上一步的 `best.pt`。

BEIR qrels 只进入 `SCIDOCSAdapter.positive_ids` 和 evaluator，不进入 corpus、FAISS、seed 或 graph。每条 query 的
`allowed_candidate_ids` 都是 `None`，因此 Top100 是从完整 CandidateCorpus 检索，不是 positives 加采样 negatives。

## 1. 下载数据

BEIR 官方打包数据约 25,657 篇论文、1,000 条 test query：

```bash
mkdir -p data/scidocs/beir
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip \
  -O data/scidocs/beir/scidocs.zip
echo "38121350fc3a4d2f48850f6aff52e4a9  data/scidocs/beir/scidocs.zip" | md5sum -c -
unzip data/scidocs/beir/scidocs.zip -d data/scidocs/beir
```

开发集使用 original SciDocs。下面只同步当前协议需要的文件，不需要安装旧版 `pytrec_eval`：

```bash
python -m pip install awscli
mkdir -p data/scidocs/original
aws s3 sync --no-sign-request \
  s3://ai2-s2-research-public/specter/scidocs/ \
  data/scidocs/original/ \
  --exclude "*" \
  --include "paper_metadata_view_cite_read.json" \
  --include "paper_metadata_recomm.json" \
  --include "recomm/train.csv" \
  --include "cite/val.qrel"
```

应存在：

```text
data/scidocs/beir/scidocs/corpus.jsonl
data/scidocs/beir/scidocs/queries.jsonl
data/scidocs/beir/scidocs/qrels/test.tsv
data/scidocs/original/paper_metadata_view_cite_read.json
data/scidocs/original/paper_metadata_recomm.json
data/scidocs/original/recomm/train.csv
data/scidocs/original/cite/val.qrel
```

## 2. 图数据（可选，但正式图实验建议提供）

BEIR 文件本身只有论文文本、query 和 qrels，不包含完整 author/topic/citation graph。builder 会读取 original metadata
里实际存在的字段；还可以放一个固定、与 test qrels 无关的快照：

```text
data/scidocs/enrichment/papers.jsonl
```

每行格式：

```json
{"paper_id":"Semantic-Scholar-paper-id","authors":[{"authorId":"a1","name":"Alice"}],"fieldsOfStudy":["Computer Science"],"references":["another-paper-id"]}
```

支持字段：

- paper ID：`paper_id` / `_id` / `id`
- author：字符串，或含 `authorId` / `author_id` / `id` / `name` 的对象
- field/topic：`fields` / `fieldsOfStudy` / `topics`
- 论文出边：`references` / `outbound_citations`

不要把 `qrels/test.tsv` 或 `cite/val.qrel` 转成 enrichment。构图还会主动删除所有 validation/test query 的
outgoing citation edge；作者和领域属性仍可保留。若 enrichment 不存在，流程仍能运行，但图只有通用 candidate proxy，
结果基本等价于 semantic-first baseline，不能把它当成完整 citation-graph 实验。

seed 由通用 `weighted_query` provider 产生：semantic Top50 加 query paper 本身。query paper 只用于进入它的
author/topic/citation 节点，最终候选中仍会排除 query 自己；若该 paper ID 不在当前 corpus，通用 provider 会安全跳过。

## 3. 先做协议审计

先只构 corpus 和检查 ID；这一步不会构建 embedding/FAISS：

```bash
python prepare_scidocs.py \
  --config configs/scidocs_development.yaml \
  --stage corpus

python prepare_scidocs.py \
  --config configs/scidocs_beir_test.yaml \
  --stage corpus
```

重点检查：

```text
outputs/scidocs_global_assets/development/protocol_report.json
outputs/scidocs_global_assets/beir_test/protocol_report.json
```

脚本会在以下情况直接报错：BEIR corpus 不是 25,657、test query 不是 1,000、positive ID 无法精确加入
corpus、official test positives 不是 4,928、出现固定候选子集。ID 映射只接受精确 paper ID，不做标题模糊匹配。

original SciDocs development metadata 已知可能缺少极少数被 train/validation 标注的论文。development 配置会只删除这些
无法检索的 judgment，并在报告中写出 `dropped_missing_positive_judgment_count` 和具体 ID；如果某条 query 的所有
positive 都缺失，该 query 也不会进入训练或 validation。BEIR official test 仍使用 `strict`，任何缺失都会报错，不能静默修改
官方 test 标签。

## 4. Development：训练并选择 best.pt

```bash
python prepare_scidocs.py \
  --config configs/scidocs_development.yaml \
  --stage graph

python prepare_scidocs.py \
  --config configs/scidocs_development.yaml \
  --stage embeddings

python run_pipeline.py \
  --config configs/scidocs_development.yaml \
  --stage graphrag

python prepare_ranker_dataset.py \
  --config configs/scidocs_development.yaml

python train.py \
  --config configs/scidocs_development.yaml

python run_pipeline.py \
  --config configs/scidocs_development.yaml \
  --stage neural \
  --split validation

python evaluate.py \
  --config configs/scidocs_development.yaml
```

checkpoint：

```text
outputs/scidocs/scidocs_development/ranker_checkpoints/best.pt
```

`recomm/train.csv` 中官方给出的 clicked paper 是 train positive；原文件中的固定 other candidates 被忽略，GraphRAG
仍检索整个 development corpus。BEIR 1,000 条 test query 会从 development adapter 中过滤。

## 5. Official BEIR test：严禁重新训练

```bash
python prepare_scidocs.py \
  --config configs/scidocs_beir_test.yaml \
  --stage graph

python prepare_scidocs.py \
  --config configs/scidocs_beir_test.yaml \
  --stage embeddings

python run_pipeline.py \
  --config configs/scidocs_beir_test.yaml \
  --stage graphrag

python prepare_ranker_dataset.py \
  --config configs/scidocs_beir_test.yaml

python run_pipeline.py \
  --config configs/scidocs_beir_test.yaml \
  --stage neural \
  --split test

python evaluate.py \
  --config configs/scidocs_beir_test.yaml
```

不要对 `configs/scidocs_beir_test.yaml` 执行 `python train.py`。该配置的 `ranker.checkpoint` 明确指向 development
的 `best.pt`，test Neural 输出到独立的 `neural.test.jsonl`。

最终输出：

```text
outputs/scidocs/scidocs_beir_test/graphrag.jsonl       # 1,000 queries, Top100
outputs/scidocs/scidocs_beir_test/graphrag.test.jsonl  # manifest 对齐副本
outputs/scidocs/scidocs_beir_test/neural.test.jsonl    # 1,000 queries, Top20
outputs/scidocs/scidocs_beir_test/metrics.json
```

GraphRAG 报告 Recall@20/50/100；Neural 报告 Recall/NDCG/Hit@5/10/20 和 MRR。LLM 暂未加入这套配置。

## 6. 复跑规则

已有文件默认复用，不覆盖。只有确认要重建当前 SCIDOCS 对应资产时才加 `--force`。这些路径与 HotpotQA 完全隔离；
任何上述命令都不会写入 `outputs/hotpotqa*`、HotpotQA ranker dataset 或 checkpoint。
