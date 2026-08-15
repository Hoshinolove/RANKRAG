# 项目现状分析

## 数据口径

- `hotpot_train.json`：90,447 个训练问题。
- `hotpot_train_paragraphs.json`：483,696 条训练段落记录，是全局候选 corpus 来源。
- `outputs/kg_train_gliner_rebel/kg_extractions.jsonl`：完整运行后应为 483,696 行，是正式全局 KG 来源。
- `hotpot_dev_subset.json` 的 7,405 只是旧开发子集，不是训练集。
- `data/kg_extractions.jsonl` 的 66,635 行属于旧开发段落 KG，正式训练配置可以搁置。

## GraphRAG

旧实现只在每个 HotpotQA query 自带的 2–10 个 context 段落中排序，不能产生真正 Top100。当前正式配置已改为全局检索：

1. 全局段落按稳定 SHA1 ID 去重；
2. 全部段落 embedding 和 FAISS index 离线构建；
3. GLiNER + REBEL 结果离线构建段落-实体、实体-段落和实体关系索引；
4. query 做 semantic Top500；
5. 从 seed paragraph/entity 做最多 2-hop graph expansion；
6. 合并去重后交给 GraphRAG scorer 排真正 Top100。

context 只用于生成 evaluation positive IDs。候选生成流程不读取 positive IDs，因此不会发生 gold injection。

## Neural

Neural 与 GraphRAG 已彻底离线解耦：

```text
graphrag.jsonl
 -> prepare_ranker_dataset.py
 -> train/validation Tensor shards
 -> Candidate Set Transformer
```

训练 epoch 只读取 `[B,K,D]` Tensor。多 worker shard 分配先构造统一 epoch permutation，再按 worker 切片。validation no-positive query 计入指标分母并贡献 0。推理默认只处理 validation，train 诊断使用独立 `neural.train.jsonl`。

## 接口稳定性

GraphRAG 仍输出 `RankingResult` 和有序 `RankedCandidate`，字段保留 semantic/graph/rag score、evidence nodes、evidence edges、paths 和 rank。因此下游 Tensor preprocessing、Neural Top20 和 LLM Top10 接口不需要修改。

## 尚需远程完成的工作

本地没有 GPU，也没有完整的 483,696 行新 KG 输出，因此以下大任务必须在远程机器执行：

1. 校验完整 KG 行数；
2. 构建全量 corpus embedding、FAISS 和 global graph assets；
3. 对 90,447 query 运行全局 GraphRAG；
4. 生成 Tensor shards 并训练 100 epochs；
5. 对 validation 推理和评估。

直接复制的命令分别见 `RUN_GLOBAL_GRAPHRAG_CN.md` 和 `RUN_TRAIN_NEURAL_CN.md`。
