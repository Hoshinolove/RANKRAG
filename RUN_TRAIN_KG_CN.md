# 训练集完整 KG 抽取与 GraphRAG 操作手册

这份文档面向第一次运行本项目的用户。你只需要按顺序复制命令执行，不需要先理解项目代码。

本手册完成的任务是：

```text
483,696 个训练段落
        -> GLiNER 抽取实体
        -> REBEL 抽取关系
        -> 合并为训练集 KG
        -> 使用完整 KG 运行 GraphRAG
```

## 0. 你需要准备什么

请在远程 Linux 服务器上准备：

- 已上传整个项目目录 `11rankRAG/code`
- Python 3.10 或更高版本
- 推荐 NVIDIA GPU，显存建议 24 GB 或更多
- 至少几十 GB 可用磁盘空间
- 可以访问 Hugging Face，以便第一次下载 GLiNER 和 REBEL 模型

本流程不会使用已有的 `data/kg_extractions.jsonl`（66,635 条），而是重新为训练段落抽取 KG。

## 1. 进入项目目录

把下面的路径替换成你上传项目的真实路径。下面假设项目位于 `/data/11rankRAG/code`：

```bash
# 进入项目根目录
cd /data/11rankRAG/code

# 确认当前目录正确，应该能看到 data、src、configs 等目录
pwd
ls
```

如果 `ls` 看不到 `data` 和 `configs`，说明还没有进入正确目录，不要继续执行。

## 2. 创建 Python 虚拟环境

如果服务器已经有项目专用环境，也可以跳过这一节。

```bash
# 创建一个独立的 Python 环境
python3 -m venv .venv

# 激活环境；以后每次重新登录服务器都要执行这一行
source .venv/bin/activate

# 升级安装工具
python -m pip install --upgrade pip setuptools wheel
```

检查 Python：

```bash
python --version
```

应该显示 Python 3.10 或更高版本。

## 3. 安装依赖

```bash
# 安装基础依赖、PyTorch、GLiNER、REBEL 所需库
python -m pip install -r requirements-kg.txt
```

检查 GPU 是否能被 PyTorch 使用：

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count())"
```

如果显示：

```text
cuda_available= True
gpu_count= 1
```

说明 GPU 可用。如果显示 `False`，仍然可以运行，但 GLiNER + REBEL 抽取会非常慢。建议先检查服务器 CUDA、驱动和 PyTorch 安装，不要直接启动完整抽取。

## 4. 检查输入数据

```bash
# 检查训练段落文件是否存在
test -f data/hotpot_train_paragraphs.json && echo "input file: OK" || echo "input file: MISSING"

# 查看训练段落文件大小
du -h data/hotpot_train_paragraphs.json
```

本次 KG 抽取的输入必须是：

```text
data/hotpot_train_paragraphs.json
```

它包含 483,696 条训练段落记录。

## 5. 先运行一个小测试

完整抽取可能运行很久。先用 1 张 GPU 确认模型能够正常下载和加载。

当前脚本默认会处理全部数据，因此小测试建议临时复制一个很小的输入文件：

```bash
# 创建测试目录
mkdir -p outputs/kg_smoke

# 复制训练段落文件的前几条 JSON 记录到测试文件
python - <<'PY'
import json
from pathlib import Path

source = json.loads(Path('data/hotpot_train_paragraphs.json').read_text(encoding='utf-8'))
Path('outputs/kg_smoke/input.json').write_text(
    json.dumps(source[:4], ensure_ascii=False),
    encoding='utf-8',
)
print('smoke records:', len(source[:4]))
PY
```

将测试配置复制一份：

```bash
# 复制正式配置，不修改正式配置文件
cp configs/kg_train_gliner_rebel.yaml configs/kg_smoke.yaml

# 将测试输入和测试输出写入测试配置
sed -i "s#data/hotpot_train_paragraphs.json#outputs/kg_smoke/input.json#" configs/kg_smoke.yaml
sed -i "s#outputs/kg_train_gliner_rebel#outputs/kg_smoke/result#" configs/kg_smoke.yaml
```

运行测试抽取：

```bash
python scripts/extract_kg_gliner_rebel.py \
  --config configs/kg_smoke.yaml
```

检查测试结果：

```bash
# 应该看到 4 条 JSONL 记录
wc -l outputs/kg_smoke/result/kg_extractions.jsonl

# 查看第一条记录的开头
head -c 1000 outputs/kg_smoke/result/kg_extractions.jsonl
echo
```

如果测试失败，先不要运行完整数据，查看错误信息。常见原因是模型下载失败、显存不足或 CUDA/PyTorch 不匹配。

## 6. 单 GPU 抽取完整训练 KG

确认小测试成功后，执行正式抽取：

```bash
# 使用 GPU 0 抽取全部 483,696 个训练段落
CUDA_VISIBLE_DEVICES=0 python scripts/extract_kg_gliner_rebel.py \
  --config configs/kg_train_gliner_rebel.yaml
```

抽取结果会实时写入：

```text
outputs/kg_train_gliner_rebel/kg_extractions.jsonl
```

查看运行日志时，看到类似下面内容表示正在处理：

```text
shard=0 processed=100 written=100
```

脚本默认支持断点续跑。如果程序因为服务器断开或任务超时中断，再执行相同命令即可：

```bash
# 默认会跳过已经写入的记录，只继续处理剩余段落
CUDA_VISIBLE_DEVICES=0 python scripts/extract_kg_gliner_rebel.py \
  --config configs/kg_train_gliner_rebel.yaml \
  --resume
```

## 7. 多 GPU 并行抽取（推荐）

如果服务器有 8 张 GPU，把数据切成 8 片。每个命令只使用一张 GPU。

打开 8 个终端，分别执行下面 8 条命令：

```bash
# 第 0 片，使用 GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 0

# 第 1 片，使用 GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 1

# 第 2 片，使用 GPU 2
CUDA_VISIBLE_DEVICES=2 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 2

# 第 3 片，使用 GPU 3
CUDA_VISIBLE_DEVICES=3 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 3

# 第 4 片，使用 GPU 4
CUDA_VISIBLE_DEVICES=4 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 4

# 第 5 片，使用 GPU 5
CUDA_VISIBLE_DEVICES=5 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 5

# 第 6 片，使用 GPU 6
CUDA_VISIBLE_DEVICES=6 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 6

# 第 7 片，使用 GPU 7
CUDA_VISIBLE_DEVICES=7 python scripts/extract_kg_gliner_rebel.py --config configs/kg_train_gliner_rebel.yaml --num-shards 8 --shard-index 7
```

如果只有 4 张 GPU，就把 `8` 全部改成 `4`，并只运行 `shard-index 0` 到 `3`。

查看分片是否生成：

```bash
ls -lh outputs/kg_train_gliner_rebel/kg_extractions.part-*.jsonl
```

## 8. 合并并检查完整 KG

只有确认所有分片都结束后，才能执行合并：

```bash
python scripts/merge_kg_jsonl.py \
  --input-dir outputs/kg_train_gliner_rebel \
  --output outputs/kg_train_gliner_rebel/kg_extractions.jsonl \
  --expected-count 483696
```

成功时应该看到：

```text
merged_files=8 records=483696 output=...
```

如果你的分片数量不是 8，`merged_files` 应该等于你的 `--num-shards`。

再次检查最终文件：

```bash
# 行数必须是 483,696
wc -l outputs/kg_train_gliner_rebel/kg_extractions.jsonl

# 查看第一条 KG 记录
head -c 1500 outputs/kg_train_gliner_rebel/kg_extractions.jsonl
echo
```

## 9. 使用完整训练 KG 运行 GraphRAG

KG 合并成功后，执行：

```bash
python run_pipeline.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage graphrag \
  --force
```

GraphRAG 输出位置：

```text
outputs/hotpotqa/hotpot_train_fullkg_graphrag/graphrag.jsonl
```

检查 GraphRAG 是否处理完全部训练问题：

```bash
wc -l outputs/hotpotqa/hotpot_train_fullkg_graphrag/graphrag.jsonl
```

预期结果：

```text
90447
```

## 10. 评估 GraphRAG

```bash
python evaluate.py \
  --config configs/hotpotqa_train_fullkg.yaml \
  --stage graphrag
```

指标文件：

```text
outputs/hotpotqa/hotpot_train_fullkg_graphrag/metrics.json
```

## 11. 训练 Neural Ranker（可选）

GraphRAG 完成后，可以训练 Neural Ranker：

```bash
python train.py \
  --config configs/hotpotqa_train_fullkg.yaml
```

训练完成后会生成：

```text
outputs/hotpotqa/hotpot_train_fullkg_graphrag/ranker.pt
```

## 12. 显存不足时怎么办

打开配置文件：

```bash
nano configs/kg_train_gliner_rebel.yaml
```

把：

```yaml
batch_size: 8
```

改成：

```yaml
batch_size: 2
```

如果仍然显存不足，改成：

```yaml
batch_size: 1
```

保存后重新执行抽取命令。已有结果会自动跳过，不会重复处理已经完成的段落。

## 13. 重要提醒

- 必须先完成所有 KG 分片，再执行合并。
- 合并命令的 `--expected-count 483696` 不要删除，它会检查是否漏数据。
- 运行 GraphRAG 时必须使用 `configs/hotpotqa_train_fullkg.yaml`，不能继续使用旧的 `configs/hotpotqa_train.yaml`。
- 如果任务中断，优先重新执行原命令，不要删除已有 JSONL；脚本会断点续跑。
- 如果模型下载失败，先确认服务器能访问 Hugging Face。
