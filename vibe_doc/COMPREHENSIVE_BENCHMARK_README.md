# Comprehensive Dataset Timing Benchmark

## 概述

基于 repo 中使用的所有数据集进行完整的性能基准测试，包括：

### 支持的数据集

1. **MMLU-Redux** (57 subjects) - 通用知识评估
   - 用于：evaluation recipes
   - 样本：57个学科，每个学科多个测试样本

2. **GSM8K** - 数学推理
   - 用于：training recipes
   - 样本：8,000+ math word problems

3. **MATH-500** - 高级数学
   - 用于：evaluation
   - 样本：500 math competition problems

4. **LongBench** (21 tasks) - 长文本理解
   - 用于：evaluation recipes
   - 任务：QA, summarization, retrieval等

5. **OpenHermes** - 指令跟随
   - 用于：training recipes (C2C_0.6+0.5.json, baseline_config.json)

6. **OpenBookQA** - 常识推理
   - 用于：evaluation

7. **AI2-ARC** - 科学问答
   - 用于：evaluation

## 快速开始

### 方法 1: 运行所有基准测试（推荐用于全面评估）

```bash
./run_all_benchmarks.sh
```

这会运行所有数据集的子集，结果保存在 `timing_results/` 目录。

### 方法 2: 单个数据集测试

```bash
# GSM8K
python comprehensive_timing_benchmark.py --dataset gsm8k --num_samples 500

# MMLU-Redux (特定学科)
python comprehensive_timing_benchmark.py --dataset mmlu-redux --subject high_school_mathematics --num_samples 100

# MMLU-Redux (所有学科)
python comprehensive_timing_benchmark.py --dataset mmlu-redux --all_subjects --num_samples 20

# LongBench (特定任务)
python comprehensive_timing_benchmark.py --dataset longbench --subject narrativeqa --num_samples 50

# MATH-500
python comprehensive_timing_benchmark.py --dataset math-500 --num_samples 200

# OpenBookQA
python comprehensive_timing_benchmark.py --dataset openbookqa --num_samples 100
```

## 输出格式

### JSONL 文件结构

每个数据集/学科生成一个 JSONL 文件：`{dataset}_{subject}_timing.jsonl`

```json
{
  "id": 0,
  "dataset": "mmlu-redux",
  "subject": "high_school_mathematics",
  "prompt": "Question: What is the derivative of x^2?...",
  "response": "The derivative is 2x...",
  "prompt_length": 150,
  "response_length": 45,
  "timing": {
    "base_embedding_ms": 432.03,
    "base_prefill_ms": 267.71,
    "teacher_embedding_ms": 0.20,
    "teacher_prefill_ms": 196.75,
    "projector_total_ms": 133.96,
    "projector_avg_ms": 4.78,
    "projector_calls": 28,
    "decode_total_ms": 651.95,
    "decode_avg_ms": 72.44,
    "num_generated_tokens": 9,
    "tokens_per_second": 13.80,
    "total_time_ms": 1682.60
  },
  "decode_step_times_ms": [72.89, 73.76, ...]
}
```

## 数据分析

### 查看结果

```bash
# 查看特定数据集结果
cat timing_results/gsm8k_main_timing.jsonl | jq .

# 查看前10个样本
cat timing_results/mmlu-redux_high_school_mathematics_timing.jsonl | head -10 | jq .

# 提取关键指标到 TSV
cat timing_results/*.jsonl | jq -r '[.dataset, .subject, .timing.tokens_per_second, .timing.decode_avg_ms] | @tsv' > summary.tsv
```

### 统计分析

```bash
# 计算平均 tokens/sec (所有数据集)
cat timing_results/*.jsonl | jq -s 'map(.timing.tokens_per_second) | add/length'

# 按数据集分组统计
cat timing_results/*.jsonl | jq -s 'group_by(.dataset) | map({dataset: .[0].dataset, count: length, avg_tps: (map(.timing.tokens_per_second) | add/length)})'

# 最慢的 10 个样本
cat timing_results/*.jsonl | jq -s 'sort_by(.timing.total_time_ms) | reverse | .[0:10] | .[] | {dataset, subject, total_ms: .timing.total_time_ms}'
```

### Python 分析脚本

```python
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 加载所有结果
data = []
for jsonl_file in Path("timing_results").glob("*.jsonl"):
    with open(jsonl_file) as f:
        for line in f:
            data.append(json.loads(line))

# 转换为 DataFrame
df = pd.DataFrame([{
    'dataset': d['dataset'],
    'subject': d['subject'],
    'tokens_per_sec': d['timing']['tokens_per_second'],
    'decode_avg_ms': d['timing']['decode_avg_ms'],
    'total_time_ms': d['timing']['total_time_ms'],
    'num_tokens': d['timing']['num_generated_tokens']
} for d in data])

# 按数据集分组统计
summary = df.groupby('dataset').agg({
    'tokens_per_sec': ['mean', 'std', 'min', 'max'],
    'decode_avg_ms': ['mean', 'std'],
    'total_time_ms': 'mean',
    'subject': 'count'
}).round(2)

print(summary)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Throughput by dataset
df.groupby('dataset')['tokens_per_sec'].mean().plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title('Average Throughput by Dataset')
axes[0,0].set_ylabel('Tokens/Second')

# 2. Decode time distribution
df['decode_avg_ms'].hist(bins=50, ax=axes[0,1])
axes[0,1].set_title('Decode Time Distribution')
axes[0,1].set_xlabel('ms/token')

# 3. Total time vs num tokens
axes[1,0].scatter(df['num_tokens'], df['total_time_ms'], alpha=0.5)
axes[1,0].set_xlabel('Number of Tokens')
axes[1,0].set_ylabel('Total Time (ms)')
axes[1,0].set_title('Total Time vs Token Count')

# 4. Boxplot by dataset
df.boxplot(column='tokens_per_sec', by='dataset', ax=axes[1,1])
axes[1,1].set_title('Throughput Distribution by Dataset')

plt.tight_layout()
plt.savefig('timing_analysis.png', dpi=300)
print("Saved: timing_analysis.png")
```

## 按数据集使用的 Recipes

### Training Recipes

1. **OpenHermes** 
   - `C2C_0.6+0.5.json`
   - `baseline_config.json`
   - 用途：通用指令微调

2. **MMLU**
   - `oracle.json`
   - `baseline_partial_config.json`
   - 用途：知识评估训练

### Evaluation Recipes

1. **MMLU-Redux**
   - `unified_eval.yaml`
   - `ablation_base.yaml`
   - 用途：通用知识评估

2. **LongBench**
   - `unified_eval.yaml`
   - 用途：长文本理解

## 高级用法

### 自定义模型配置

```bash
python comprehensive_timing_benchmark.py \
    --dataset gsm8k \
    --num_samples 1000 \
    --base_model "Qwen/Qwen3-0.6B" \
    --teacher_model "Qwen/Qwen3-4B-Base" \
    --fuser_checkpoint "qwen3_0.6b+qwen3_4b_base_Fuser" \
    --max_new_tokens 512 \
    --output_dir custom_results
```

### 并行运行多个 subjects

```bash
# MMLU-Redux 并行处理
subjects=("physics" "chemistry" "biology" "mathematics")
for subject in "${subjects[@]}"; do
    python comprehensive_timing_benchmark.py \
        --dataset mmlu-redux \
        --subject $subject \
        --num_samples 100 \
        --output_dir timing_results &
done
wait
```

### 批量处理 LongBench

```bash
# Run all LongBench tasks
longbench_tasks=(
    "narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" 
    "2wikimqa" "musique" "gov_report" "qmsum" 
    "multi_news" "trec" "triviaqa" "samsum"
)

for task in "${longbench_tasks[@]}"; do
    python comprehensive_timing_benchmark.py \
        --dataset longbench \
        --subject $task \
        --num_samples 50 \
        --output_dir timing_results
done
```

## 预期性能（参考）

基于 Qwen3-0.6B + Qwen3-4B-Base：

| Dataset | Avg Throughput | Avg Decode Time | Notes |
|---------|---------------|-----------------|-------|
| GSM8K | ~13 tok/s | ~75 ms/token | 数学推理较慢 |
| MMLU-Redux | ~14 tok/s | ~72 ms/token | 选择题较快 |
| LongBench | ~12 tok/s | ~80 ms/token | 长上下文影响 |
| MATH-500 | ~12 tok/s | ~82 ms/token | 复杂数学 |

## Troubleshooting

### CUDA 内存不足
```bash
# 减少 max_new_tokens
python comprehensive_timing_benchmark.py --dataset gsm8k --max_new_tokens 128

# 或使用更小的 batch
python comprehensive_timing_benchmark.py --dataset gsm8k --num_samples 50
```

### 数据集加载失败
```bash
# 测试数据集是否可访问
python -c "from datasets import load_dataset; ds = load_dataset('openai/gsm8k', 'main', split='test'); print(len(ds))"

# 使用缓存
export HF_DATASETS_CACHE="/path/to/cache"
```

### 速度慢
- 确认 GPU 使用：`nvidia-smi`
- 检查其他进程：`fuser -v /dev/nvidia*`
- 使用 `--num_samples` 限制样本数量

## 输出文件命名

```
timing_results/
├── gsm8k_main_timing.jsonl
├── mmlu-redux_high_school_mathematics_timing.jsonl
├── mmlu-redux_physics_timing.jsonl
├── longbench_narrativeqa_timing.jsonl
├── math-500_main_timing.jsonl
└── openbookqa_main_timing.jsonl
```

## 下一步

1. **聚合分析**: 使用 Python 脚本分析所有 JSONL 文件
2. **可视化**: 生成性能对比图表
3. **优化**: 基于 breakdown 数据识别瓶颈
4. **对比**: 与 baseline 模型比较性能

更多详情见：`BATCH_BENCHMARK_README.md`
