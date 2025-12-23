# Batch Timing Benchmark Tools

## 概述

提供三个脚本用于 C2C 模型的性能测试和分析：

1. **demo_torch_profiler.py** - 单个样本的详细 profiling
2. **batch_timing_benchmark.py** - 批量测试预定义的 prompts
3. **dataset_timing_benchmark.py** - 从数据集加载并批量测试

## 输出格式

所有脚本都会生成 **JSONL** 格式的输出文件（每行一个 JSON 对象），方便批量分析。

### JSONL 记录格式

```json
{
  "id": 1,
  "prompt": "Say hello in one short sentence.",
  "response": "Hello! How can I assist you today?",
  "model_config": {
    "base_model": "Qwen/Qwen3-0.6B",
    "teacher_model": "Qwen/Qwen3-4B-Base"
  },
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
    "tokens_per_second": 13.80
  },
  "decode_step_times_ms": [72.89, 73.76, 72.49, ...]
}
```

## 使用方法

### 1. 单样本 Profiling（详细分析）

```bash
python demo_torch_profiler.py
```

**输出文件：**
- `demo_timing_data.jsonl` - JSONL 格式计时数据
- `demo_timing_data.tsv` - TSV 格式表格数据
- `demo_profiler_trace.json` - Chrome tracing 数据（可在 chrome://tracing 查看）
- `demo_profiler_stats.json` - 操作统计

### 2. 批量测试（预定义 prompts）

```bash
python batch_timing_benchmark.py
```

测试 5 个内置的示例 prompts，输出到 `batch_timing_results.jsonl`

### 3. 数据集批量测试

#### 从 HuggingFace 数据集加载

```bash
# GSM8K 数据集（数学问题）
python dataset_timing_benchmark.py \
  --dataset gsm8k \
  --split test \
  --num_samples 100 \
  --output gsm8k_timing.jsonl

# MMLU 数据集
python dataset_timing_benchmark.py \
  --dataset cais/mmlu \
  --split test \
  --num_samples 500 \
  --output mmlu_timing.jsonl
```

#### 从本地 JSONL 文件加载

```bash
python dataset_timing_benchmark.py \
  --dataset_file my_prompts.jsonl \
  --max_new_tokens 512 \
  --output my_results.jsonl
```

**输入文件格式（my_prompts.jsonl）：**
```json
{"prompt": "What is the capital of France?"}
{"prompt": "Explain quantum computing."}
{"question": "Calculate 123 * 456"}
```

## 分析结果

### 查看 JSONL 文件

```bash
# 查看所有数据（格式化）
cat timing_results.jsonl | jq .

# 提取关键指标
cat timing_results.jsonl | jq -r '[.id, .timing.num_generated_tokens, .timing.tokens_per_second, .timing.decode_avg_ms] | @tsv'

# 计算平均性能
cat timing_results.jsonl | jq -s 'map(.timing.tokens_per_second) | add/length'

# 查看最慢的 10 个样本
cat timing_results.jsonl | jq -s 'sort_by(.timing.decode_total_ms) | reverse | .[0:10] | .[] | {id, prompt: .prompt[:50], decode_ms: .timing.decode_total_ms}'
```

### Python 分析脚本

```python
import json
import pandas as pd

# 加载数据
data = []
with open('timing_results.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# 转换为 DataFrame
df = pd.DataFrame([{
    'id': d['id'],
    'tokens': d['timing']['num_generated_tokens'],
    'tok_per_sec': d['timing']['tokens_per_second'],
    'decode_avg_ms': d['timing']['decode_avg_ms'],
    'base_prefill_ms': d['timing']['base_prefill_ms'],
    'teacher_prefill_ms': d['timing']['teacher_prefill_ms'],
    'projector_ms': d['timing']['projector_total_ms']
} for d in data])

# 统计分析
print(df.describe())

# 绘图
import matplotlib.pyplot as plt
df['decode_avg_ms'].hist(bins=50)
plt.xlabel('Decode Time (ms/token)')
plt.ylabel('Frequency')
plt.savefig('decode_time_distribution.png')
```

## 高级选项

### 自定义模型

```bash
python dataset_timing_benchmark.py \
  --dataset my_dataset \
  --base_model "Qwen/Qwen3-0.6B" \
  --teacher_model "Qwen/Qwen3-4B-Base" \
  --num_samples 1000 \
  --max_new_tokens 512 \
  --output custom_timing.jsonl
```

### 并行处理大数据集

```bash
# 分割数据集
total_samples=10000
batch_size=100
num_batches=$((total_samples / batch_size))

for i in $(seq 0 $((num_batches-1))); do
    start=$((i * batch_size))
    python dataset_timing_benchmark.py \
      --dataset my_dataset \
      --num_samples $batch_size \
      --output "timing_batch_${i}.jsonl" &
done
wait

# 合并结果
cat timing_batch_*.jsonl > all_timing_results.jsonl
```

## 计时指标说明

- **base_embedding_ms**: Base model 将输入 tokens 转换为 embeddings 的时间
- **base_prefill_ms**: Base model transformer 层处理时间（不含 embedding）
- **teacher_embedding_ms**: Teacher model embedding 转换时间
- **teacher_prefill_ms**: Teacher model transformer 层处理时间
- **projector_total_ms**: KV cache 投影总时间
- **projector_calls**: 投影调用次数（通常等于 layer 数量）
- **decode_total_ms**: 生成所有 tokens 的总时间
- **decode_avg_ms**: 平均每个 token 的生成时间
- **tokens_per_second**: 生成速度（tokens/秒）

## 性能基准

在 NVIDIA GPU 上的参考性能：

| 模型组合 | Prefill (ms) | Decode (ms/token) | Throughput (tok/s) |
|---------|-------------|------------------|-------------------|
| 0.6B + 0.5B | ~800 | ~50 | ~20 |
| 0.6B + 4B | ~950 | ~75 | ~13 |

## 故障排查

### CUDA 内存不足
减少 `--max_new_tokens` 或使用更小的 batch size

### 速度太慢
- 确认使用 GPU：`device=cuda`
- 检查是否有其他进程占用 GPU
- 考虑使用 flash attention（如果可用）

### 数据集加载失败
```bash
# 先测试数据集是否可访问
python -c "from datasets import load_dataset; print(load_dataset('gsm8k', split='test'))"
```
