# 🚀 完整的数据集时间性能基准测试工具套件

## ✅ 已完成功能

### 1. 核心脚本

- ✅ **demo_torch_profiler.py** - 单样本详细分析（含 Chrome trace）
- ✅ **batch_timing_benchmark.py** - 批量测试内置样本
- ✅ **dataset_timing_benchmark.py** - 从 HuggingFace/本地加载数据集
- ✅ **comprehensive_timing_benchmark.py** - 支持 repo 所有数据集
- ✅ **run_all_benchmarks.sh** - 一键运行所有基准测试

### 2. 支持的数据集（基于 repo recipes）

#### Training Datasets
- ✅ **OpenHermes** - 通用指令微调 (C2C_0.6+0.5.json, baseline_config.json)
- ✅ **MMLU** - 知识评估训练 (oracle.json, baseline_partial_config.json)

#### Evaluation Datasets  
- ✅ **MMLU-Redux** (57 subjects) - 通用知识 (unified_eval.yaml, ablation_base.yaml)
- ✅ **GSM8K** - 数学推理
- ✅ **MATH-500** - 高级数学
- ✅ **LongBench** (21 tasks) - 长文本理解 (unified_eval.yaml)
- ✅ **OpenBookQA** - 常识推理
- ✅ **AI2-ARC** - 科学问答
- ✅ **MMLU-Pro** - 进阶评估

### 3. 计时 Breakdown

每个样本记录：
```json
{
  "timing": {
    "base_embedding_ms": 432.03,      // Base model embedding 转换
    "base_prefill_ms": 267.71,        // Base model 前向计算
    "teacher_embedding_ms": 0.20,     // Teacher model embedding
    "teacher_prefill_ms": 196.75,     // Teacher model 前向计算
    "projector_total_ms": 133.96,     // KV cache 投影总时间
    "projector_avg_ms": 4.78,         // 每层投影平均时间
    "projector_calls": 28,            // 投影调用次数
    "decode_total_ms": 651.95,        // 总解码时间
    "decode_avg_ms": 72.44,           // 平均每 token 时间
    "num_generated_tokens": 9,        // 生成 token 数
    "tokens_per_second": 13.80,       // 生成速度
    "total_time_ms": 1682.60          // 总时间
  },
  "decode_step_times_ms": [...]       // 每个 token 详细时间
}
```

## 🎯 使用方法

### 快速开始

```bash
# 1. 运行所有基准测试
./run_all_benchmarks.sh

# 2. 单个数据集
python comprehensive_timing_benchmark.py --dataset gsm8k --num_samples 100

# 3. 特定学科/任务
python comprehensive_timing_benchmark.py --dataset mmlu-redux --subject physics --num_samples 50

# 4. 所有学科
python comprehensive_timing_benchmark.py --dataset mmlu-redux --all_subjects --num_samples 20
```

### 数据集示例

```bash
# GSM8K (数学推理)
python comprehensive_timing_benchmark.py --dataset gsm8k --num_samples 500

# MMLU-Redux (特定学科)
python comprehensive_timing_benchmark.py --dataset mmlu-redux --subject high_school_mathematics

# LongBench (长文本)
python comprehensive_timing_benchmark.py --dataset longbench --subject narrativeqa --num_samples 30

# MATH-500 (高级数学)
python comprehensive_timing_benchmark.py --dataset math-500 --num_samples 200
```

## 📊 输出和分析

### 输出文件

```
timing_results/
├── gsm8k_main_timing.jsonl                      # GSM8K 结果
├── mmlu-redux_physics_timing.jsonl              # MMLU Physics
├── mmlu-redux_high_school_mathematics_timing.jsonl
├── longbench_narrativeqa_timing.jsonl           # LongBench QA
└── math-500_main_timing.jsonl                   # MATH-500
```

### 分析命令

```bash
# 查看结果
cat timing_results/gsm8k_main_timing.jsonl | jq .

# 计算平均性能
cat timing_results/*.jsonl | jq -s 'map(.timing.tokens_per_second) | add/length'

# 按数据集统计
cat timing_results/*.jsonl | jq -s 'group_by(.dataset) | map({
  dataset: .[0].dataset, 
  count: length, 
  avg_tps: (map(.timing.tokens_per_second) | add/length)
})'

# 导出 TSV
cat timing_results/*.jsonl | jq -r '[.dataset, .subject, .timing.tokens_per_second, .timing.decode_avg_ms] | @tsv' > analysis.tsv
```

### Python 分析

```python
import json
import pandas as pd
from pathlib import Path

# 加载所有结果
data = []
for f in Path("timing_results").glob("*.jsonl"):
    with open(f) as file:
        for line in file:
            data.append(json.loads(line))

# 创建 DataFrame
df = pd.DataFrame([{
    'dataset': d['dataset'],
    'subject': d['subject'],
    'tps': d['timing']['tokens_per_second'],
    'decode_ms': d['timing']['decode_avg_ms'],
    'total_ms': d['timing']['total_time_ms']
} for d in data])

# 统计
print(df.groupby('dataset').agg({
    'tps': ['mean', 'std'],
    'decode_ms': 'mean',
    'subject': 'count'
}))

# 可视化
df.groupby('dataset')['tps'].mean().plot(kind='bar')
```

## 📁 文件清单

### 核心脚本
- `demo_torch_profiler.py` - 单样本 profiling + Chrome trace
- `batch_timing_benchmark.py` - 批量内置样本
- `dataset_timing_benchmark.py` - 通用数据集加载
- `comprehensive_timing_benchmark.py` - **完整数据集支持**
- `run_all_benchmarks.sh` - **一键运行脚本**

### 文档
- `TIMING_README.md` - 单样本 profiling 说明
- `BATCH_BENCHMARK_README.md` - 批量测试说明
- `COMPREHENSIVE_BENCHMARK_README.md` - **完整数据集基准说明**
- `README_SUMMARY.md` - **本文件：总体概述**

### 输出示例
- `demo_timing_data.jsonl` - 单样本 JSONL
- `demo_timing_data.tsv` - 单样本 TSV
- `demo_profiler_trace.json` - Chrome trace
- `demo_profiler_stats.json` - 操作统计

## 🔬 测试状态

已验证运行的数据集：
- ✅ GSM8K (2 samples) - 成功
- ✅ 单样本 demo - 成功
- 🔄 其他数据集待测试

## 📈 性能参考

基于 Qwen3-0.6B + Qwen3-4B-Base：

| 阶段 | 时间 (ms) | 说明 |
|-----|-----------|------|
| Base Embedding | ~432 | Base model 输入转换 |
| Base Prefill | ~268 | Base model 计算 |
| Teacher Embedding | ~0.2 | Teacher model 输入 |
| Teacher Prefill | ~197 | Teacher model 计算 |
| Projector | ~134 (28 calls) | KV cache 投影 |
| Decode (per token) | ~72 | 单 token 生成 |
| **Total** | ~1683 | 完整推理 |
| **Throughput** | ~13.8 tok/s | 生成速度 |

## 🚀 下一步

1. **运行完整基准测试**
   ```bash
   ./run_all_benchmarks.sh
   ```

2. **分析结果**
   - 使用 jq 命令查看统计
   - Python 脚本进行深度分析
   - 生成可视化图表

3. **识别瓶颈**
   - 对比各阶段时间
   - 分析数据集差异
   - 优化慢速部分

4. **对比实验**
   - 不同模型组合
   - 不同数据集性能
   - Baseline vs C2C

## 📞 使用支持

查看详细文档：
- 完整数据集基准：`COMPREHENSIVE_BENCHMARK_README.md`
- 批量测试：`BATCH_BENCHMARK_README.md`
- 单样本分析：`TIMING_README.md`

## ✨ 关键特性

1. **完整数据集覆盖** - 支持 repo 中所有使用的数据集
2. **详细时间 breakdown** - 分离 embedding, prefill, projection, decode
3. **JSONL 格式** - 易于批量分析和聚合
4. **一键运行** - `run_all_benchmarks.sh` 自动化测试
5. **灵活配置** - 支持自定义模型、采样数、输出路径
6. **进度追踪** - tqdm 进度条实时显示
7. **错误处理** - 单样本失败不影响其他样本

---

**开始测试：** `./run_all_benchmarks.sh`
