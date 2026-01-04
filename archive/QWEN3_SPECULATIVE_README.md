# Qwen3-14B + Qwen3-0.6B 推测解码

这个脚本展示了如何使用 Qwen3-14B 作为目标模型（Target Model）和 Qwen3-0.6B 作为草稿模型（Draft Model）进行推测解码（Speculative Decoding）。

## 模型架构

- **目标模型（Target Model）**: Qwen/Qwen3-14B - 14B参数的大型模型，生成高质量输出
- **草稿模型（Draft Model）**: Qwen/Qwen3-0.6B - 0.6B参数的小型模型，快速生成候选token
- **KV Fusion**: 使用C2C Fuser进行KV缓存融合，提升推测解码效率

## 推测解码原理

推测解码通过以下方式加速生成：

1. **快速草稿生成**: 小模型（Qwen3-0.6B）快速生成K个候选token
2. **并行验证**: 大模型（Qwen3-14B）并行验证这些候选token
3. **接受/拒绝**: 接受正确的token，拒绝错误的token
4. **加速效果**: 在保持输出质量的同时，显著提升生成速度

## 快速开始

### 1. 基础使用（默认提示词）

```bash
python qwen3_14b_speculative_demo.py
```

或者使用便捷脚本：

```bash
./run_qwen3_14b_speculative.sh
```

### 2. 使用自定义提示词

```bash
python qwen3_14b_speculative_demo.py --prompts example_prompts.txt
```

### 3. 调整推测窗口大小

```bash
# 使用更大的推测窗口（可能获得更高的加速比）
python qwen3_14b_speculative_demo.py --gamma 6

# 使用较小的推测窗口（更保守，但可能更稳定）
python qwen3_14b_speculative_demo.py --gamma 3
```

### 4. 仅运行推测解码（跳过标准生成）

```bash
python qwen3_14b_speculative_demo.py --skip_standard --save_results
```

### 5. 长文本生成

```bash
python qwen3_14b_speculative_demo.py --max_new_tokens 1024 --gamma 5
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--prompts` | str | None | 提示词文件路径（每行一个提示词） |
| `--max_new_tokens` | int | 256 | 最大生成token数 |
| `--gamma` | int | 4 | 推测窗口大小（K值） |
| `--base_model` | str | Qwen/Qwen3-0.6B | 草稿模型路径 |
| `--teacher_model` | str | Qwen/Qwen3-14B | 目标模型路径 |
| `--fuser_checkpoint` | str | qwen3_0.6b+qwen3_14b_Fuser | Fuser checkpoint名称 |
| `--skip_standard` | flag | False | 跳过标准生成（仅运行推测解码） |
| `--no_prefill_fusion` | flag | False | 禁用预填充阶段的KV融合 |
| `--no_decode_fusion` | flag | False | 禁用解码阶段的KV融合 |
| `--save_results` | flag | False | 保存结果到JSON文件 |
| `--output_dir` | str | output/qwen3_speculative | 结果保存目录 |

## 性能指标

脚本会输出以下性能指标：

### 标准生成（Baseline）
- **延迟（Latency）**: 生成所有token的总时间
- **吞吐量（Throughput）**: 每秒生成的token数
- **生成token数**: 实际生成的token数量

### 推测解码
- **延迟（Latency）**: 生成所有token的总时间
- **吞吐量（Throughput）**: 每秒生成的token数
- **接受率（Acceptance Rate）**: 草稿模型生成的token被接受的比例
- **加速比（Speedup）**: 相对于标准生成的理论加速倍数
- **平均接受长度**: 每次验证平均接受的token数
- **草稿模型调用次数**: Draft model前向传播次数
- **目标模型调用次数**: Target model前向传播次数

### 性能对比
- **延迟改善**: 推测解码相对标准生成的实际加速倍数
- **吞吐量提升**: 推测解码相对标准生成的吞吐量提升倍数

## 示例输出

```
================================================================================
Qwen3-14B + Qwen3-0.6B 推测解码演示
================================================================================

提示词数量: 7
最大生成token数: 256
推测窗口大小 (γ): 4
草稿模型（Draft）: Qwen/Qwen3-0.6B
目标模型（Target）: Qwen/Qwen3-14B
预填充融合: 启用
解码融合: 启用

加载模型中...
模型已加载到 cuda

================================================================================
提示词 1/7
================================================================================

输入: 请解释量子计算的基本原理。

运行标准生成（仅目标模型）...

[标准生成 - Qwen3-14B]
输出: 量子计算是基于量子力学原理的新型计算范式...
延迟: 2.456s
生成token数: 128
吞吐量: 52.11 tokens/s

运行推测解码生成...

[推测解码 - Qwen3-14B + Qwen3-0.6B (γ=4)]
输出: 量子计算是基于量子力学原理的新型计算范式...
延迟: 1.234s
生成token数: 128
吞吐量: 103.72 tokens/s
接受率: 68.50%
加速比: 1.95x
平均接受长度: 2.74
草稿模型调用次数: 47
目标模型调用次数: 47

[性能对比]
延迟改善: 1.99x 更快
吞吐量提升: 1.99x 更高
```

## 输出文件

如果使用 `--save_results` 参数，结果会保存到：

```
output/qwen3_speculative/results_20231229_143022.json
```

包含完整的配置、输入输出和性能指标。

## 提示词文件格式

提示词文件应该是纯文本文件，每行一个提示词：

```
提示词1
提示词2
提示词3
```

参考 `example_prompts.txt` 获取示例。

## 性能调优建议

### 1. 调整推测窗口大小（gamma）

- **较小值（2-3）**: 更保守，接受率可能较高，但加速效果有限
- **中等值（4-5）**: 平衡接受率和加速效果（推荐）
- **较大值（6-8）**: 可能获得更高加速，但接受率可能下降

### 2. 启用/禁用KV融合

```bash
# 测试不同融合配置的影响
python qwen3_14b_speculative_demo.py --no_prefill_fusion  # 仅禁用预填充融合
python qwen3_14b_speculative_demo.py --no_decode_fusion   # 仅禁用解码融合
```

### 3. 批量测试不同配置

```bash
# 创建测试脚本
for gamma in 3 4 5 6; do
    echo "Testing gamma=$gamma"
    python qwen3_14b_speculative_demo.py \
        --gamma $gamma \
        --skip_standard \
        --save_results \
        --output_dir output/qwen3_gamma_${gamma}
done
```

## 预期性能

根据推测解码理论和实践经验：

- **接受率**: 通常在 60-75% 范围内
- **加速比**: 1.5x - 2.5x（取决于gamma和任务）
- **吞吐量提升**: 与加速比相当
- **输出质量**: 与直接使用Qwen3-14B完全一致（数学上等价）

## 故障排除

### 1. CUDA内存不足

```bash
# 减少最大生成长度
python qwen3_14b_speculative_demo.py --max_new_tokens 128

# 减少推测窗口
python qwen3_14b_speculative_demo.py --gamma 3
```

### 2. 模型下载失败

确保可以访问Hugging Face，或设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 加速效果不明显

- 尝试调整gamma值
- 检查是否启用了KV融合
- 某些任务可能天然接受率较低

## 相关文件

- `qwen3_14b_speculative_demo.py` - 主演示脚本
- `run_qwen3_14b_speculative.sh` - 便捷运行脚本
- `example_prompts.txt` - 示例提示词文件
- `demo_speculative_decoding.py` - 通用推测解码演示
- `speculative_benchmark.py` - 完整的benchmark脚本

## 参考资料

- [Speculative Decoding论文](https://arxiv.org/abs/2211.17192)
- [C2C: Cross-Model KV Cache Fusion](相关论文/文档)
- [Qwen3模型文档](https://huggingface.co/Qwen)
