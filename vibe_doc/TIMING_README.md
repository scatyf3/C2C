# C2C Profiling & Timing Documentation

## Overview
The `demo_torch_profiler.py` script provides detailed performance profiling and timing measurements for the C2C (Cache-to-Cache) model inference pipeline.

## Generated Files

### 1. `demo_timing_data.tsv`
Tab-separated values file containing detailed timing breakdowns:

- **Base Model Embedding**: Time for base model to convert input tokens to embeddings
- **Base Model Prefill (computation)**: Base model forward pass computation time (excluding embedding)
- **Teacher Model Embedding**: Time for teacher model embedding conversion
- **Teacher Model Prefill (computation)**: Teacher model forward pass computation time
- **Projector (KV projection)**: Time spent projecting KV cache from teacher to base model
- **Decode Step (per token)**: Individual decode step timings for each generated token

### 2. `demo_profiler_trace.json`
Chrome trace format profiling data that can be visualized at `chrome://tracing`

### 3. `demo_profiler_stats.json`
JSON summary of top operations by CPU and CUDA time

## Current Configuration

- **Base Model**: Qwen/Qwen3-0.6B
- **Teacher Model**: Qwen/Qwen3-4B-Base
- **Fuser Checkpoint**: qwen3_0.6b+qwen3_4b_base_Fuser

## Key Timing Stages

### Prefill Phase
1. **Base Model Embedding**: Converts input tokens to embeddings (~915ms total, ~183ms avg per call)
2. **Base Model Computation**: Processes embeddings through transformer layers (~394ms total)
3. **Teacher Model Embedding**: Teacher model embedding conversion (~0.2ms)
4. **Teacher Model Computation**: Teacher generates KV cache (~198ms)
5. **KV Projection**: Projects teacher KV cache to base model dimensions (~133ms, 28 projector calls)

### Decode Phase
- **Per-token Generation**: Each token takes ~75ms on average
- Multiple decode steps shown individually in TSV file

## Performance Insights

From the example run:
- Total prefill time: ~1.6 seconds
- Per-token decode time: ~75ms (13 tokens/second)
- Projector overhead: ~5ms per layer projection
- Teacher model is only called once during prefill (for instruction tokens)

## Viewing Trace Data

```bash
# The trace file can be viewed in Chrome DevTools
# 1. Open Chrome browser
# 2. Navigate to chrome://tracing
# 3. Load demo_profiler_trace.json
```

## Modifying the Script

To test different model combinations, update the `model_config` section:

```python
model_config = {
    "rosetta_config": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "Qwen/Qwen3-4B-Base",  # Change this
        "checkpoints_dir": f"{checkpoint_dir}/...",  # Update path
    }
}
```

## Understanding the Timing

The timing measurements use CUDA synchronization points to ensure accurate GPU timing:
- `torch.cuda.synchronize()` is called before and after each measured operation
- All times are in milliseconds
- The TSV format allows easy import into Excel/spreadsheet tools for further analysis
