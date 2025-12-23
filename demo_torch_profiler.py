import torch
import logging
from huggingface_hub import snapshot_download
from script.playground.inference_example import load_rosetta_model
from torch.profiler import profile, record_function, ProfilerActivity
import json
import time
import csv

# 配置全局logging
logging.basicConfig(
    level=logging.WARNING,  # 减少日志输出，方便看profiler结果
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

checkpoint_dir = snapshot_download(
    repo_id="nics-efc/C2C_Fuser",
    allow_patterns=["qwen3_0.6b+qwen3_4b_base_Fuser/*"],
)

model_config = {
    "rosetta_config": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "Qwen/Qwen3-4B-Base",
        "checkpoints_dir": f"{checkpoint_dir}/qwen3_0.6b+qwen3_4b_base_Fuser/final",
    }
}

print("Loading model...")
rosetta_model, tokenizer = load_rosetta_model(model_config, eval_config={}, device=torch.device("cuda"))
device = rosetta_model.device

# Initialize timing data structure
timing_data = {
    'base_embedding_times': [],
    'base_model_prefill_times': [],
    'teacher_embedding_times': [],
    'teacher_model_prefill_times': [],
    'projector_times': [],
    'decode_step_times': []
}

# Monkey patch to add timing
original_forward = rosetta_model.forward
original_generate = rosetta_model.generate

def timed_forward(self, *args, **kwargs):
    # Check if we're in prefill or decode by looking at past_key_values
    past_key_values = kwargs.get('past_key_values', None)
    is_prefill = past_key_values is None or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0)
    
    if is_prefill:
        # Prefill phase - measure teacher and base model
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        
        # Call original forward
        result = original_forward(*args, **kwargs)
        
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        
        # After first prefill, switch to decode phase
        if hasattr(self, '_is_prefill_phase'):
            self._is_prefill_phase = False
    else:
        # Decode step
        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        
        result = original_forward(*args, **kwargs)
        
        torch.cuda.synchronize()
        decode_end = time.perf_counter()
        
        timing_data['decode_step_times'].append(decode_end - decode_start)
    
    return result

def timed_generate(self, *args, **kwargs):
    # Reset prefill flag for new generation
    self._is_prefill_phase = True
    
    return original_generate(*args, **kwargs)

# Apply monkey patches
rosetta_model.forward = lambda *args, **kwargs: timed_forward(rosetta_model, *args, **kwargs)
rosetta_model.generate = lambda *args, **kwargs: timed_generate(rosetta_model, *args, **kwargs)

# Add detailed hooks for base and teacher models
def add_detailed_timing_hooks(model):
    """Add hooks to track embedding and model computation separately"""
    
    base_model = model.model_list[model.base_model_idx]
    
    # Track prefill phase - only record during prefill, not decode
    model._is_prefill_phase = True
    
    # Base model computation hook (after embedding)
    original_base_forward = base_model.forward
    def base_forward_with_timing(*args, **kwargs):
        # Check if this is prefill or decode by looking at past_key_values
        past_kv = kwargs.get('past_key_values', None)
        is_prefill = past_kv is None or (hasattr(past_kv, 'get_seq_length') and past_kv.get_seq_length() == 0)
        
        # Only record timing during actual prefill phase
        if is_prefill and model._is_prefill_phase:
            # Time embedding
            torch.cuda.synchronize()
            embed_start = time.perf_counter()
            
            # Temporarily wrap embed to capture embedding time
            original_embed = base_model.model.embed_tokens.forward if hasattr(base_model.model, 'embed_tokens') else None
            embed_end_time = [None]
            
            if original_embed is not None:
                def embed_with_timing(*embed_args, **embed_kwargs):
                    result = original_embed(*embed_args, **embed_kwargs)
                    torch.cuda.synchronize()
                    embed_end_time[0] = time.perf_counter()
                    return result
                base_model.model.embed_tokens.forward = embed_with_timing
            
            # Execute forward
            result = original_base_forward(*args, **kwargs)
            torch.cuda.synchronize()
            compute_end = time.perf_counter()
            
            # Restore original embed
            if original_embed is not None:
                base_model.model.embed_tokens.forward = original_embed
                if embed_end_time[0] is not None:
                    timing_data['base_embedding_times'].append(embed_end_time[0] - embed_start)
                    timing_data['base_model_prefill_times'].append(compute_end - embed_end_time[0])
        else:
            result = original_base_forward(*args, **kwargs)
        
        return result
    
    base_model.forward = base_forward_with_timing
    
    # Teacher model hooks (if exists)
    if len(model.model_list) > 1:
        teacher_model = model.model_list[1]
        
        # Store original embed_tokens forward
        original_embed_forward = teacher_model.model.embed_tokens.forward if hasattr(teacher_model.model, 'embed_tokens') else None
        
        # Wrap teacher model forward to capture timing
        original_teacher_forward = teacher_model.forward
        def teacher_forward_with_timing(*args, **kwargs):
            # Timing for embedding
            torch.cuda.synchronize()
            embed_start = time.perf_counter()
            
            # Temporarily wrap embed_tokens to capture embedding time
            if original_embed_forward is not None:
                embed_end_time = [None]
                def embed_with_timing(*embed_args, **embed_kwargs):
                    result = original_embed_forward(*embed_args, **embed_kwargs)
                    torch.cuda.synchronize()
                    embed_end_time[0] = time.perf_counter()
                    return result
                teacher_model.model.embed_tokens.forward = embed_with_timing
            
            # Execute the full forward pass
            result = original_teacher_forward(*args, **kwargs)
            torch.cuda.synchronize()
            compute_end = time.perf_counter()
            
            # Restore original embed_tokens forward
            if original_embed_forward is not None:
                teacher_model.model.embed_tokens.forward = original_embed_forward
                if embed_end_time[0] is not None:
                    timing_data['teacher_embedding_times'].append(embed_end_time[0] - embed_start)
                    timing_data['teacher_model_prefill_times'].append(compute_end - embed_end_time[0])
            
            return result
        
        teacher_model.forward = teacher_forward_with_timing
    
    # Projector timing
    if len(model.projector_list) > 0:
        for projector in model.projector_list:
            original_proj_forward = projector.forward
            def proj_forward_with_timing(*args, _orig=original_proj_forward, **kwargs):
                torch.cuda.synchronize()
                proj_start = time.perf_counter()
                result = _orig(*args, **kwargs)
                torch.cuda.synchronize()
                proj_end = time.perf_counter()
                timing_data['projector_times'].append(proj_end - proj_start)
                return result
            projector.forward = proj_forward_with_timing

add_detailed_timing_hooks(rosetta_model)

prompt = [{"role": "user", "content": "Say hello in one short sentence."}]
prompt_text = prompt[0]["content"]  # Store original prompt
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(input_text, return_tensors="pt").to(device)

instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(inputs['input_ids'].shape[1] - 1, 1).unsqueeze(0).to(device)
label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
kv_cache_index = [instruction_index, label_index]

print("\nStarting profiling with torch profiler...")
print("="*80)

# Clear timing data for this sample
timing_data = {
    'base_embedding_times': [],
    'base_model_prefill_times': [],
    'teacher_embedding_times': [],
    'teacher_model_prefill_times': [],
    'projector_times': [],
    'decode_step_times': []
}

# 使用 torch profiler 进行性能分析
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with torch.no_grad():
        sampling_params = {
            'do_sample': False,
            'max_new_tokens': 256
        }
        outputs = rosetta_model.generate(**inputs, kv_cache_index=kv_cache_index, **sampling_params)
        output_text = tokenizer.decode(outputs[0, instruction_index.shape[1] + 1:], skip_special_tokens=True)

print(f"\nC2C output text: {output_text}")
print("\n" + "="*80)

# 保存详细的profiler结果
prof.export_chrome_trace("demo_profiler_trace.json")
print("Detailed trace saved to: demo_profiler_trace.json")
print("You can view it at: chrome://tracing")

# 按 CPU 时间排序的前20个操作
print("\n" + "="*80)
print("Top 20 operations by CPU time:")
print("="*80)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# 按 CUDA 时间排序的前20个操作
print("\n" + "="*80)
print("Top 20 operations by CUDA time:")
print("="*80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 按操作名称分组统计
print("\n" + "="*80)
print("Grouped by operation name:")
print("="*80)
print(prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", row_limit=30))

# 提取特定操作的统计信息
key_averages = prof.key_averages()

def find_ops_by_keyword(key_averages, keyword):
    """查找包含特定关键字的操作"""
    ops = []
    for evt in key_averages:
        if keyword.lower() in evt.key.lower():
            ops.append(evt)
    return ops

def summarize_ops(ops, name):
    """汇总操作统计"""
    if not ops:
        print(f"\n{name}: No operations found")
        return
    
    total_cpu = sum(op.cpu_time_total for op in ops)
    # Use device_time_total instead of cuda specific attributes
    total_cuda = sum(getattr(op, 'device_time_total', getattr(op, 'cuda_time', 0)) for op in ops)
    total_count = sum(op.count for op in ops)
    
    print(f"\n{name}:")
    print(f"  Total CPU time: {total_cpu/1000:.2f} ms")
    print(f"  Total CUDA time: {total_cuda/1000:.2f} ms")
    print(f"  Call count: {total_count}")
    print(f"  Operations breakdown:")
    for op in sorted(ops, key=lambda x: getattr(x, 'device_time_total', getattr(x, 'cuda_time', 0)), reverse=True)[:10]:
        cuda_time = getattr(op, 'device_time_total', getattr(op, 'cuda_time', 0))
        print(f"    - {op.key}: {cuda_time/1000:.2f} ms (CUDA), {op.cpu_time_total/1000:.2f} ms (CPU), count={op.count}")

print("\n" + "="*80)
print("Analysis of Key Components:")
print("="*80)

# 分析线性层操作（包括 projector 中的线性变换）
linear_ops = find_ops_by_keyword(key_averages, "linear") + find_ops_by_keyword(key_averages, "addmm") + find_ops_by_keyword(key_averages, "matmul")
summarize_ops(linear_ops, "Linear/Matrix Operations (including projections)")

# 分析注意力相关操作
attention_ops = find_ops_by_keyword(key_averages, "attention") + find_ops_by_keyword(key_averages, "scaled_dot_product")
summarize_ops(attention_ops, "Attention Operations")

# 分析归一化操作
norm_ops = find_ops_by_keyword(key_averages, "norm") + find_ops_by_keyword(key_averages, "layer_norm")
summarize_ops(norm_ops, "Normalization Operations")

# 分析激活函数
activation_ops = find_ops_by_keyword(key_averages, "gelu") + find_ops_by_keyword(key_averages, "relu") + find_ops_by_keyword(key_averages, "silu")
summarize_ops(activation_ops, "Activation Functions")

# 保存简化的统计信息到JSON
stats = {
    "total_operations": len(key_averages),
    "top_10_cpu_ops": [
        {
            "name": evt.key,
            "cpu_time_ms": evt.cpu_time_total / 1000,
            "cuda_time_ms": getattr(evt, 'device_time_total', getattr(evt, 'cuda_time', 0)) / 1000,
            "count": evt.count
        }
        for evt in sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]
    ],
    "top_10_cuda_ops": [
        {
            "name": evt.key,
            "cpu_time_ms": evt.cpu_time_total / 1000,
            "cuda_time_ms": getattr(evt, 'device_time_total', getattr(evt, 'cuda_time', 0)) / 1000,
            "count": evt.count
        }
        for evt in sorted(key_averages, key=lambda x: getattr(x, 'device_time_total', getattr(x, 'cuda_time', 0)), reverse=True)[:10]
    ]
}

with open("demo_profiler_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n" + "="*80)
print("Summary statistics saved to: demo_profiler_stats.json")
print("="*80)

# Save timing data as JSONL (one sample per line)
print("\n" + "="*80)
print("Saving timing data to JSONL format...")
print("="*80)

timing_record = {
    "prompt": prompt_text,
    "response": output_text,
    "model_config": {
        "base_model": model_config["rosetta_config"]["base_model"],
        "teacher_model": model_config["rosetta_config"]["teacher_model"]
    },
    "timing": {
        "base_embedding_ms": sum(timing_data['base_embedding_times']) * 1000 if timing_data['base_embedding_times'] else 0,
        "base_prefill_ms": sum(timing_data['base_model_prefill_times']) * 1000 if timing_data['base_model_prefill_times'] else 0,
        "teacher_embedding_ms": sum(timing_data['teacher_embedding_times']) * 1000 if timing_data['teacher_embedding_times'] else 0,
        "teacher_prefill_ms": sum(timing_data['teacher_model_prefill_times']) * 1000 if timing_data['teacher_model_prefill_times'] else 0,
        "projector_total_ms": sum(timing_data['projector_times']) * 1000 if timing_data['projector_times'] else 0,
        "projector_avg_ms": (sum(timing_data['projector_times']) / len(timing_data['projector_times'])) * 1000 if timing_data['projector_times'] else 0,
        "projector_calls": len(timing_data['projector_times']),
        "decode_total_ms": sum(timing_data['decode_step_times']) * 1000 if timing_data['decode_step_times'] else 0,
        "decode_avg_ms": (sum(timing_data['decode_step_times']) / len(timing_data['decode_step_times'])) * 1000 if timing_data['decode_step_times'] else 0,
        "decode_steps": timing_data['decode_step_times'] if timing_data['decode_step_times'] else [],
        "num_generated_tokens": len(timing_data['decode_step_times'])
    },
    "timing_breakdown_ms": [t * 1000 for t in timing_data['decode_step_times']] if timing_data['decode_step_times'] else []
}

# Append to JSONL file (one line per sample)
with open("demo_timing_data.jsonl", "a") as f:
    f.write(json.dumps(timing_record, ensure_ascii=False) + "\n")

print(f"Timing data appended to: demo_timing_data.jsonl")

# Save timing data to TSV
print("\n" + "="*80)
print("Saving detailed timing data...")
print("="*80)

with open("demo_timing_data.tsv", "w", newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    
    # Write header
    writer.writerow(["Stage", "Time(ms)", "Count", "Avg(ms)", "Total(ms)"])
    
    # Base model embedding
    if timing_data['base_embedding_times']:
        times = timing_data['base_embedding_times']
        writer.writerow([
            "Base Model Embedding",
            f"{times[0]*1000:.4f}",
            len(times),
            f"{sum(times)/len(times)*1000:.4f}",
            f"{sum(times)*1000:.4f}"
        ])
    
    # Base model prefill computation
    if timing_data['base_model_prefill_times']:
        times = timing_data['base_model_prefill_times']
        writer.writerow([
            "Base Model Prefill (computation)",
            f"{times[0]*1000:.4f}",
            len(times),
            f"{sum(times)/len(times)*1000:.4f}",
            f"{sum(times)*1000:.4f}"
        ])
    
    # Teacher model embedding
    if timing_data['teacher_embedding_times']:
        times = timing_data['teacher_embedding_times']
        writer.writerow([
            "Teacher Model Embedding",
            f"{times[0]*1000:.4f}",
            len(times),
            f"{sum(times)/len(times)*1000:.4f}",
            f"{sum(times)*1000:.4f}"
        ])
    
    # Teacher model prefill computation
    if timing_data['teacher_model_prefill_times']:
        times = timing_data['teacher_model_prefill_times']
        writer.writerow([
            "Teacher Model Prefill (computation)",
            f"{times[0]*1000:.4f}",
            len(times),
            f"{sum(times)/len(times)*1000:.4f}",
            f"{sum(times)*1000:.4f}"
        ])
    
    # Projector
    if timing_data['projector_times']:
        times = timing_data['projector_times']
        writer.writerow([
            "Projector (KV projection)",
            f"{times[0]*1000:.4f}",
            len(times),
            f"{sum(times)/len(times)*1000:.4f}",
            f"{sum(times)*1000:.4f}"
        ])
    
    # Decode steps
    if timing_data['decode_step_times']:
        times = timing_data['decode_step_times']
        writer.writerow([
            "Decode Step (per token)",
            f"{times[0]*1000:.4f}",
            len(times),
            f"{sum(times)/len(times)*1000:.4f}",
            f"{sum(times)*1000:.4f}"
        ])
        
        # Add individual decode step times
        writer.writerow([])
        writer.writerow(["Decode Step Details", "Step #", "Time(ms)", "", ""])
        for i, t in enumerate(times):
            writer.writerow(["", str(i+1), f"{t*1000:.4f}", "", ""])

print(f"Timing data saved to: demo_timing_data.tsv")
print("\nTiming Summary:")
print("-" * 80)
if timing_data['base_embedding_times']:
    print(f"Base Embedding: {sum(timing_data['base_embedding_times'])*1000:.2f} ms")
if timing_data['base_model_prefill_times']:
    print(f"Base Prefill: {sum(timing_data['base_model_prefill_times'])*1000:.2f} ms")
if timing_data['teacher_embedding_times']:
    print(f"Teacher Embedding: {sum(timing_data['teacher_embedding_times'])*1000:.2f} ms")
if timing_data['teacher_model_prefill_times']:
    print(f"Teacher Prefill: {sum(timing_data['teacher_model_prefill_times'])*1000:.2f} ms")
if timing_data['projector_times']:
    print(f"Projector: {sum(timing_data['projector_times'])*1000:.2f} ms (calls: {len(timing_data['projector_times'])})")
if timing_data['decode_step_times']:
    avg_decode = sum(timing_data['decode_step_times'])/len(timing_data['decode_step_times'])*1000
    total_decode = sum(timing_data['decode_step_times'])*1000
    print(f"Decode: {total_decode:.2f} ms total, {avg_decode:.2f} ms/token avg ({len(timing_data['decode_step_times'])} tokens)")
print("="*80)
