"""
Batch timing benchmark script for C2C model
Processes multiple prompts and saves timing data to JSONL
C2C agent模式计时
"""

import torch
import logging
import time
import json
from huggingface_hub import snapshot_download
from script.playground.inference_example import load_rosetta_model

# 配置logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
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

# Timing data structure (will be reset for each sample)
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
    past_key_values = kwargs.get('past_key_values', None)
    is_prefill = past_key_values is None or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0)
    
    if is_prefill:
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        result = original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        
        if hasattr(self, '_is_prefill_phase'):
            self._is_prefill_phase = False
    else:
        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        result = original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        decode_end = time.perf_counter()
        timing_data['decode_step_times'].append(decode_end - decode_start)
    
    return result

def timed_generate(self, *args, **kwargs):
    self._is_prefill_phase = True
    return original_generate(*args, **kwargs)

rosetta_model.forward = lambda *args, **kwargs: timed_forward(rosetta_model, *args, **kwargs)
rosetta_model.generate = lambda *args, **kwargs: timed_generate(rosetta_model, *args, **kwargs)

# Add detailed hooks
def add_detailed_timing_hooks(model):
    base_model = model.model_list[model.base_model_idx]
    model._is_prefill_phase = True
    
    original_base_forward = base_model.forward
    def base_forward_with_timing(*args, **kwargs):
        past_kv = kwargs.get('past_key_values', None)
        is_prefill = past_kv is None or (hasattr(past_kv, 'get_seq_length') and past_kv.get_seq_length() == 0)
        
        if is_prefill and model._is_prefill_phase:
            torch.cuda.synchronize()
            embed_start = time.perf_counter()
            
            original_embed = base_model.model.embed_tokens.forward if hasattr(base_model.model, 'embed_tokens') else None
            embed_end_time = [None]
            
            if original_embed is not None:
                def embed_with_timing(*embed_args, **embed_kwargs):
                    result = original_embed(*embed_args, **embed_kwargs)
                    torch.cuda.synchronize()
                    embed_end_time[0] = time.perf_counter()
                    return result
                base_model.model.embed_tokens.forward = embed_with_timing
            
            result = original_base_forward(*args, **kwargs)
            torch.cuda.synchronize()
            compute_end = time.perf_counter()
            
            if original_embed is not None:
                base_model.model.embed_tokens.forward = original_embed
                if embed_end_time[0] is not None:
                    timing_data['base_embedding_times'].append(embed_end_time[0] - embed_start)
                    timing_data['base_model_prefill_times'].append(compute_end - embed_end_time[0])
        else:
            result = original_base_forward(*args, **kwargs)
        
        return result
    
    base_model.forward = base_forward_with_timing
    
    # Teacher model hooks
    if len(model.model_list) > 1:
        teacher_model = model.model_list[1]
        original_embed_forward = teacher_model.model.embed_tokens.forward if hasattr(teacher_model.model, 'embed_tokens') else None
        
        original_teacher_forward = teacher_model.forward
        def teacher_forward_with_timing(*args, **kwargs):
            torch.cuda.synchronize()
            embed_start = time.perf_counter()
            
            if original_embed_forward is not None:
                embed_end_time = [None]
                def embed_with_timing(*embed_args, **embed_kwargs):
                    result = original_embed_forward(*embed_args, **embed_kwargs)
                    torch.cuda.synchronize()
                    embed_end_time[0] = time.perf_counter()
                    return result
                teacher_model.model.embed_tokens.forward = embed_with_timing
            
            result = original_teacher_forward(*args, **kwargs)
            torch.cuda.synchronize()
            compute_end = time.perf_counter()
            
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

# Test prompts
test_prompts = [
    "Say hello in one short sentence.",
    "What is the capital of France?",
    "Explain machine learning in simple terms.",
    "Write a haiku about coding.",
    "What is 25 * 17?",
]

output_file = "batch_timing_results.jsonl"
print(f"\nProcessing {len(test_prompts)} prompts...")
print(f"Results will be saved to: {output_file}")
print("=" * 80)

# Clear output file
open(output_file, "w").close()

for idx, prompt_text in enumerate(test_prompts, 1):
    print(f"\n[{idx}/{len(test_prompts)}] Processing: {prompt_text[:50]}...")
    
    # Reset timing data for this sample
    timing_data = {
        'base_embedding_times': [],
        'base_model_prefill_times': [],
        'teacher_embedding_times': [],
        'teacher_model_prefill_times': [],
        'projector_times': [],
        'decode_step_times': []
    }
    
    # Prepare input
    prompt = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(inputs['input_ids'].shape[1] - 1, 1).unsqueeze(0).to(device)
    label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
    kv_cache_index = [instruction_index, label_index]
    
    # Generate
    with torch.no_grad():
        sampling_params = {
            'do_sample': False,
            'max_new_tokens': 128
        }
        outputs = rosetta_model.generate(**inputs, kv_cache_index=kv_cache_index, **sampling_params)
        output_text = tokenizer.decode(outputs[0, instruction_index.shape[1] + 1:], skip_special_tokens=True)
    
    print(f"Response: {output_text[:80]}...")
    
    # Create timing record
    timing_record = {
        "id": idx,
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
            "num_generated_tokens": len(timing_data['decode_step_times']),
            "tokens_per_second": len(timing_data['decode_step_times']) / sum(timing_data['decode_step_times']) if timing_data['decode_step_times'] else 0
        },
        "decode_step_times_ms": [t * 1000 for t in timing_data['decode_step_times']]
    }
    
    # Append to JSONL
    with open(output_file, "a") as f:
        f.write(json.dumps(timing_record, ensure_ascii=False) + "\n")
    
    print(f"  Tokens: {timing_record['timing']['num_generated_tokens']}, "
          f"Speed: {timing_record['timing']['tokens_per_second']:.2f} tok/s, "
          f"Avg decode: {timing_record['timing']['decode_avg_ms']:.2f}ms")

print("\n" + "=" * 80)
print(f"✓ Completed! Results saved to: {output_file}")
print(f"Total samples processed: {len(test_prompts)}")
print("\nTo view results:")
print(f"  cat {output_file} | jq .")
print(f"  cat {output_file} | jq -r '[.prompt, .response, .timing.decode_avg_ms] | @tsv'")
