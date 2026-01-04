"""
Dataset timing benchmark script for C2C model
Loads prompts from a dataset and saves timing data to JSONL

Usage:
    python dataset_timing_benchmark.py --dataset gsm8k --split test --num_samples 100
    python dataset_timing_benchmark.py --dataset_file my_prompts.jsonl
"""

import torch
import logging
import time
import json
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from script.playground.inference_example import load_rosetta_model
from tqdm import tqdm

# 配置logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_prompts_from_dataset(dataset_name, split="test", num_samples=None):
    """Load prompts from HuggingFace dataset"""
    from datasets import load_dataset
    
    print(f"Loading dataset: {dataset_name} ({split})")
    dataset = load_dataset(dataset_name, split=split)
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    prompts = []
    for item in dataset:
        # Adapt based on dataset structure
        if 'question' in item:
            prompts.append(item['question'])
        elif 'text' in item:
            prompts.append(item['text'])
        elif 'prompt' in item:
            prompts.append(item['prompt'])
        else:
            # Try first text field
            text_field = next((k for k in item.keys() if isinstance(item[k], str)), None)
            if text_field:
                prompts.append(item[text_field])
    
    return prompts

def load_prompts_from_file(file_path):
    """Load prompts from JSONL file"""
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'prompt' in data:
                prompts.append(data['prompt'])
            elif 'text' in data:
                prompts.append(data['text'])
            elif 'question' in data:
                prompts.append(data['question'])
    return prompts

def setup_timing_hooks(model):
    """Setup timing hooks on the model"""
    timing_data = {
        'base_embedding_times': [],
        'base_model_prefill_times': [],
        'teacher_embedding_times': [],
        'teacher_model_prefill_times': [],
        'projector_times': [],
        'decode_step_times': []
    }
    
    original_forward = model.forward
    original_generate = model.generate
    
    def timed_forward(self, *args, **kwargs):
        past_key_values = kwargs.get('past_key_values', None)
        is_prefill = past_key_values is None or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0)
        
        if is_prefill:
            torch.cuda.synchronize()
            result = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
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
    
    model.forward = lambda *args, **kwargs: timed_forward(model, *args, **kwargs)
    model.generate = lambda *args, **kwargs: timed_generate(model, *args, **kwargs)
    
    # Base model hooks
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
    
    # Projector hooks
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
    
    return timing_data

def main():
    parser = argparse.ArgumentParser(description="Benchmark C2C model timing on a dataset")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name (e.g., gsm8k)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--dataset_file", type=str, help="Path to JSONL file with prompts")
    parser.add_argument("--num_samples", type=int, help="Number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--output", type=str, default="timing_results.jsonl", help="Output file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-4B-Base")
    
    args = parser.parse_args()
    
    # Load prompts
    if args.dataset_file:
        prompts = load_prompts_from_file(args.dataset_file)
    elif args.dataset:
        prompts = load_prompts_from_dataset(args.dataset, args.split, args.num_samples)
    else:
        raise ValueError("Must provide either --dataset or --dataset_file")
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Load model
    checkpoint_dir = snapshot_download(
        repo_id="nics-efc/C2C_Fuser",
        allow_patterns=["qwen3_0.6b+qwen3_4b_base_Fuser/*"],
    )
    
    model_config = {
        "rosetta_config": {
            "base_model": args.base_model,
            "teacher_model": args.teacher_model,
            "checkpoints_dir": f"{checkpoint_dir}/qwen3_0.6b+qwen3_4b_base_Fuser/final",
        }
    }
    
    print("Loading model...")
    rosetta_model, tokenizer = load_rosetta_model(model_config, eval_config={}, device=torch.device("cuda"))
    device = rosetta_model.device
    
    # Setup timing
    timing_data = setup_timing_hooks(rosetta_model)
    
    # Clear output file
    open(args.output, "w").close()
    
    print(f"\nProcessing {len(prompts)} prompts...")
    print(f"Results will be saved to: {args.output}")
    print("=" * 80)
    
    for idx, prompt_text in enumerate(tqdm(prompts, desc="Processing"), 1):
        # Reset timing
        for key in timing_data:
            timing_data[key] = []
        
        # Prepare input
        prompt = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(inputs['input_ids'].shape[1] - 1, 1).unsqueeze(0).to(device)
        label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
        kv_cache_index = [instruction_index, label_index]
        
        # Generate
        try:
            with torch.no_grad():
                sampling_params = {
                    'do_sample': False,
                    'max_new_tokens': args.max_new_tokens
                }
                outputs = rosetta_model.generate(**inputs, kv_cache_index=kv_cache_index, **sampling_params)
                output_text = tokenizer.decode(outputs[0, instruction_index.shape[1] + 1:], skip_special_tokens=True)
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
        
        # Create record
        timing_record = {
            "id": idx,
            "prompt": prompt_text,
            "response": output_text,
            "model_config": {
                "base_model": args.base_model,
                "teacher_model": args.teacher_model
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
        
        # Save
        with open(args.output, "a") as f:
            f.write(json.dumps(timing_record, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 80)
    print(f"✓ Completed! Results saved to: {args.output}")
    print(f"Total samples processed: {len(prompts)}")

if __name__ == "__main__":
    main()
