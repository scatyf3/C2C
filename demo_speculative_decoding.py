"""
Speculative Decoding Demo

This script demonstrates the usage of speculative decoding with KV fusion.
It compares standard generation vs speculative generation on example prompts.

Usage:
    python demo_speculative_decoding.py
    python demo_speculative_decoding.py --gamma 6 --prompts custom_prompts.txt
"""

import torch
import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from script.playground.inference_example import load_rosetta_model

# Example prompts for testing
DEFAULT_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the main causes of climate change?",
    "Solve: If a train travels 120 km in 2 hours, what is its average speed?",
    "List the planets in our solar system in order from the sun.",
]


def load_prompts_from_file(file_path):
    """Load prompts from a text file (one per line)"""
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def generate_standard(model, tokenizer, prompt_text, max_new_tokens=256, device="cuda"):
    """Generate using standard method"""
    # Prepare input
    prompt = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Create kv_cache_index
    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(
        inputs['input_ids'].shape[1] - 1, 1
    ).unsqueeze(0).to(device)
    label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
    kv_cache_index = [instruction_index, label_index]
    
    # Generate
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        sampling_params = {
            'do_sample': False,
            'max_new_tokens': max_new_tokens
        }
        outputs = model.generate(**inputs, kv_cache_index=kv_cache_index, **sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Decode
    output_text = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    latency = end_time - start_time
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / latency if latency > 0 else 0
    
    return {
        "text": output_text,
        "latency": latency,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_sec,
    }


def generate_speculative(model, tokenizer, prompt_text, max_new_tokens=256, gamma=4, device="cuda"):
    """Generate using speculative decoding"""
    # Prepare input
    prompt = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs, stats = model.speculative_generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            return_stats=True,
        )
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Decode
    output_text = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    latency = end_time - start_time
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / latency if latency > 0 else 0
    
    return {
        "text": output_text,
        "latency": latency,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_sec,
        "acceptance_rate": stats["acceptance_rate"],
        "speedup": stats["speedup"],
        "draft_calls": stats["draft_calls"],
        "target_calls": stats["target_calls"],
    }


def main():
    parser = argparse.ArgumentParser(description="Speculative decoding demo")
    parser.add_argument("--prompts", type=str, help="Path to text file with prompts (one per line)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--gamma", type=int, default=4, help="Speculation window (K)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--fuser_checkpoint", type=str, default="qwen3_0.6b+qwen3_4b_base_Fuser")
    parser.add_argument("--skip_standard", action="store_true", help="Skip standard generation")
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts:
        prompts = load_prompts_from_file(args.prompts)
    else:
        prompts = DEFAULT_PROMPTS
    
    print(f"\n{'='*80}")
    print(f"Speculative Decoding Demo")
    print(f"{'='*80}\n")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Speculation window (γ): {args.gamma}")
    print(f"Base model: {args.base_model}")
    print(f"Teacher model: {args.teacher_model}")
    
    # Load model
    print("\nLoading model...")
    checkpoint_dir = snapshot_download(
        repo_id="nics-efc/C2C_Fuser",
        allow_patterns=[f"{args.fuser_checkpoint}/*"],
    )
    
    model_config = {
        "rosetta_config": {
            "base_model": args.base_model,
            "teacher_model": args.teacher_model,
            "checkpoints_dir": f"{checkpoint_dir}/{args.fuser_checkpoint}/final",
        }
    }
    
    rosetta_model, tokenizer = load_rosetta_model(model_config, eval_config={}, device=torch.device("cuda"))
    device = rosetta_model.device
    print(f"Model loaded on {device}")
    
    # Process each prompt
    for idx, prompt_text in enumerate(prompts, 1):
        print(f"\n{'='*80}")
        print(f"Prompt {idx}/{len(prompts)}")
        print(f"{'='*80}\n")
        print(f"Input: {prompt_text}\n")
        
        # Standard generation
        if not args.skip_standard:
            print("Running standard generation...")
            standard_result = generate_standard(
                model=rosetta_model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            
            print(f"\n[Standard Generation]")
            print(f"Output: {standard_result['text'][:200]}..." if len(standard_result['text']) > 200 else f"Output: {standard_result['text']}")
            print(f"Latency: {standard_result['latency']:.3f}s")
            print(f"Tokens: {standard_result['num_tokens']}")
            print(f"Throughput: {standard_result['tokens_per_second']:.2f} tokens/s")
        
        # Speculative generation
        print("\nRunning speculative generation...")
        speculative_result = generate_speculative(
            model=rosetta_model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
            gamma=args.gamma,
            device=device,
        )
        
        print(f"\n[Speculative Generation (γ={args.gamma})]")
        print(f"Output: {speculative_result['text'][:200]}..." if len(speculative_result['text']) > 200 else f"Output: {speculative_result['text']}")
        print(f"Latency: {speculative_result['latency']:.3f}s")
        print(f"Tokens: {speculative_result['num_tokens']}")
        print(f"Throughput: {speculative_result['tokens_per_second']:.2f} tokens/s")
        print(f"Acceptance rate: {speculative_result['acceptance_rate']:.2%}")
        print(f"Speedup: {speculative_result['speedup']:.2f}x")
        print(f"Draft calls: {speculative_result['draft_calls']}")
        print(f"Target calls: {speculative_result['target_calls']}")
        
        # Compare
        if not args.skip_standard:
            latency_speedup = standard_result['latency'] / speculative_result['latency']
            throughput_speedup = speculative_result['tokens_per_second'] / standard_result['tokens_per_second']
            
            print(f"\n[Comparison]")
            print(f"Latency improvement: {latency_speedup:.2f}x faster")
            print(f"Throughput improvement: {throughput_speedup:.2f}x higher")
    
    print(f"\n{'='*80}")
    print(f"Demo completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
