"""
Speculative Decoding Benchmark with KV Fusion

This script benchmarks the performance of speculative decoding compared to
standard generation, measuring:
- Token acceptance rate
- Generation speedup
- Latency improvements
- Memory usage

Usage:
    python speculative_benchmark.py --dataset gsm8k --num_samples 100 --gamma 4
    python speculative_benchmark.py --dataset mmlu-redux --subject high_school_mathematics --gamma 6
"""

import argparse
import json
import time
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download
from script.playground.inference_example import load_rosetta_model
import numpy as np

# Suppress warnings
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress rosetta internal logging
logging.getLogger('rosetta').setLevel(logging.CRITICAL)
logging.getLogger('rosetta.model.wrapper').setLevel(logging.CRITICAL)

# Dataset configurations (subset from comprehensive benchmark)
DATASET_CONFIGS = {
    "gsm8k": {
        "dataset_name": "openai/gsm8k",
        "split": "test",
        "config": "main",
    },
    "mmlu-redux": {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "split": "test",
        "subjects": ["high_school_mathematics", "high_school_physics", "college_computer_science"],
    },
}


def load_dataset_samples(dataset_name, subject=None, num_samples=None):
    """Load samples from specified dataset"""
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    hf_dataset_name = config["dataset_name"]
    split = config["split"]
    
    print(f"Loading dataset: {hf_dataset_name}")
    
    # Load dataset based on type
    if dataset_name == "mmlu-redux":
        if subject:
            dataset = load_dataset(hf_dataset_name, subject, split=split)
        else:
            subject = config["subjects"][0]
            dataset = load_dataset(hf_dataset_name, subject, split=split)
    elif dataset_name == "gsm8k":
        dataset_config = config.get("config")
        dataset = load_dataset(hf_dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(hf_dataset_name, split=split)
    
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    return dataset, subject


def format_prompt(example, dataset_name):
    """Format example into a prompt based on dataset type"""
    if dataset_name == "mmlu-redux":
        question = example["question"]
        choices = example["choices"]
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        return f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
    
    elif dataset_name == "gsm8k":
        return example["question"]
    
    # Default
    for field in ["question", "prompt", "text", "input"]:
        if field in example:
            return example[field]
    
    return str(example)


def benchmark_standard_generation(model, tokenizer, prompts, max_new_tokens=256, device="cuda"):
    """Benchmark standard generation (no speculation)"""
    results = {
        "latencies": [],
        "tokens_per_second": [],
        "total_tokens": 0,
    }
    
    for prompt_text in tqdm(prompts, desc="Standard Generation"):
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
        
        latency = end_time - start_time
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = num_tokens / latency if latency > 0 else 0
        
        results["latencies"].append(latency)
        results["tokens_per_second"].append(tokens_per_sec)
        results["total_tokens"] += num_tokens
    
    return results


def benchmark_speculative_generation(model, tokenizer, prompts, max_new_tokens=256, gamma=4, device="cuda", fuse_kv=True):
    """Benchmark speculative decoding generation"""
    results = {
        "latencies": [],
        "tokens_per_second": [],
        "total_tokens": 0,
        "acceptance_rates": [],
        "speedups": [],
        "average_accepted_lengths": [],
    }
    
    desc = f"Speculative (γ={gamma}, fuse_kv={'on' if fuse_kv else 'off'})"
    for prompt_text in tqdm(prompts, desc=desc):
        # Prepare input
        prompt = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate with speculative decoding
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs, stats = model.speculative_generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_new_tokens,
                gamma=gamma,
                return_stats=True,
                fuse_kv=fuse_kv,
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = num_tokens / latency if latency > 0 else 0
        
        results["latencies"].append(latency)
        results["tokens_per_second"].append(tokens_per_sec)
        results["total_tokens"] += num_tokens
        results["acceptance_rates"].append(stats["acceptance_rate"])
        results["speedups"].append(stats["speedup"])
        results["average_accepted_lengths"].append(stats.get("average_accepted_length", 0))
    return results


def compute_statistics(results):
    """Compute summary statistics"""
    stats = {
        "mean_latency": np.mean(results["latencies"]),
        "std_latency": np.std(results["latencies"]),
        "median_latency": np.median(results["latencies"]),
        "mean_tokens_per_second": np.mean(results["tokens_per_second"]),
        "std_tokens_per_second": np.std(results["tokens_per_second"]),
        "total_tokens": results["total_tokens"],
    }
    
    if "acceptance_rates" in results:
        stats["mean_acceptance_rate"] = np.mean(results["acceptance_rates"])
        stats["std_acceptance_rate"] = np.std(results["acceptance_rates"])
        stats["mean_speedup"] = np.mean(results["speedups"])
        stats["std_speedup"] = np.std(results["speedups"])
    if "average_accepted_lengths" in results:
        stats["mean_accepted_length"] = np.mean(results["average_accepted_lengths"])
        stats["std_accepted_length"] = np.std(results["average_accepted_lengths"])
    return stats


def main():
    parser = argparse.ArgumentParser(description="Speculative decoding benchmark")
    parser.add_argument("--dataset", type=str, default="gsm8k", 
                       choices=list(DATASET_CONFIGS.keys()),
                       help="Dataset to benchmark")
    parser.add_argument("--subject", type=str, help="Specific subject (for MMLU)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--gamma", type=int, default=4, help="Speculation window (K)")
    parser.add_argument("--output_dir", type=str, default="speculative_results", help="Output directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--fuser_checkpoint", type=str, default="qwen3_0.6b+qwen3_4b_base_Fuser")
    parser.add_argument("--skip_standard", action="store_true", help="Skip standard generation benchmark")
    parser.add_argument("--skip_speculative", action="store_true", help="Skip speculative generation benchmark")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    print(f"\n{'='*80}")
    print(f"Loading dataset: {args.dataset}")
    print(f"{'='*80}\n")
    
    dataset, subject = load_dataset_samples(args.dataset, args.subject, args.num_samples)
    print(f"Loaded {len(dataset)} samples")
    
    # Format prompts
    prompts = [format_prompt(example, args.dataset) for example in dataset]
    
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
    print(f"Base model: {args.base_model}")
    print(f"Teacher model: {args.teacher_model}")
    
    # Run benchmarks
    benchmark_results = {
        "config": {
            "dataset": args.dataset,
            "subject": subject,
            "num_samples": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "gamma": args.gamma,
            "base_model": args.base_model,
            "teacher_model": args.teacher_model,
        },
        "standard": None,
        "speculative": None,
    }
    
    # Standard generation
    if not args.skip_standard:
        print(f"\n{'='*80}")
        print("Running Standard Generation Benchmark")
        print(f"{'='*80}\n")
        
        standard_results = benchmark_standard_generation(
            model=rosetta_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        
        standard_stats = compute_statistics(standard_results)
        benchmark_results["standard"] = {
            "raw_results": standard_results,
            "statistics": standard_stats,
        }
        
        print("\nStandard Generation Results:")
        print(f"  Mean latency: {standard_stats['mean_latency']:.3f}s ± {standard_stats['std_latency']:.3f}s")
        print(f"  Median latency: {standard_stats['median_latency']:.3f}s")
        print(f"  Mean throughput: {standard_stats['mean_tokens_per_second']:.2f} tokens/s")
        print(f"  Total tokens: {standard_stats['total_tokens']}")
    
    # Speculative generation (with and without KV fusion)
    if not args.skip_speculative:
        print(f"\n{'='*80}")
        print(f"Running Speculative Generation Benchmark (γ={args.gamma}, fuse_kv=on)")
        print(f"{'='*80}\n")
        speculative_results = benchmark_speculative_generation(
            model=rosetta_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            gamma=args.gamma,
            device=device,
            fuse_kv=True,
        )
        speculative_stats = compute_statistics(speculative_results)
        benchmark_results["speculative_fuse_kv"] = {
            "raw_results": speculative_results,
            "statistics": speculative_stats,
        }
        print("\nSpeculative Generation Results (KV Fused):")
        print(f"  Mean latency: {speculative_stats['mean_latency']:.3f}s ± {speculative_stats['std_latency']:.3f}s")
        print(f"  Median latency: {speculative_stats['median_latency']:.3f}s")
        print(f"  Mean throughput: {speculative_stats['mean_tokens_per_second']:.2f} tokens/s")
        print(f"  Mean acceptance rate: {speculative_stats['mean_acceptance_rate']:.2%} ± {speculative_stats['std_acceptance_rate']:.2%}")
        print(f"  Mean accepted length: {speculative_stats['mean_accepted_length']:.2f} ± {speculative_stats['std_accepted_length']:.2f}")
        print(f"  Mean speedup: {speculative_stats['mean_speedup']:.2f}x ± {speculative_stats['std_speedup']:.2f}x")
        print(f"  Total tokens: {speculative_stats['total_tokens']}")

        print(f"\n{'='*80}")
        print(f"Running Speculative Generation Benchmark (γ={args.gamma}, fuse_kv=off)")
        print(f"{'='*80}\n")
        speculative_results_nokv = benchmark_speculative_generation(
            model=rosetta_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            gamma=args.gamma,
            device=device,
            fuse_kv=False,
        )
        speculative_stats_nokv = compute_statistics(speculative_results_nokv)
        benchmark_results["speculative_no_fuse_kv"] = {
            "raw_results": speculative_results_nokv,
            "statistics": speculative_stats_nokv,
        }
        print("\nSpeculative Generation Results (No KV Fusion):")
        print(f"  Mean latency: {speculative_stats_nokv['mean_latency']:.3f}s ± {speculative_stats_nokv['std_latency']:.3f}s")
        print(f"  Median latency: {speculative_stats_nokv['median_latency']:.3f}s")
        print(f"  Mean throughput: {speculative_stats_nokv['mean_tokens_per_second']:.2f} tokens/s")
        print(f"  Mean acceptance rate: {speculative_stats_nokv['mean_acceptance_rate']:.2%} ± {speculative_stats_nokv['std_acceptance_rate']:.2%}")
        print(f"  Mean accepted length: {speculative_stats_nokv['mean_accepted_length']:.2f} ± {speculative_stats_nokv['std_accepted_length']:.2f}")
        print(f"  Mean speedup: {speculative_stats_nokv['mean_speedup']:.2f}x ± {speculative_stats_nokv['std_speedup']:.2f}x")
        print(f"  Total tokens: {speculative_stats_nokv['total_tokens']}")

    # Compare results
    if benchmark_results.get("speculative_fuse_kv") and benchmark_results.get("speculative_no_fuse_kv"):
        print(f"\n{'='*80}")
        print("Comparison (KV Fusion vs No Fusion)")
        print(f"{'='*80}\n")
        stats_kv = benchmark_results["speculative_fuse_kv"]["statistics"]
        stats_nokv = benchmark_results["speculative_no_fuse_kv"]["statistics"]
        print(f"  Acceptance rate: {stats_kv['mean_acceptance_rate']:.2%} (fuse) vs {stats_nokv['mean_acceptance_rate']:.2%} (no fuse)")
        print(f"  Avg accepted length: {stats_kv['mean_accepted_length']:.2f} (fuse) vs {stats_nokv['mean_accepted_length']:.2f} (no fuse)")
        print(f"  Speedup: {stats_kv['mean_speedup']:.2f}x (fuse) vs {stats_nokv['mean_speedup']:.2f}x (no fuse)")
        print(f"  Throughput: {stats_kv['mean_tokens_per_second']:.2f} (fuse) vs {stats_nokv['mean_tokens_per_second']:.2f} (no fuse)")
        print(f"  Latency: {stats_kv['mean_latency']:.3f}s (fuse) vs {stats_nokv['mean_latency']:.3f}s (no fuse)")
        print(f"  Total tokens: {stats_kv['total_tokens']} (fuse) vs {stats_nokv['total_tokens']} (no fuse)")
    
    # Save results
    subject_name = subject or "main"
    output_file = output_dir / f"{args.dataset}_{subject_name}_speculative_benchmark.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    benchmark_results = convert_to_native(benchmark_results)
    
    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Benchmark completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
