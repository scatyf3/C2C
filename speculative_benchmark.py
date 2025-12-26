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


def benchmark_teacher_generation(teacher_model, tokenizer, prompts, max_new_tokens=256, device="cuda"):
    """Benchmark teacher model direct generation (baseline)"""
    results = {
        "latencies": [],
        "tokens_per_second": [],
        "total_tokens": 0,
    }
    
    for prompt_text in tqdm(prompts, desc="Teacher Model (Baseline)"):
        # Prepare input
        prompt = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate directly with teacher model
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            sampling_params = {
                'do_sample': False,
                'max_new_tokens': max_new_tokens
            }
            outputs = teacher_model.generate(**inputs, **sampling_params)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = num_tokens / latency if latency > 0 else 0
        
        results["latencies"].append(latency)
        results["tokens_per_second"].append(tokens_per_sec)
        results["total_tokens"] += num_tokens
    
    return results


def benchmark_speculative_generation(model, tokenizer, prompts, max_new_tokens=256, gamma=4, device="cuda", prefill_fusion=True, decode_fusion=True):
    """Benchmark speculative decoding generation"""
    results = {
        "latencies": [],
        "tokens_per_second": [],
        "total_tokens": 0,
        "acceptance_rates": [],
        "speedups": [],
        "average_accepted_lengths": [],
    }
    
    desc = f"Speculative (γ={gamma}, prefill={'on' if prefill_fusion else 'off'}, decode={'on' if decode_fusion else 'off'})"
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
                prefill_fusion=prefill_fusion,
                decode_fusion=decode_fusion,
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
    
    # Get teacher model for baseline
    teacher_model = rosetta_model.model_list[1]  # Teacher model is at index 1
    
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
        "baseline_teacher": None,
        "speculative_prefill_on_decode_on": None,
        "speculative_prefill_on_decode_off": None,
        "speculative_prefill_off_decode_on": None,
        "speculative_prefill_off_decode_off": None,
    }
    
    # Baseline: Teacher model direct generation
    if not args.skip_standard:
        print(f"\n{'='*80}")
        print("Running Baseline: Teacher Model Direct Generation")
        print(f"{'='*80}\n")
        
        baseline_results = benchmark_teacher_generation(
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        
        baseline_stats = compute_statistics(baseline_results)
        benchmark_results["baseline_teacher"] = {
            "raw_results": baseline_results,
            "statistics": baseline_stats,
        }
        
        print("\nBaseline (Teacher Model) Results:")
        print(f"  Mean latency: {baseline_stats['mean_latency']:.3f}s ± {baseline_stats['std_latency']:.3f}s")
        print(f"  Median latency: {baseline_stats['median_latency']:.3f}s")
        print(f"  Mean throughput: {baseline_stats['mean_tokens_per_second']:.2f} tokens/s")
        print(f"  Total tokens: {baseline_stats['total_tokens']}")
    
    # Speculative generation: Test 4 configurations
    if not args.skip_speculative:
        configs = [
            ("prefill_on_decode_on", True, True, "Prefill=ON, Decode=ON (Full C2C)"),
            ("prefill_on_decode_off", True, False, "Prefill=ON, Decode=OFF"),
            ("prefill_off_decode_on", False, True, "Prefill=OFF, Decode=ON"),
            ("prefill_off_decode_off", False, False, "Prefill=OFF, Decode=OFF (No Fusion)"),
        ]
        
        for config_name, prefill_fusion, decode_fusion, description in configs:
            print(f"\n{'='*80}")
            print(f"Running Speculative Decoding: {description}")
            print(f"{'='*80}\n")
            
            spec_results = benchmark_speculative_generation(
                model=rosetta_model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                device=device,
                prefill_fusion=prefill_fusion,
                decode_fusion=decode_fusion,
            )
            
            spec_stats = compute_statistics(spec_results)
            benchmark_results[f"speculative_{config_name}"] = {
                "raw_results": spec_results,
                "statistics": spec_stats,
            }
            
            print(f"\nSpeculative Results ({description}):")
            print(f"  Mean latency: {spec_stats['mean_latency']:.3f}s ± {spec_stats['std_latency']:.3f}s")
            print(f"  Median latency: {spec_stats['median_latency']:.3f}s")
            print(f"  Mean throughput: {spec_stats['mean_tokens_per_second']:.2f} tokens/s")
            print(f"  Mean acceptance rate: {spec_stats['mean_acceptance_rate']:.2%} ± {spec_stats['std_acceptance_rate']:.2%}")
            print(f"  Mean accepted length: {spec_stats['mean_accepted_length']:.2f} ± {spec_stats['std_accepted_length']:.2f}")
            print(f"  Mean speedup: {spec_stats['mean_speedup']:.2f}x ± {spec_stats['std_speedup']:.2f}x")
            print(f"  Total tokens: {spec_stats['total_tokens']}")

    # Compare all results
    if not args.skip_speculative:
        print(f"\n{'='*80}")
        print("Summary Comparison: All Configurations")
        print(f"{'='*80}\n")
        
        config_labels = [
            ("baseline_teacher", "Baseline (Teacher)"),
            ("speculative_prefill_on_decode_on", "Spec: Prefill=ON, Decode=ON"),
            ("speculative_prefill_on_decode_off", "Spec: Prefill=ON, Decode=OFF"),
            ("speculative_prefill_off_decode_on", "Spec: Prefill=OFF, Decode=ON"),
            ("speculative_prefill_off_decode_off", "Spec: Prefill=OFF, Decode=OFF"),
        ]
        
        print(f"{'Configuration':<40} {'Latency (s)':<15} {'Throughput':<15} {'Accept Rate':<15} {'Speedup':<10}")
        print("-" * 95)
        
        for config_key, label in config_labels:
            if benchmark_results.get(config_key):
                stats = benchmark_results[config_key]["statistics"]
                latency = f"{stats['mean_latency']:.3f}"
                throughput = f"{stats['mean_tokens_per_second']:.2f} tok/s"
                
                if 'mean_acceptance_rate' in stats:
                    accept_rate = f"{stats['mean_acceptance_rate']:.2%}"
                    speedup = f"{stats['mean_speedup']:.2f}x"
                else:
                    accept_rate = "N/A"
                    speedup = "N/A"
                
                print(f"{label:<40} {latency:<15} {throughput:<15} {accept_rate:<15} {speedup:<10}")
    
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
