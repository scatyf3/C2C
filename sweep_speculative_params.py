"""
参数扫描脚本 - 测试不同配置下 Speculative Decoding 的性能

扫描参数：
1. max_new_tokens: 不同的生成长度
2. dataset: 不同的数据集
3. target_model: 不同的目标模型

用法:
    python sweep_speculative_params.py
    python sweep_speculative_params.py --quick  # 快速测试（少量样本）
"""

import torch
import argparse
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_test_prompts(dataset_name="gsm8k", num_samples=10):
    """加载测试提示词"""
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [example["question"] for example in dataset.select(range(num_samples))]
    
    elif dataset_name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test")
        prompts = [
            f"Question: {example['question']}\nChoices: A) {example['choices'][0]} B) {example['choices'][1]} C) {example['choices'][2]} D) {example['choices'][3]}\nAnswer:"
            for example in dataset.select(range(num_samples))
        ]
    
    elif dataset_name == "arc":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        prompts = [
            f"Question: {example['question']}\nChoices: {', '.join([f'{choice}' for choice in example['choices']['text']])}\nAnswer:"
            for example in dataset.select(range(num_samples))
        ]
    
    elif dataset_name == "math":
        dataset = load_dataset("hendrycks/competition_math", split="test")
        prompts = [example["problem"] for example in dataset.select(range(num_samples))]
    
    elif dataset_name == "simple":
        prompts = [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms.",
            "Write a short story about a robot learning to paint.",
            "Calculate the sum of 123 and 456.",
            "What are the benefits of exercise?",
        ][:num_samples]
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return prompts


def warm_up(model, tokenizer, draft_model=None, device="cuda"):
    """预热模型"""
    warm_up_prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": warm_up_prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    if draft_model is not None:
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                assistant_model=draft_model,
            )
    
    torch.cuda.synchronize()


def generate_standard(model, tokenizer, prompt_text, max_new_tokens=256, device="cuda"):
    """标准生成"""
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency = end_time - start_time
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / latency if latency > 0 else 0
    
    return {
        "latency": latency,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_sec,
    }


def generate_speculative(draft_model, target_model, tokenizer, prompt_text, 
                         max_new_tokens=256, device="cuda"):
    """Speculative Decoding 生成"""
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            assistant_model=draft_model,
        )
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency = end_time - start_time
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / latency if latency > 0 else 0
    
    return {
        "latency": latency,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_sec,
    }


def run_single_experiment(draft_model, target_model, tokenizer, prompts, 
                          max_new_tokens, device="cuda"):
    """运行单个实验配置"""
    standard_results = []
    speculative_results = []
    
    for prompt in prompts:
        # 标准生成
        std_result = generate_standard(
            model=target_model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        standard_results.append(std_result)
        
        # Speculative Decoding
        spec_result = generate_speculative(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        speculative_results.append(spec_result)
    
    # 计算平均值
    avg_std_latency = np.mean([r["latency"] for r in standard_results])
    avg_std_throughput = np.mean([r["tokens_per_second"] for r in standard_results])
    avg_std_tokens = np.mean([r["num_tokens"] for r in standard_results])
    
    avg_spec_latency = np.mean([r["latency"] for r in speculative_results])
    avg_spec_throughput = np.mean([r["tokens_per_second"] for r in speculative_results])
    avg_spec_tokens = np.mean([r["num_tokens"] for r in speculative_results])
    
    speedup_latency = avg_std_latency / avg_spec_latency if avg_spec_latency > 0 else 0
    speedup_throughput = avg_spec_throughput / avg_std_throughput if avg_std_throughput > 0 else 0
    
    return {
        "standard": {
            "avg_latency": avg_std_latency,
            "avg_throughput": avg_std_throughput,
            "avg_tokens": avg_std_tokens,
        },
        "speculative": {
            "avg_latency": avg_spec_latency,
            "avg_throughput": avg_spec_throughput,
            "avg_tokens": avg_spec_tokens,
        },
        "speedup": {
            "latency": speedup_latency,
            "throughput": speedup_throughput,
        }
    }


def plot_results(sweep_results, output_dir="output/sweep_results"):
    """生成可视化图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    data_records = []
    for result in sweep_results:
        data_records.append({
            "target_model": result["config"]["target_model"],
            "dataset": result["config"]["dataset"],
            "max_new_tokens": result["config"]["max_new_tokens"],
            "latency_speedup": result["results"]["speedup"]["latency"],
            "throughput_speedup": result["results"]["speedup"]["throughput"],
            "std_latency": result["results"]["standard"]["avg_latency"],
            "spec_latency": result["results"]["speculative"]["avg_latency"],
            "std_throughput": result["results"]["standard"]["avg_throughput"],
            "spec_throughput": result["results"]["speculative"]["avg_throughput"],
        })
    
    df = pd.DataFrame(data_records)
    
    # 1. Max tokens vs Speedup (按模型分组)
    plt.figure(figsize=(12, 6))
    for model in df["target_model"].unique():
        model_data = df[df["target_model"] == model]
        model_name = model.split("/")[-1]  # 简化名称
        plt.plot(model_data["max_new_tokens"], model_data["latency_speedup"], 
                marker='o', label=f"{model_name} (Latency)")
        plt.plot(model_data["max_new_tokens"], model_data["throughput_speedup"], 
                marker='s', linestyle='--', label=f"{model_name} (Throughput)")
    
    plt.xlabel("Max New Tokens")
    plt.ylabel("Speedup (x)")
    plt.title("Speculative Decoding Speedup vs Max Tokens")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "speedup_vs_max_tokens.png", dpi=150)
    plt.close()
    
    # 2. Dataset comparison (热力图)
    if len(df["dataset"].unique()) > 1:
        pivot_latency = df.pivot_table(
            values="latency_speedup", 
            index="dataset", 
            columns="max_new_tokens"
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_latency, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Latency Speedup (x)'})
        plt.title("Latency Speedup Across Datasets and Max Tokens")
        plt.xlabel("Max New Tokens")
        plt.ylabel("Dataset")
        plt.tight_layout()
        plt.savefig(output_path / "heatmap_dataset_speedup.png", dpi=150)
        plt.close()
    
    # 3. Model comparison bar chart
    if len(df["target_model"].unique()) > 1:
        model_avg = df.groupby("target_model")[["latency_speedup", "throughput_speedup"]].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(model_avg))
        width = 0.35
        
        ax.bar(x - width/2, model_avg["latency_speedup"], width, label="Latency Speedup")
        ax.bar(x + width/2, model_avg["throughput_speedup"], width, label="Throughput Speedup")
        
        ax.set_xlabel("Target Model")
        ax.set_ylabel("Average Speedup (x)")
        ax.set_title("Average Speedup Comparison Across Models")
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("/")[-1] for m in model_avg.index])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path / "model_comparison.png", dpi=150)
        plt.close()
    
    # 4. Throughput comparison (standard vs speculative)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 按 max_tokens 分组
    for max_tokens in sorted(df["max_new_tokens"].unique()):
        data_subset = df[df["max_new_tokens"] == max_tokens]
        axes[0].plot(range(len(data_subset)), data_subset["std_throughput"], 
                    marker='o', label=f"Standard ({max_tokens} tokens)")
        axes[1].plot(range(len(data_subset)), data_subset["spec_throughput"], 
                    marker='s', label=f"Speculative ({max_tokens} tokens)")
    
    axes[0].set_xlabel("Experiment Index")
    axes[0].set_ylabel("Throughput (tokens/s)")
    axes[0].set_title("Standard Generation Throughput")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Experiment Index")
    axes[1].set_ylabel("Throughput (tokens/s)")
    axes[1].set_title("Speculative Decoding Throughput")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "throughput_comparison.png", dpi=150)
    plt.close()
    
    print(f"\n可视化图表已保存到: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding 参数扫描")
    parser.add_argument("--draft_model", type=str, default="Qwen/Qwen3-0.6B",
                        help="草稿模型路径")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式（少量样本和参数组合）")
    parser.add_argument("--output_dir", type=str, default="output/sweep_results",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 定义参数扫描范围
    if args.quick:
        print("运行快速测试模式...")
        target_models = ["Qwen/Qwen3-8B"]
        datasets = ["simple"]
        max_tokens_list = [128, 256]
        num_samples = 3
    else:
        print("运行完整参数扫描...")
        target_models = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
        datasets = ["simple", "gsm8k", "arc"]
        max_tokens_list = [128, 256, 512, 1024]
        num_samples = 10
    
    print(f"\n{'='*80}")
    print(f"参数扫描配置")
    print(f"{'='*80}\n")
    print(f"草稿模型: {args.draft_model}")
    print(f"目标模型: {', '.join(target_models)}")
    print(f"数据集: {', '.join(datasets)}")
    print(f"Max tokens: {max_tokens_list}")
    print(f"每个配置样本数: {num_samples}")
    print(f"总实验数: {len(target_models) * len(datasets) * len(max_tokens_list)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 加载草稿模型（固定）
    print(f"\n加载草稿模型: {args.draft_model}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("草稿模型加载完成！")
    
    # 存储所有实验结果
    sweep_results = []
    current_target_model = None
    target_model = None
    tokenizer = None
    
    # 计算总实验数
    total_experiments = len(target_models) * len(datasets) * len(max_tokens_list)
    experiment_count = 0
    
    # 遍历所有参数组合
    for target_model_name in target_models:
        # 只在需要时加载/切换目标模型
        if target_model_name != current_target_model:
            print(f"\n{'='*80}")
            print(f"加载目标模型: {target_model_name}")
            print(f"{'='*80}")
            
            # 清理之前的模型
            if target_model is not None:
                del target_model
                torch.cuda.empty_cache()
            
            target_model = AutoModelForCausalLM.from_pretrained(
                target_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(target_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 预热模型
            print("预热模型...")
            warm_up(target_model, tokenizer, draft_model, device)
            current_target_model = target_model_name
        
        for dataset_name in datasets:
            # 加载数据集
            print(f"\n加载数据集: {dataset_name}")
            prompts = load_test_prompts(dataset_name, num_samples)
            
            for max_tokens in max_tokens_list:
                experiment_count += 1
                print(f"\n{'='*80}")
                print(f"实验 {experiment_count}/{total_experiments}")
                print(f"{'='*80}")
                print(f"目标模型: {target_model_name}")
                print(f"数据集: {dataset_name}")
                print(f"Max tokens: {max_tokens}")
                print(f"样本数: {len(prompts)}")
                
                # 运行实验
                results = run_single_experiment(
                    draft_model=draft_model,
                    target_model=target_model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=max_tokens,
                    device=device,
                )
                
                # 保存结果
                sweep_results.append({
                    "config": {
                        "target_model": target_model_name,
                        "draft_model": args.draft_model,
                        "dataset": dataset_name,
                        "max_new_tokens": max_tokens,
                        "num_samples": len(prompts),
                    },
                    "results": results,
                })
                
                # 打印结果
                print(f"\n结果:")
                print(f"  标准生成 - 平均延迟: {results['standard']['avg_latency']:.3f}s, "
                      f"吞吐量: {results['standard']['avg_throughput']:.2f} tokens/s")
                print(f"  Speculative - 平均延迟: {results['speculative']['avg_latency']:.3f}s, "
                      f"吞吐量: {results['speculative']['avg_throughput']:.2f} tokens/s")
                print(f"  加速比: 延迟 {results['speedup']['latency']:.2f}x, "
                      f"吞吐量 {results['speedup']['throughput']:.2f}x")
    
    # 保存所有结果
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"sweep_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "draft_model": args.draft_model,
                "target_models": target_models,
                "datasets": datasets,
                "max_tokens_list": max_tokens_list,
                "num_samples": num_samples,
            },
            "results": sweep_results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"所有实验完成！")
    print(f"{'='*80}")
    print(f"\n结果已保存到: {results_file}")
    
    # 生成可视化
    print("\n生成可视化图表...")
    df = plot_results(sweep_results, args.output_dir)
    
    # 保存 CSV
    csv_file = output_path / f"sweep_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV 数据已保存到: {csv_file}")
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"总结")
    print(f"{'='*80}\n")
    print(f"总实验数: {len(sweep_results)}")
    print(f"\n最佳配置（按延迟加速比）:")
    best_config = max(sweep_results, key=lambda x: x["results"]["speedup"]["latency"])
    print(f"  目标模型: {best_config['config']['target_model']}")
    print(f"  数据集: {best_config['config']['dataset']}")
    print(f"  Max tokens: {best_config['config']['max_new_tokens']}")
    print(f"  延迟加速比: {best_config['results']['speedup']['latency']:.2f}x")
    print(f"  吞吐量加速比: {best_config['results']['speedup']['throughput']:.2f}x")


if __name__ == "__main__":
    main()
