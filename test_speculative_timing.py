"""
测试标准生成 vs Speculative Decoding 的时间对比

使用 Transformers 库内置的 speculative decoding 接口 (assistant_model)
对比有无 speculative decoding 的性能差异。

2026-01-03 use

用法:
    python test_speculative_timing.py
    python test_speculative_timing.py --target_model Qwen/Qwen3-14B --draft_model Qwen/Qwen3-0.6B
    python test_speculative_timing.py --num_samples 5 --max_new_tokens 512
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


def load_test_prompts(dataset_name="gsm8k", num_samples=10):
    """加载测试提示词"""
    print(f"从 {dataset_name} 数据集加载前 {num_samples} 个样本...")
    
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
    
    elif dataset_name == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        prompts = [
            f"{example['ctx']}\nWhat happens next?\nA) {example['endings'][0]}\nB) {example['endings'][1]}\nC) {example['endings'][2]}\nD) {example['endings'][3]}\nAnswer:"
            for example in dataset.select(range(num_samples))
        ]
    
    elif dataset_name == "math":
        dataset = load_dataset("hendrycks/competition_math", split="test")
        prompts = [example["problem"] for example in dataset.select(range(num_samples))]
    
    elif dataset_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompts = [
            f"Complete the following Python function:\n{example['prompt']}"
            for example in dataset.select(range(num_samples))
        ]
    
    elif dataset_name == "simple":
        # 简单的测试提示词
        prompts = [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms.",
            "Write a short story about a robot learning to paint.",
            "Calculate the sum of 123 and 456.",
            "What are the benefits of exercise?",
        ][:num_samples]
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: gsm8k, mmlu, arc, hellaswag, math, humaneval, simple")
    
    return prompts


def warm_up(model, tokenizer, draft_model=None, device="cuda"):
    """预热模型（warm up）以确保 CUDA 内核初始化和缓存预热"""
    print("正在预热模型...")
    
    warm_up_prompt = "Hello, how are you?"
    
    # 检查是否有 chat_template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": warm_up_prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # 直接使用文本（适用于 Llama-2 等没有 chat template 的模型）
        input_text = warm_up_prompt
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Warm up 标准生成
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 如果提供了 draft_model，也预热 speculative decoding
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
    print("模型预热完成！")


def generate_standard(model, tokenizer, prompt_text, max_new_tokens=256, device="cuda"):
    """标准生成（baseline，仅使用目标模型）"""
    # 准备输入
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # 直接使用文本（适用于 Llama-2 等没有 chat template 的模型）
        input_text = prompt_text
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 生成
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
    
    # 统计信息
    latency = end_time - start_time
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / latency if latency > 0 else 0
    
    # 解码输出
    output_text = tokenizer.decode(
        outputs[0, inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return {
        "text": output_text,
        "latency": latency,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_sec,
    }


def generate_speculative(draft_model, target_model, tokenizer, prompt_text, 
                         max_new_tokens=256, device="cuda"):
    """Speculative Decoding 生成（使用 Transformers 内置 API）"""
    # 准备输入
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # 直接使用文本（适用于 Llama-2 等没有 chat template 的模型）
        input_text = prompt_text
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 生成（启用 speculative decoding）
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            assistant_model=draft_model,  # 启用 speculative decoding
        )
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 统计信息
    latency = end_time - start_time
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / latency if latency > 0 else 0
    
    # 解码输出
    output_text = tokenizer.decode(
        outputs[0, inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return {
        "text": output_text,
        "latency": latency,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_sec,
    }


def print_comparison(idx, prompt, standard_result, speculative_result):
    """打印单个样本的对比结果"""
    print(f"\n{'='*80}")
    print(f"样本 {idx}")
    print(f"{'='*80}")
    print(f"\n提示词: {prompt[:100]}..." if len(prompt) > 100 else f"\n提示词: {prompt}")
    
    print(f"\n[标准生成 - Baseline]")
    print(f"  延迟: {standard_result['latency']:.3f}s")
    print(f"  生成token数: {standard_result['num_tokens']}")
    print(f"  吞吐量: {standard_result['tokens_per_second']:.2f} tokens/s")
    print(f"  输出预览: {standard_result['text'][:100]}..." 
          if len(standard_result['text']) > 100 
          else f"  输出预览: {standard_result['text']}")
    
    print(f"\n[Speculative Decoding]")
    print(f"  延迟: {speculative_result['latency']:.3f}s")
    print(f"  生成token数: {speculative_result['num_tokens']}")
    print(f"  吞吐量: {speculative_result['tokens_per_second']:.2f} tokens/s")
    print(f"  输出预览: {speculative_result['text'][:100]}..." 
          if len(speculative_result['text']) > 100 
          else f"  输出预览: {speculative_result['text']}")
    
    # 计算加速比
    latency_speedup = standard_result['latency'] / speculative_result['latency']
    throughput_speedup = speculative_result['tokens_per_second'] / standard_result['tokens_per_second']
    
    print(f"\n[性能提升]")
    print(f"  延迟加速: {latency_speedup:.2f}x")
    print(f"  吞吐量提升: {throughput_speedup:.2f}x")
    print(f"  时间节省: {(1 - 1/latency_speedup)*100:.1f}%")


def save_results(results, output_file="output/speculative_timing_results.json"):
    """保存结果到文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="测试标准生成 vs Speculative Decoding 的时间对比"
    )
    parser.add_argument(
        "--target_model", 
        type=str, 
        default="Qwen/Qwen3-14B",
        help="目标模型（target model）路径"
    )
    parser.add_argument(
        "--draft_model", 
        type=str, 
        default="Qwen/Qwen3-0.6B",
        help="草稿模型（draft model）路径"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="gsm8k",
        choices=["gsm8k", "mmlu", "arc", "hellaswag", "math", "humaneval", "simple"],
        help="测试数据集"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
        help="测试样本数量"
    )
    parser.add_argument(
        "--warm_up",
        action="store_true",
        help="在测试前预热模型"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="最大生成token数"
    )
    parser.add_argument(
        "--save_results", 
        action="store_true",
        help="保存结果到文件"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="output/speculative_timing_results.json",
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Speculative Decoding 时间对比测试")
    print(f"{'='*80}\n")
    print(f"目标模型: {args.target_model}")
    print(f"草稿模型: {args.draft_model}")
    print(f"数据集: {args.dataset}")
    print(f"样本数: {args.num_samples}")
    print(f"最大生成token数: {args.max_new_tokens}")
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 加载模型
    print("\n加载模型中...")
    print(f"  加载草稿模型: {args.draft_model}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print(f"  加载目标模型: {args.target_model}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("模型加载完成！")
    
    # 预热模型（可选）
    if args.warm_up:
        warm_up(target_model, tokenizer, draft_model, device)
    
    # 加载测试提示词
    prompts = load_test_prompts(args.dataset, args.num_samples)
    print(f"加载了 {len(prompts)} 个测试提示词")
    
    # 存储所有结果
    all_results = {
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "samples": []
    }
    
    # 对每个样本进行测试
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n处理样本 {idx}/{len(prompts)}...")
        
        # 标准生成
        print("  运行标准生成...")
        standard_result = generate_standard(
            model=target_model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        
        # Speculative Decoding
        print("  运行 Speculative Decoding...")
        speculative_result = generate_speculative(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        
        # 打印对比结果
        print_comparison(idx, prompt, standard_result, speculative_result)
        
        # 保存结果
        all_results["samples"].append({
            "prompt": prompt,
            "standard": standard_result,
            "speculative": speculative_result,
            "speedup": {
                "latency": standard_result['latency'] / speculative_result['latency'],
                "throughput": speculative_result['tokens_per_second'] / standard_result['tokens_per_second'],
            }
        })
    
    # 计算总体统计
    print(f"\n{'='*80}")
    print(f"总体统计")
    print(f"{'='*80}\n")
    
    avg_standard_latency = np.mean([s["standard"]["latency"] for s in all_results["samples"]])
    avg_standard_throughput = np.mean([s["standard"]["tokens_per_second"] for s in all_results["samples"]])
    avg_standard_tokens = np.mean([s["standard"]["num_tokens"] for s in all_results["samples"]])
    
    avg_spec_latency = np.mean([s["speculative"]["latency"] for s in all_results["samples"]])
    avg_spec_throughput = np.mean([s["speculative"]["tokens_per_second"] for s in all_results["samples"]])
    avg_spec_tokens = np.mean([s["speculative"]["num_tokens"] for s in all_results["samples"]])
    
    avg_latency_speedup = np.mean([s["speedup"]["latency"] for s in all_results["samples"]])
    avg_throughput_speedup = np.mean([s["speedup"]["throughput"] for s in all_results["samples"]])
    
    print(f"[标准生成 - Baseline]")
    print(f"  平均延迟: {avg_standard_latency:.3f}s")
    print(f"  平均吞吐量: {avg_standard_throughput:.2f} tokens/s")
    print(f"  平均生成token数: {avg_standard_tokens:.2f}")
    
    print(f"\n[Speculative Decoding]")
    print(f"  平均延迟: {avg_spec_latency:.3f}s")
    print(f"  平均吞吐量: {avg_spec_throughput:.2f} tokens/s")
    print(f"  平均生成token数: {avg_spec_tokens:.2f}")
    
    print(f"\n[总体性能提升]")
    print(f"  平均延迟加速: {avg_latency_speedup:.2f}x")
    print(f"  平均吞吐量提升: {avg_throughput_speedup:.2f}x")
    print(f"  平均时间节省: {(1 - 1/avg_latency_speedup)*100:.1f}%")
    
    # 保存总体统计
    all_results["summary"] = {
        "standard": {
            "avg_latency": avg_standard_latency,
            "avg_throughput": avg_standard_throughput,
            "avg_tokens": avg_standard_tokens,
        },
        "speculative": {
            "avg_latency": avg_spec_latency,
            "avg_throughput": avg_spec_throughput,
            "avg_tokens": avg_spec_tokens,
        },
        "speedup": {
            "avg_latency_speedup": avg_latency_speedup,
            "avg_throughput_speedup": avg_throughput_speedup,
            "time_saved_percentage": (1 - 1/avg_latency_speedup) * 100,
        }
    }
    
    # 保存结果
    if args.save_results:
        save_results(all_results, args.output_file)
    
    print(f"\n{'='*80}")
    print(f"测试完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

'''
# Qwen 示例
python test_speculative_timing.py \
  --target_model Qwen/Qwen3-14B \
  --draft_model Qwen/Qwen3-0.6B \
  --dataset simple \
  --max_new_tokens 256 \
  --warm_up

# Llama-2 示例（推荐：无需额外权限）
python test_speculative_timing.py \
  --target_model meta-llama/Llama-2-7b-hf \
  --draft_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset simple \
  --num_samples 5 \
  --max_new_tokens 256 \
  --warm_up \
  --save_results

# Llama-2-13B 示例
python test_speculative_timing.py \
  --target_model meta-llama/Llama-2-13b-hf \
  --draft_model JackFram/llama-160m \
  --dataset gsm8k \
  --num_samples 10 \
  --max_new_tokens 512 \
  --warm_up \
  --save_results
'''