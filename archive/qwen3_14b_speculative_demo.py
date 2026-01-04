"""
Qwen3-14B + Qwen3-0.6B Speculative Decoding Demo

使用 Qwen3-14B 作为目标模型，Qwen3-0.6B 作为草稿模型进行推测解码。
默认从 GSM8K 数据集加载前 10 个问题进行测试。

用法:
    python qwen3_14b_speculative_demo.py
    python qwen3_14b_speculative_demo.py --gamma 6 --max_new_tokens 512
    python qwen3_14b_speculative_demo.py --num_samples 20
    python qwen3_14b_speculative_demo.py --num_samples 20 --gamma 8
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


def load_gsm8k_samples(num_samples=10):
    """从 GSM8K 数据集加载样本"""
    print(f"从 GSM8K 数据集加载前 {num_samples} 个问题...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    prompts = [example["question"] for example in dataset]
    return prompts


def generate_standard(target_model, tokenizer, prompt_text, max_new_tokens=1024, device="cuda"):
    """使用标准方法生成（仅目标模型）"""
    # 准备输入
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 生成
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 解码输出
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


def generate_speculative(draft_model, target_model, tokenizer, prompt_text, 
                         max_new_tokens=1024, gamma=4, device="cuda"):
    """使用 Transformers 内置的推测解码 API"""
    # 准备输入
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 生成（使用 Transformers 内置推测解码）
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            assistant_model=draft_model,  # 指定草稿模型启用推测解码
        )
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 解码输出
    output_text = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    latency = end_time - start_time
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = num_tokens / latency if latency > 0 else 0
    
    return {
        "text": output_text,
        "latency": latency,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_sec,
        "acceptance_rate": 0.0,  # 官方API不直接提供
        "average_accepted_length": 0.0,
        "draft_calls": 0,
        "target_calls": 0,
        "total_drafted": 0,
        "total_accepted": 0,
        "draft_time": 0.0,
        "target_time": 0.0,
        "draft_percentage": 0.0,
        "target_percentage": 0.0,
    }


def generate_speculative_manual(draft_model, target_model, tokenizer, prompt_text, 
                                max_new_tokens=1024, gamma=4, device="cuda"):
    """手动实现推测解码（带KV cache管理和详细统计）"""
    # 准备输入
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    
    # 初始化统计
    total_drafted = 0
    total_accepted = 0
    draft_calls = 0
    target_calls = 0
    draft_time = 0.0
    target_time = 0.0
    
    # 生成
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        current_ids = input_ids.clone()
        generated_tokens = 0
        
        # 使用 past_key_values 进行 KV cache
        draft_past = None
        target_past = None
        
        while generated_tokens < max_new_tokens:
            # Step 1: 草稿模型生成 gamma 个 token
            torch.cuda.synchronize()
            draft_start = time.perf_counter()
            
            # Draft 模型增量生成
            draft_tokens = []
            draft_input = current_ids if draft_past is None else current_ids[:, -1:]
            temp_past = draft_past
            
            for _ in range(gamma):
                outputs = draft_model(
                    draft_input,
                    past_key_values=temp_past,
                    use_cache=True,
                    return_dict=True
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                temp_past = outputs.past_key_values
                draft_input = next_token
                draft_calls += 1
            
            draft_past = temp_past
            drafted_tokens = torch.cat(draft_tokens, dim=-1)
            
            torch.cuda.synchronize()
            draft_time += time.perf_counter() - draft_start
            total_drafted += gamma
            
            # Step 2: 目标模型验证
            torch.cuda.synchronize()
            target_start = time.perf_counter()
            
            # Target 模型验证草稿 tokens
            verify_input = drafted_tokens if target_past is not None else torch.cat([current_ids, drafted_tokens], dim=-1)
            target_outputs = target_model(
                verify_input,
                past_key_values=target_past,
                use_cache=True,
                return_dict=True
            )
            
            torch.cuda.synchronize()
            target_time += time.perf_counter() - target_start
            target_calls += 1
            
            # Step 3: 验证并接受 tokens
            target_logits = target_outputs.logits
            accepted = 0
            
            for i in range(gamma):
                # 获取目标模型对第 i 个位置的预测
                if target_past is not None:
                    target_token = target_logits[:, i, :].argmax(dim=-1)
                else:
                    # 第一次需要考虑原始序列长度
                    target_token = target_logits[:, current_ids.shape[1] - 1 + i, :].argmax(dim=-1)
                
                drafted_token = drafted_tokens[:, i]
                
                if target_token == drafted_token:
                    accepted += 1
                else:
                    # 拒绝，使用目标模型预测的 token
                    drafted_tokens[:, i] = target_token
                    break
            
            # 更新序列
            if accepted == gamma:
                # 全部接受，额外采样一个 token
                extra_token = target_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                accepted_tokens = torch.cat([drafted_tokens, extra_token], dim=-1)
                current_ids = torch.cat([current_ids, accepted_tokens], dim=-1)
                generated_tokens += gamma + 1
                total_accepted += gamma + 1
                target_past = target_outputs.past_key_values
            else:
                # 部分接受
                accepted_tokens = drafted_tokens[:, :accepted+1]
                current_ids = torch.cat([current_ids, accepted_tokens], dim=-1)
                generated_tokens += accepted + 1
                total_accepted += accepted + 1
                # 重置 cache（简化处理）
                draft_past = None
                target_past = None
            
            # 检查是否遇到结束符
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in accepted_tokens[0]:
                break
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 解码输出
    output_text = tokenizer.decode(current_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    latency = end_time - start_time
    num_tokens = current_ids.shape[1] - input_ids.shape[1]
    tokens_per_sec = num_tokens / latency if latency > 0 else 0
    
    # 计算统计信息
    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    avg_accepted_length = total_accepted / target_calls if target_calls > 0 else 0
    draft_percentage = (draft_time / latency * 100) if latency > 0 else 0
    target_percentage = (target_time / latency * 100) if latency > 0 else 0
    
    return {
        "text": output_text,
        "latency": latency,
        "num_tokens": num_tokens,
        "tokens_per_second": tokens_per_sec,
        "acceptance_rate": acceptance_rate,
        "average_accepted_length": avg_accepted_length,
        "draft_calls": draft_calls,
        "target_calls": target_calls,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "draft_time": draft_time,
        "target_time": target_time,
        "draft_percentage": draft_percentage,
        "target_percentage": target_percentage,
    }


def save_results(results, output_dir="output/qwen3_speculative"):
    """保存结果到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON格式
    json_path = output_path / f"results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-14B + Qwen3-0.6B 推测解码演示")
    parser.add_argument("--num_samples", type=int, default=10, help="从 GSM8K 加载的样本数")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成token数")
    parser.add_argument("--gamma", type=int, default=4, help="推测窗口大小 (K)")
    parser.add_argument("--draft_model", type=str, default="Qwen/Qwen3-0.6B", 
                        help="草稿模型（draft model）")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-14B", 
                        help="目标模型（target model）")
    parser.add_argument("--skip_standard", action="store_true", help="跳过标准生成（仅运行推测解码）")
    parser.add_argument("--save_results", action="store_true", help="保存结果到文件")
    parser.add_argument("--use_manual", action="store_true", help="使用手动实现版本（带详细统计）")
    parser.add_argument("--output_dir", type=str, default="output/qwen3_speculative", 
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 加载 GSM8K 数据集
    prompts = load_gsm8k_samples(num_samples=args.num_samples)
    
    print(f"\n{'='*80}")
    print(f"Qwen3 推测解码演示 - GSM8K 测试")
    print(f"{'='*80}\n")
    print(f"数据集: GSM8K")
    print(f"样本数量: {len(prompts)}")
    print(f"最大生成token数: {args.max_new_tokens}")
    print(f"推测窗口大小 (γ): {args.gamma}")
    print(f"草稿模型（Draft）: {args.draft_model}")
    print(f"目标模型（Target）: {args.target_model}")
    
    # 加载模型
    print("\n加载模型中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"加载草稿模型: {args.draft_model}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print(f"加载目标模型: {args.target_model}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"模型已加载到 {device}")
    
    # 收集所有结果
    all_results = {
        "config": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "gamma": args.gamma,
            "max_new_tokens": args.max_new_tokens,
            "dataset": "gsm8k",
            "num_samples": args.num_samples,
        },
        "prompts": []
    }
    
    # 处理每个提示词
    for idx, prompt_text in enumerate(prompts, 1):
        print(f"\n{'='*80}")
        print(f"提示词 {idx}/{len(prompts)}")
        print(f"{'='*80}\n")
        print(f"输入: {prompt_text}\n")
        
        prompt_result = {
            "prompt": prompt_text,
            "standard": None,
            "speculative": None,
        }
        
        # 标准生成（基线）
        if not args.skip_standard:
            print("运行标准生成（仅目标模型）...")
            standard_result = generate_standard(
                target_model=target_model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            
            prompt_result["standard"] = standard_result
            
            print(f"\n[标准生成]")
            print(f"输出: {standard_result['text'][:150]}..." if len(standard_result['text']) > 150 else f"输出: {standard_result['text']}")
            print(f"延迟: {standard_result['latency']:.3f}s")
            print(f"生成token数: {standard_result['num_tokens']}")
            print(f"吞吐量: {standard_result['tokens_per_second']:.2f} tokens/s")
        
        # 推测解码生成
        print("\n运行推测解码生成...")
        
        # 选择使用哪种实现
        if args.use_manual:
            print("(使用手动实现版本)")
            speculative_result = generate_speculative_manual(
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                device=device,
            )
        else:
            print("(使用官方API)")
            speculative_result = generate_speculative(
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                device=device,
            )
        
        prompt_result["speculative"] = speculative_result
        
        print(f"\n[推测解码 (γ={args.gamma})]")
        print(f"输出: {speculative_result['text'][:150]}..." if len(speculative_result['text']) > 150 else f"输出: {speculative_result['text']}")
        print(f"延迟: {speculative_result['latency']:.3f}s")
        
        # 如果使用手动实现，显示详细统计
        if args.use_manual:
            print(f"  - 草稿模型生成时间: {speculative_result['draft_time']:.3f}s ({speculative_result['draft_percentage']:.1f}%)")
            print(f"  - 目标模型验证时间: {speculative_result['target_time']:.3f}s ({speculative_result['target_percentage']:.1f}%)")
        
        print(f"生成token数: {speculative_result['num_tokens']}")
        print(f"吞吐量: {speculative_result['tokens_per_second']:.2f} tokens/s")
        
        # 如果使用手动实现，显示额外统计
        if args.use_manual:
            print(f"接受率: {speculative_result['acceptance_rate']:.2%}")
            print(f"平均接受长度: {speculative_result['average_accepted_length']:.2f}")
            print(f"草稿模型调用次数: {speculative_result['draft_calls']}")
            print(f"目标模型调用次数: {speculative_result['target_calls']}")
        
        # 对比分析
        if not args.skip_standard:
            latency_speedup = standard_result['latency'] / speculative_result['latency']
            throughput_speedup = speculative_result['tokens_per_second'] / standard_result['tokens_per_second']
            
            print(f"\n[性能对比]")
            print(f"延迟改善: {latency_speedup:.2f}x 更快")
            print(f"吞吐量提升: {throughput_speedup:.2f}x 更高")
            
            prompt_result["comparison"] = {
                "latency_speedup": latency_speedup,
                "throughput_speedup": throughput_speedup,
            }
        
        all_results["prompts"].append(prompt_result)
    
    # 计算总体统计
    print(f"\n{'='*80}")
    print(f"总体统计")
    print(f"{'='*80}\n")
    
    if not args.skip_standard:
        avg_standard_latency = sum(p["standard"]["latency"] for p in all_results["prompts"]) / len(prompts)
        avg_standard_throughput = sum(p["standard"]["tokens_per_second"] for p in all_results["prompts"]) / len(prompts)
        avg_standard_tokens = sum(p["standard"]["num_tokens"] for p in all_results["prompts"]) / len(prompts)
        print(f"标准生成平均延迟: {avg_standard_latency:.3f}s")
        print(f"标准生成平均吞吐量: {avg_standard_throughput:.2f} tokens/s")
        print(f"标准生成平均token数: {avg_standard_tokens:.2f}\n")
    
    avg_spec_latency = sum(p["speculative"]["latency"] for p in all_results["prompts"]) / len(prompts)
    avg_spec_throughput = sum(p["speculative"]["tokens_per_second"] for p in all_results["prompts"]) / len(prompts)
    avg_spec_tokens = sum(p["speculative"]["num_tokens"] for p in all_results["prompts"]) / len(prompts)
    
    print(f"推测解码平均延迟: {avg_spec_latency:.3f}s")
    
    # 如果使用手动实现，显示详细统计
    if args.use_manual:
        avg_acceptance_rate = sum(p["speculative"]["acceptance_rate"] for p in all_results["prompts"]) / len(prompts)
        avg_accepted_length = sum(p["speculative"]["average_accepted_length"] for p in all_results["prompts"]) / len(prompts)
        avg_draft_time = sum(p["speculative"]["draft_time"] for p in all_results["prompts"]) / len(prompts)
        avg_target_time = sum(p["speculative"]["target_time"] for p in all_results["prompts"]) / len(prompts)
        avg_draft_percentage = sum(p["speculative"]["draft_percentage"] for p in all_results["prompts"]) / len(prompts)
        avg_target_percentage = sum(p["speculative"]["target_percentage"] for p in all_results["prompts"]) / len(prompts)
        
        print(f"  - 平均草稿模型生成时间: {avg_draft_time:.3f}s ({avg_draft_percentage:.1f}%)")
        print(f"  - 平均目标模型验证时间: {avg_target_time:.3f}s ({avg_target_percentage:.1f}%)")
    
    print(f"推测解码平均吞吐量: {avg_spec_throughput:.2f} tokens/s")
    print(f"推测解码平均token数: {avg_spec_tokens:.2f}")
    
    if args.use_manual:
        print(f"平均接受率: {avg_acceptance_rate:.2%}")
        print(f"平均接受长度: {avg_accepted_length:.2f}")
    
    all_results["summary"] = {
        "avg_speculative_latency": avg_spec_latency,
        "avg_speculative_throughput": avg_spec_throughput,
        "avg_speculative_tokens": avg_spec_tokens,
    }
    
    if args.use_manual:
        all_results["summary"].update({
            "avg_acceptance_rate": avg_acceptance_rate,
            "avg_accepted_length": avg_accepted_length,
            "avg_draft_time": avg_draft_time,
            "avg_target_time": avg_target_time,
            "avg_draft_percentage": avg_draft_percentage,
            "avg_target_percentage": avg_target_percentage,
        })
    
    if not args.skip_standard:
        overall_latency_speedup = avg_standard_latency / avg_spec_latency
        overall_throughput_speedup = avg_spec_throughput / avg_standard_throughput
        print(f"\n总体延迟改善: {overall_latency_speedup:.2f}x")
        print(f"总体吞吐量提升: {overall_throughput_speedup:.2f}x")
        
        all_results["summary"]["avg_standard_latency"] = avg_standard_latency
        all_results["summary"]["avg_standard_throughput"] = avg_standard_throughput
        all_results["summary"]["avg_standard_tokens"] = avg_standard_tokens
        all_results["summary"]["overall_latency_speedup"] = overall_latency_speedup
        all_results["summary"]["overall_throughput_speedup"] = overall_throughput_speedup
    
    # 保存结果
    if args.save_results:
        save_results(all_results, args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"演示完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
