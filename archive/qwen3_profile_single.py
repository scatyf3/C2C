"""
Qwen3 推测解码 - 单样本 Profiling

专门用于 profiling 单个样本的推测解码性能分析。

用法:
    python qwen3_profile_single.py
    python qwen3_profile_single.py --gamma 6
    python qwen3_profile_single.py --use_manual
"""

import torch
import argparse
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datetime import datetime
import torch.profiler as profiler


def generate_speculative_manual(draft_model, target_model, tokenizer, prompt_text, 
                                max_new_tokens=256, gamma=4, device="cuda"):
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


def generate_speculative_api(draft_model, target_model, tokenizer, prompt_text, 
                             max_new_tokens=256, gamma=4, device="cuda"):
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
    }


def profile_generation(draft_model, target_model, tokenizer, prompt_text, 
                      max_new_tokens=256, gamma=4, use_manual=False, device="cuda"):
    """使用 PyTorch Profiler 进行性能分析"""
    
    print("\n开始 Profiling...")
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        if use_manual:
            result = generate_speculative_manual(
                draft_model, target_model, tokenizer, prompt_text,
                max_new_tokens, gamma, device
            )
        else:
            result = generate_speculative_api(
                draft_model, target_model, tokenizer, prompt_text,
                max_new_tokens, gamma, device
            )
    
    return result, prof


def main():
    parser = argparse.ArgumentParser(description="Qwen3 推测解码 - 单样本 Profiling")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成token数")
    parser.add_argument("--gamma", type=int, default=4, help="推测窗口大小 (K)")
    parser.add_argument("--draft_model", type=str, default="Qwen/Qwen3-0.6B", 
                        help="草稿模型（draft model）")
    parser.add_argument("--target_model", type=str, default="Qwen/Qwen3-14B", 
                        help="目标模型（target model）")
    parser.add_argument("--use_manual", action="store_true", help="使用手动实现版本")
    parser.add_argument("--output_dir", type=str, default="output/qwen3_profile", 
                        help="输出目录")
    parser.add_argument("--prompt", type=str, default=None, 
                        help="自定义提示词（默认从GSM8K加载）")
    
    args = parser.parse_args()
    
    # 准备提示词
    if args.prompt:
        prompt_text = args.prompt
    else:
        print("从 GSM8K 数据集加载第一个问题...")
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_text = dataset[0]["question"]
    
    print(f"\n{'='*80}")
    print(f"Qwen3 推测解码 - 单样本 Profiling")
    print(f"{'='*80}\n")
    print(f"提示词: {prompt_text}")
    print(f"最大生成token数: {args.max_new_tokens}")
    print(f"推测窗口大小 (γ): {args.gamma}")
    print(f"实现方式: {'手动实现' if args.use_manual else '官方API'}")
    print(f"草稿模型: {args.draft_model}")
    print(f"目标模型: {args.target_model}")
    
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
    
    # 运行 profiling
    result, prof = profile_generation(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        use_manual=args.use_manual,
        device=device,
    )
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"生成结果")
    print(f"{'='*80}\n")
    print(f"输出: {result['text'][:200]}..." if len(result['text']) > 200 else f"输出: {result['text']}")
    print(f"\n延迟: {result['latency']:.3f}s")
    print(f"生成token数: {result['num_tokens']}")
    print(f"吞吐量: {result['tokens_per_second']:.2f} tokens/s")
    
    if args.use_manual:
        print(f"\n详细统计:")
        print(f"  - 草稿模型生成时间: {result['draft_time']:.3f}s ({result['draft_percentage']:.1f}%)")
        print(f"  - 目标模型验证时间: {result['target_time']:.3f}s ({result['target_percentage']:.1f}%)")
        print(f"  - 接受率: {result['acceptance_rate']:.2%}")
        print(f"  - 平均接受长度: {result['average_accepted_length']:.2f}")
        print(f"  - 草稿模型调用次数: {result['draft_calls']}")
        print(f"  - 目标模型调用次数: {result['target_calls']}")
    
    # 保存 profiling 结果
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    impl_type = "manual" if args.use_manual else "api"
    
    # 保存详细的 profiling trace
    trace_path = output_path / f"profile_trace_{impl_type}_{timestamp}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nProfiler trace 已保存到: {trace_path}")
    print(f"使用 Chrome 浏览器打开 chrome://tracing 查看")
    
    # 打印 profiling 统计
    print(f"\n{'='*80}")
    print(f"Profiler 统计 (按 CUDA 时间排序)")
    print(f"{'='*80}\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # 保存统计结果
    stats_path = output_path / f"profile_stats_{impl_type}_{timestamp}.txt"
    with open(stats_path, 'w') as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    print(f"\nProfiler 统计已保存到: {stats_path}")
    
    # 保存结果到 JSON
    result_path = output_path / f"result_{impl_type}_{timestamp}.json"
    result_data = {
        "config": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "gamma": args.gamma,
            "max_new_tokens": args.max_new_tokens,
            "use_manual": args.use_manual,
        },
        "prompt": prompt_text,
        "result": result,
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {result_path}")
    
    print(f"\n{'='*80}")
    print(f"Profiling 完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
