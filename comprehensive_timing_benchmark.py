"""
Comprehensive Dataset Timing Benchmark for C2C Model

Supports all datasets used in the repo:
- MMLU-Redux (57 subjects) - General knowledge
- GSM8K (math problems)
- MATH-500 (advanced math)
- LongBench (21 tasks) - Long-context understanding
- OpenHermes (instruction following)
- OpenBookQA - Fact-based reasoning
- AI2-ARC (ARC-Challenge) - Science and logic reasoning
- C-Eval (52 subjects) - Chinese comprehensive knowledge

Usage:
    # MMLU-Redux
    python comprehensive_timing_benchmark.py --dataset mmlu-redux --num_samples 100
    
    # GSM8K
    python comprehensive_timing_benchmark.py --dataset gsm8k --num_samples 500
    
    # ARC-Challenge
    python comprehensive_timing_benchmark.py --dataset ai2-arc --num_samples 100
    
    # C-Eval specific subject
    python comprehensive_timing_benchmark.py --dataset ceval --subject high_school_mathematics --num_samples 50
    
    # All subjects in C-Eval
    python comprehensive_timing_benchmark.py --dataset ceval --all_subjects --num_samples 20

好像没用过，归档...
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

# Suppress warnings
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset configurations (from unified_evaluator.py)
DATASET_CONFIGS = {
    "mmlu-redux": {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "split": "test",
        "subjects": [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
            'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
            'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
            'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics',
            'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
            'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology',
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
        ],
    },
    "gsm8k": {
        "dataset_name": "openai/gsm8k",
        "split": "test",
        "config": "main",
        "subjects": ["main"],
    },
    "math-500": {
        "dataset_name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "subjects": ["main"],
    },
    "longbench": {
        "dataset_name": "THUDM/LongBench",
        "split": "test",
        "subjects": [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", 
            "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", 
            "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", 
            "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
        ],
    },
    "openbookqa": {
        "dataset_name": "allenai/openbookqa",
        "split": "test",
        "config": "main",
        "subjects": ["main"],
    },
    "ai2-arc": {
        "dataset_name": "allenai/ai2_arc",
        "split": "test",
        "config": "ARC-Challenge",
        "subjects": ["ARC-Challenge"],
    },
    "ceval": {
        "dataset_name": "ceval/ceval-exam",
        "split": "test",
        "subjects": [
            "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
            "business_administration", "chinese_language_and_literature", "civil_servant",
            "clinical_medicine", "college_chemistry", "college_economics", "college_physics",
            "college_programming", "computer_architecture", "computer_network",
            "discrete_mathematics", "education_science", "electrical_engineer",
            "environmental_impact_assessment_engineer", "fire_engineer", "high_school_biology",
            "high_school_chemistry", "high_school_chinese", "high_school_geography",
            "high_school_history", "high_school_mathematics", "high_school_physics",
            "high_school_politics", "ideological_and_moral_cultivation", "law",
            "legal_professional", "logic", "mao_zedong_thought", "marxism",
            "metrology_engineer", "middle_school_biology", "middle_school_chemistry",
            "middle_school_geography", "middle_school_history", "middle_school_mathematics",
            "middle_school_physics", "middle_school_politics", "modern_chinese_history",
            "operating_system", "physician", "plant_protection", "probability_and_statistics",
            "professional_tour_guide", "sports_science", "tax_accountant",
            "teacher_qualification", "telecommunication_engineer", "urban_and_rural_planner",
            "veterinary_medicine"
        ],
    },
}

def setup_timing_hooks(model):
    """Setup comprehensive timing hooks"""
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
            # Load first subject as default
            subject = config["subjects"][0]
            dataset = load_dataset(hf_dataset_name, subject, split=split)
    elif dataset_name in ["gsm8k", "openbookqa", "ai2-arc"]:
        dataset_config = config.get("config")
        dataset = load_dataset(hf_dataset_name, dataset_config, split=split)
    elif dataset_name == "ceval":
        if subject:
            dataset = load_dataset(hf_dataset_name, subject, split=split)
        else:
            subject = config["subjects"][0]
            dataset = load_dataset(hf_dataset_name, subject, split=split)
    elif dataset_name == "longbench":
        if subject:
            dataset = load_dataset(hf_dataset_name, subject, split=split)
        else:
            subject = config["subjects"][0]
            dataset = load_dataset(hf_dataset_name, subject, split=split)
    else:
        dataset = load_dataset(hf_dataset_name, split=split)
    
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    return dataset, subject

def format_prompt(example, dataset_name, subject=None):
    """Format example into a prompt based on dataset type"""
    if dataset_name == "mmlu-redux":
        question = example["question"]
        choices = example["choices"]
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        return f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
    
    elif dataset_name == "gsm8k":
        return example["question"]
    
    elif dataset_name == "longbench":
        # LongBench has different formats per task
        if "context" in example and "input" in example:
            return f"Context: {example['context']}\n\nQuestion: {example['input']}\n\nAnswer:"
        elif "input" in example:
            return example["input"]
        else:
            return str(example)
    
    elif dataset_name in ["openbookqa", "ai2-arc"]:
        question = example.get("question_stem", example.get("question", ""))
        choices = example.get("choices", {})
        if isinstance(choices, dict) and "text" in choices:
            choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])])
            return f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
        return question
    
    elif dataset_name == "ceval":
        # C-Eval format: question with A, B, C, D choices
        question = example.get("question", "")
        choices_text = ""
        for letter in ["A", "B", "C", "D"]:
            if letter in example and example[letter]:
                choices_text += f"{letter}. {example[letter]}\n"
        return f"问题：{question}\n\n选项：\n{choices_text}\n答案："
    
    # Default: try common fields
    for field in ["question", "prompt", "text", "input"]:
        if field in example:
            return example[field]
    
    return str(example)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive timing benchmark on multiple datasets")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=list(DATASET_CONFIGS.keys()),
                       help="Dataset to benchmark")
    parser.add_argument("--subject", type=str, help="Specific subject/task (for MMLU, LongBench)")
    parser.add_argument("--all_subjects", action="store_true", help="Run on all subjects")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples per subject")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--output_dir", type=str, default="timing_results", help="Output directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--fuser_checkpoint", type=str, default="qwen3_0.6b+qwen3_4b_base_Fuser")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print("Loading model...")
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
    
    # Setup timing
    timing_data = setup_timing_hooks(rosetta_model)
    
    # Determine subjects to process
    if args.all_subjects:
        subjects = DATASET_CONFIGS[args.dataset]["subjects"]
    elif args.subject:
        subjects = [args.subject]
    else:
        subjects = [None]
    
    print(f"\n{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Subjects to process: {len(subjects) if subjects[0] else 1}")
    print(f"Samples per subject: {args.num_samples}")
    print(f"{'='*80}\n")
    
    # Process each subject
    for subject in subjects:
        subject_name = subject or "main"
        output_file = output_dir / f"{args.dataset}_{subject_name}_timing.jsonl"
        
        # Clear output file
        output_file.write_text("")
        
        print(f"\nProcessing: {args.dataset} - {subject_name}")
        print(f"Output: {output_file}")
        
        # Load dataset
        try:
            dataset, actual_subject = load_dataset_samples(args.dataset, subject, args.num_samples)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue
        
        print(f"Loaded {len(dataset)} samples")
        
        # Process samples
        for idx, example in enumerate(tqdm(dataset, desc=f"{subject_name}")):
            # Reset timing
            for key in timing_data:
                timing_data[key] = []
            
            # Format prompt
            try:
                prompt_text = format_prompt(example, args.dataset, actual_subject)
            except Exception as e:
                print(f"\nError formatting prompt for sample {idx}: {e}")
                continue
            
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
                print(f"\nError generating for sample {idx}: {e}")
                continue
            
            # Create record
            timing_record = {
                "id": idx,
                "dataset": args.dataset,
                "subject": subject_name,
                "prompt": prompt_text[:500],  # Truncate long prompts
                "response": output_text[:500],  # Truncate long responses
                "prompt_length": len(prompt_text),
                "response_length": len(output_text),
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
                    "tokens_per_second": len(timing_data['decode_step_times']) / sum(timing_data['decode_step_times']) if timing_data['decode_step_times'] else 0,
                    "total_time_ms": (
                        sum(timing_data['base_embedding_times']) + 
                        sum(timing_data['base_model_prefill_times']) +
                        sum(timing_data['teacher_embedding_times']) + 
                        sum(timing_data['teacher_model_prefill_times']) +
                        sum(timing_data['projector_times']) +
                        sum(timing_data['decode_step_times'])
                    ) * 1000
                },
                "decode_step_times_ms": [t * 1000 for t in timing_data['decode_step_times'][:50]]  # First 50 steps only
            }
            
            # Save
            with open(output_file, "a") as f:
                f.write(json.dumps(timing_record, ensure_ascii=False) + "\n")
        
        print(f"✓ Completed {subject_name}: {output_file}")
    
    print(f"\n{'='*80}")
    print(f"✓ All subjects completed!")
    print(f"Results saved in: {output_dir}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
