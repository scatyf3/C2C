# Timing Benchmark Datasets

This document lists all datasets supported by the comprehensive timing benchmark, matching the evaluation benchmarks used in the paper.

## Supported Datasets

### 1. **GSM8K** - Math Reasoning
- **Dataset**: `openai/gsm8k`
- **Split**: test (8K samples)
- **Type**: Math word problems
- **Max tokens**: 1024 (from `unified_evaluator.py`)
- **Example**:
  ```
  Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast...
  ```

### 2. **MMLU-Redux** - General Knowledge
- **Dataset**: `edinburgh-dawg/mmlu-redux-2.0`
- **Split**: test
- **Subjects**: 57 subjects across STEM, humanities, social sciences
- **Type**: Multiple-choice questions (4 options)
- **Max tokens**: 64 (from `unified_eval.yaml`)
- **Example subjects**: 
  - high_school_mathematics
  - college_computer_science
  - college_biology
  - high_school_physics

### 3. **MATH-500** - Advanced Math
- **Dataset**: `HuggingFaceH4/MATH-500`
- **Split**: test (500 samples)
- **Type**: Competition-level math problems
- **Max tokens**: 1024 (from `unified_evaluator.py`)

### 4. **LongBench** - Long-Context Understanding
- **Dataset**: `THUDM/LongBench`
- **Split**: test
- **Tasks**: 21 tasks across QA, summarization, code
- **Max tokens**: 1024 (from `unified_evaluator.py`)
- **Example tasks**:
  - narrativeqa (reading comprehension)
  - qasper (scientific QA)
  - hotpotqa (multi-hop reasoning)
  - triviaqa (factual QA)

### 5. **OpenBookQA** - Fact-based Reasoning
- **Dataset**: `allenai/openbookqa`
- **Split**: test (500 samples)
- **Type**: Multiple-choice questions requiring science facts
- **Max tokens**: 64 (multiple-choice like MMLU)

### 6. **ARC-Challenge** - Science & Logic Reasoning
- **Dataset**: `allenai/ai2_arc`
- **Config**: ARC-Challenge
- **Split**: test (1,172 samples)
- **Type**: Multiple-choice science questions (Grade 3-9)
- **Max tokens**: 64 (multiple-choice like MMLU)
- **Example**:
  ```
  Question: An astronomer observes that a planet rotates faster after a meteorite impact...
  Choices:
  A. Planetary density will decrease.
  B. Planetary years will become longer.
  C. Planetary days will become shorter.
  D. Planetary gravity will become stronger.
  ```

### 7. **C-Eval** - Chinese Comprehensive Knowledge
- **Dataset**: `ceval/ceval-exam`
- **Split**: test
- **Subjects**: 52 subjects covering Chinese educational curriculum
- **Type**: Multiple-choice questions (4 options, in Chinese)
- **Max tokens**: 64 (multiple-choice like MMLU)
- **Example subjects**:
  - high_school_mathematics (高中数学)
  - college_computer_science (大学计算机)
  - high_school_physics (高中物理)
  - college_chemistry (大学化学)
- **Example**:
  ```
  问题：圆锥的底面半径为2，高为4...
  选项：
  A. $\pi$
  B. $2\pi$
  C. $3\pi$
  D. $4\pi$
  ```

## Configuration Summary

| Dataset | Samples (test) | Type | Max Tokens | Config File |
|---------|---------------|------|------------|-------------|
| GSM8K | 8,000+ | Open-ended | 1024 | unified_evaluator.py |
| MMLU-Redux | 57 subjects | Multiple-choice | 64 | unified_eval.yaml |
| MATH-500 | 500 | Open-ended | 1024 | unified_evaluator.py |
| LongBench | 21 tasks | Mixed | 1024 | unified_evaluator.py |
| OpenBookQA | 500 | Multiple-choice | 64 | Inferred from MMLU |
| ARC-Challenge | 1,172 | Multiple-choice | 64 | Inferred from MMLU |
| C-Eval | 52 subjects | Multiple-choice | 64 | Inferred from MMLU |

## Paper Benchmarks Coverage

As mentioned in the paper, the evaluation covers:
1. ✅ **OpenBookQA** - Fact-based reasoning
2. ✅ **MMLU-Redux** - General domain knowledge
3. ✅ **ARC-Challenge** - Science and logic reasoning
4. ✅ **C-Eval** - Chinese comprehensive knowledge

Plus additional benchmarks:
- GSM8K, MATH-500 (math reasoning)
- LongBench (long-context understanding)

## Usage Examples

```bash
# Run GSM8K benchmark
python comprehensive_timing_benchmark.py --dataset gsm8k --num_samples 100 --max_new_tokens 1024

# Run MMLU-Redux specific subject
python comprehensive_timing_benchmark.py --dataset mmlu-redux --subject high_school_mathematics --num_samples 50 --max_new_tokens 64

# Run ARC-Challenge
python comprehensive_timing_benchmark.py --dataset ai2-arc --num_samples 100 --max_new_tokens 64

# Run C-Eval specific subject
python comprehensive_timing_benchmark.py --dataset ceval --subject high_school_mathematics --num_samples 50 --max_new_tokens 64

# Run all C-Eval subjects
python comprehensive_timing_benchmark.py --dataset ceval --all_subjects --num_samples 20 --max_new_tokens 64

# Run all benchmarks at once
./run_all_benchmarks.sh
```

## Output Format

Each benchmark generates a JSONL file with timing breakdowns:

```json
{
  "prompt": "Question: ...",
  "response": "Answer: ...",
  "base_embedding_ms": 432.03,
  "base_prefill_ms": 267.71,
  "teacher_embedding_ms": 89.45,
  "teacher_prefill_ms": 196.75,
  "projector_ms": 15.23,
  "decode_avg_ms": 72.44,
  "total_ms": 3845.67,
  "num_decode_steps": 42,
  "throughput_tokens_per_sec": 13.8
}
```
