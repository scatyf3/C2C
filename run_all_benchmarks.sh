#!/bin/bash
#SBATCH --job-name=c2c_timing_benchmark
#SBATCH --output=logs/timing_benchmark_%j.out
#SBATCH --error=logs/timing_benchmark_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --account=pr_289_general

# Quick start script for running timing benchmarks on repo datasets

# Create output directory
mkdir -p logs
mkdir -p timing_results

# Activate conda environment
source activate /scratch/yf3005/rosetta

# Dataset configurations
PYTHON="/scratch/yf3005/rosetta/bin/python"
SCRIPT="comprehensive_timing_benchmark.py"
OUTPUT_DIR="timing_results"

# Sample sizes based on dataset sizes
GSM8K_SAMPLES=100        # GSM8K has ~8K samples
MMLU_SAMPLES=50          # Each MMLU subject has ~100-300 samples
MATH_SAMPLES=100         # MATH-500 has 500 samples
LONGBENCH_SAMPLES=30     # LongBench tasks vary, 30-200 samples each
OPENBOOKQA_SAMPLES=100   # OpenBookQA test has 500 samples
ARC_SAMPLES=100          # ARC-Challenge test has ~1.1K samples
CEVAL_SAMPLES=50         # Each C-Eval subject has ~40-250 samples

echo "======================================================================"
echo "Starting Comprehensive Timing Benchmark"
echo "======================================================================"

# 1. GSM8K (Math reasoning - used in training recipes)
# Uses max_new_tokens=1024 in unified_evaluator.py
echo -e "\n[1/5] Running GSM8K benchmark..."
$PYTHON $SCRIPT \
    --dataset gsm8k \
    --num_samples $GSM8K_SAMPLES \
    --max_new_tokens 1024 \
    --output_dir $OUTPUT_DIR

# 2. MMLU-Redux (General knowledge - used in evaluation recipes)
# Uses max_new_tokens=8 (ablation) or 64 (unified_eval) - multiple choice
echo -e "\n[2/5] Running MMLU-Redux benchmark (subset of subjects)..."
# Run on a few representative subjects
for subject in "high_school_mathematics" "college_computer_science" "high_school_physics" "college_biology"; do
    $PYTHON $SCRIPT \
        --dataset mmlu-redux \
        --subject $subject \
        --num_samples $MMLU_SAMPLES \
        --max_new_tokens 64 \
        --output_dir $OUTPUT_DIR
done

# 3. MMLU-Redux All Subjects (if you want comprehensive results - takes longer)
# Uncomment to run all 57 subjects
# echo -e "\n[3/5] Running MMLU-Redux ALL subjects..."
# $PYTHON $SCRIPT \
#     --dataset mmlu-redux \
#     --all_subjects \
#     --num_samples 20 \
#     --max_new_tokens 64 \
#     --output_dir $OUTPUT_DIR

# 4. MATH-500 (Advanced math)
# Uses max_new_tokens=1024 in unified_evaluator.py
echo -e "\n[3/5] Running MATH-500 benchmark..."
$PYTHON $SCRIPT \
    --dataset math-500 \
    --num_samples $MATH_SAMPLES \
    --max_new_tokens 1024 \
    --output_dir $OUTPUT_DIR

# 5. LongBench (subset of tasks)
# Uses max_new_tokens=1024 in unified_evaluator.py
echo -e "\n[4/5] Running LongBench benchmark (subset)..."
for task in "narrativeqa" "qasper" "hotpotqa" "triviaqa"; do
    $PYTHON $SCRIPT \
        --dataset longbench \
        --subject $task \
        --num_samples $LONGBENCH_SAMPLES \
        --max_new_tokens 1024 \
        --output_dir $OUTPUT_DIR
done

# 6. OpenBookQA
# Multiple choice like MMLU, uses similar max_new_tokens
echo -e "\n[5/7] Running OpenBookQA benchmark..."
$PYTHON $SCRIPT \
    --dataset openbookqa \
    --num_samples $OPENBOOKQA_SAMPLES \
    --max_new_tokens 64 \
    --output_dir $OUTPUT_DIR

# 7. ARC-Challenge
# Science and logic reasoning, multiple choice like MMLU
echo -e "\n[6/7] Running ARC-Challenge benchmark..."
$PYTHON $SCRIPT \
    --dataset ai2-arc \
    --num_samples $ARC_SAMPLES \
    --max_new_tokens 64 \
    --output_dir $OUTPUT_DIR

# 8. C-Eval (Chinese evaluation - subset of subjects)
# Multiple choice, uses similar max_new_tokens as MMLU
echo -e "\n[7/7] Running C-Eval benchmark (subset of subjects)..."
for subject in "high_school_mathematics" "college_computer_science" "high_school_physics" "college_chemistry"; do
    $PYTHON $SCRIPT \
        --dataset ceval \
        --subject $subject \
        --num_samples $CEVAL_SAMPLES \
        --max_new_tokens 64 \
        --output_dir $OUTPUT_DIR
done

echo ""
echo "======================================================================"
echo "✓ All benchmarks completed!"
echo "Results saved in: $OUTPUT_DIR/"
echo "======================================================================"

# Generate summary statistics
echo -e "\nGenerating summary statistics..."
$PYTHON -c "
import json
from pathlib import Path
from collections import defaultdict

stats = defaultdict(list)
results_dir = Path('$OUTPUT_DIR')

for jsonl_file in results_dir.glob('*.jsonl'):
    with open(jsonl_file) as f:
        for line in f:
            record = json.loads(line)
            dataset = record['dataset']
            stats[dataset].append({
                'tokens_per_second': record['timing']['tokens_per_second'],
                'decode_avg_ms': record['timing']['decode_avg_ms'],
                'total_time_ms': record['timing']['total_time_ms']
            })

print('\n' + '='*80)
print('Summary Statistics by Dataset')
print('='*80)
for dataset, records in sorted(stats.items()):
    avg_tps = sum(r['tokens_per_second'] for r in records) / len(records)
    avg_decode = sum(r['decode_avg_ms'] for r in records) / len(records)
    avg_total = sum(r['total_time_ms'] for r in records) / len(records)
    print(f'\n{dataset}:')
    print(f'  Samples: {len(records)}')
    print(f'  Avg Throughput: {avg_tps:.2f} tokens/sec')
    print(f'  Avg Decode Time: {avg_decode:.2f} ms/token')
    print(f'  Avg Total Time: {avg_total:.2f} ms')
print('='*80)
"

echo -e "\nTo view detailed results:"
echo "  cat $OUTPUT_DIR/gsm8k_main_timing.jsonl | jq ."
echo "  cat $OUTPUT_DIR/*.jsonl | jq -s 'map(.timing.tokens_per_second) | add/length'"
