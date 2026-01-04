#!/bin/bash
#SBATCH --job-name=qwen_param_sweep
#SBATCH --output=logs/qwen_sweep_%j.out
#SBATCH --error=logs/qwen_sweep_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --account=pr_289_general

# Qwen参数扫描脚本
# 用法: sbatch run_qwen_sweep.sh
# 或带参数: sbatch run_qwen_sweep.sh --quick

echo "======================================================================"
echo "Starting Qwen Speculative Decoding Parameter Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================================================"

# Create output directories
mkdir -p logs
mkdir -p output/sweep_results

# Activate conda environment
echo -e "\nActivating conda environment..."
source activate /scratch/yf3005/rosetta

# Set Python executable
PYTHON="/scratch/yf3005/rosetta/bin/python"
SCRIPT="sweep_speculative_params.py"

# Print environment info
echo -e "\nEnvironment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Parse command line arguments - forward all arguments to Python script
ARGS="$@"

# Default configurations (can be overridden by command line args)
# These are the default values from the Python script:
# - draft_model: Qwen/Qwen2.5-0.5B-Instruct (default)
# - target_models: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, etc.
# - datasets: gsm8k, mmlu, arc, math, simple
# - max_tokens_list: 64, 128, 256, 512
# - num_samples: 10 (default) or 3 (quick mode)

echo -e "\n======================================================================"
echo "Running Parameter Sweep"
echo "======================================================================"
echo "Arguments: $ARGS"
echo ""

# Run the sweep
$PYTHON $SCRIPT $ARGS

EXIT_CODE=$?

echo -e "\n======================================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "======================================================================"

# Print output file locations
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\nResults saved to: output/sweep_results/"
    echo "Log files:"
    echo "  - stdout: logs/qwen_sweep_${SLURM_JOB_ID}.out"
    echo "  - stderr: logs/qwen_sweep_${SLURM_JOB_ID}.err"
    
    # List generated files
    echo -e "\nGenerated files:"
    ls -lh output/sweep_results/ | tail -10
fi

exit $EXIT_CODE
