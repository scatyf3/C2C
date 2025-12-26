#!/bin/bash
# Speculative Decoding Examples

echo "================================"
echo "Speculative Decoding Examples"
echo "================================"
echo ""

# Example 1: Basic Demo
echo "1. Running basic demo with default prompts..."
python demo_speculative_decoding.py --gamma 4 --max_new_tokens 128

echo ""
echo "================================"
echo ""

# Example 2: Custom prompts
echo "2. Running with custom prompts..."
cat > /tmp/test_prompts.txt << EOF
What is the Pythagorean theorem?
Explain photosynthesis.
Write a function to sort an array in Python.
EOF

python demo_speculative_decoding.py --prompts /tmp/test_prompts.txt --gamma 4

echo ""
echo "================================"
echo ""

# Example 3: Benchmark on GSM8K
echo "3. Benchmarking on GSM8K dataset..."
python speculative_benchmark.py \
    --dataset gsm8k \
    --num_samples 20 \
    --gamma 4 \
    --max_new_tokens 256 \
    --output_dir results_gsm8k

echo ""
echo "================================"
echo ""

# Example 4: Compare different gamma values
echo "4. Comparing different gamma values..."
for gamma in 2 4 6; do
    echo "Testing gamma=$gamma..."
    python speculative_benchmark.py \
        --dataset gsm8k \
        --num_samples 10 \
        --gamma $gamma \
        --max_new_tokens 128 \
        --output_dir results_gamma_comparison
done

echo ""
echo "================================"
echo ""

echo "All examples completed!"
echo "Check results in:"
echo "  - results_gsm8k/"
echo "  - results_gamma_comparison/"
