import torch
import logging
import time
from huggingface_hub import snapshot_download
from script.playground.inference_example import load_rosetta_model, run_inference_example

# 配置全局logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 用于记录时间的全局变量
profiling_data = {
    'teacher_kv_generation_time': 0.0,
    'kv_projection_time': 0.0,
    'teacher_kv_generation_count': 0,
    'kv_projection_count': 0
}

checkpoint_dir = snapshot_download(
    repo_id="nics-efc/C2C_Fuser",
    allow_patterns=["qwen3_0.6b+qwen2.5_0.5b_Fuser/*"],
)

model_config = {
    "rosetta_config": {
        "base_model": "Qwen/Qwen3-0.6B",
        "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "checkpoints_dir": f"{checkpoint_dir}/qwen3_0.6b+qwen2.5_0.5b_Fuser/final",
    }
}

rosetta_model, tokenizer = load_rosetta_model(model_config, eval_config={}, device=torch.device("cuda"))
device = rosetta_model.device

# 将profiling_data注入到rosetta_model中以便在wrapper中访问
rosetta_model.profiling_data = profiling_data

prompt = [{"role": "user", "content": "Say hello in one short sentence."}]
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(input_text, return_tensors="pt").to(device)

instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(inputs['input_ids'].shape[1] - 1, 1).unsqueeze(0).to(device)
label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
kv_cache_index = [instruction_index, label_index]

# 开始计时总体生成时间
start_time = time.time()

with torch.no_grad():
    sampling_params = {
        'do_sample': False,
        'max_new_tokens': 256
    }
    outputs = rosetta_model.generate(**inputs, kv_cache_index=kv_cache_index, **sampling_params)
    output_text = tokenizer.decode(outputs[0, instruction_index.shape[1] + 1:], skip_special_tokens=True)
    print(f"C2C output text: {output_text}")

# 结束计时
total_time = time.time() - start_time

# 打印profiling结果
print("\n" + "="*80)
print("PROFILING RESULTS")
print("="*80)
print(f"Total generation time: {total_time:.4f}s")
print(f"\nTeacher model KV generation:")
print(f"  - Total time: {profiling_data['teacher_kv_generation_time']:.4f}s")
print(f"  - Call count: {profiling_data['teacher_kv_generation_count']}")
print(f"  - Percentage: {profiling_data['teacher_kv_generation_time']/total_time*100:.2f}%")
print(f"\nKV cache projection:")
print(f"  - Total time: {profiling_data['kv_projection_time']:.4f}s")
print(f"  - Call count: {profiling_data['kv_projection_count']}")
print(f"  - Percentage: {profiling_data['kv_projection_time']/total_time*100:.2f}%")
print(f"\nOther operations:")
print(f"  - Time: {total_time - profiling_data['teacher_kv_generation_time'] - profiling_data['kv_projection_time']:.4f}s")
print(f"  - Percentage: {(total_time - profiling_data['teacher_kv_generation_time'] - profiling_data['kv_projection_time'])/total_time*100:.2f}%")
print("="*80)