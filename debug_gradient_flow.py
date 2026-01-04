"""Debug gradient flow in training"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import create_projector
from rosetta.train.model_utils import last_aligned_sources

# Load config
with open("recipe/train_recipe/C2C_llama160m+llama2_13b.json") as f:
    config = json.load(f)

model_config = config["model"]
device = "cuda"
dtype = torch.bfloat16

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading models...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_config["base_model"],
    torch_dtype=dtype,
    device_map=device
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    model_config["teacher_model"],
    torch_dtype=dtype,
    device_map=device
)

# Freeze models
for param in base_model.parameters():
    param.requires_grad = False
for param in teacher_model.parameters():
    param.requires_grad = False

# Get dimensions
base_dim = int(base_model.model.layers[0].self_attn.k_proj.out_features / base_model.config.num_key_value_heads)
teacher_dim = int(teacher_model.model.layers[0].self_attn.k_proj.out_features / teacher_model.config.num_key_value_heads)
base_num_heads = base_model.config.num_key_value_heads
teacher_num_heads = teacher_model.config.num_key_value_heads
slm_num_layers = base_model.config.num_hidden_layers
llm_num_layers = teacher_model.config.num_hidden_layers

print(f"Base: {slm_num_layers} layers, dim={base_dim}, heads={base_num_heads}")
print(f"Teacher: {llm_num_layers} layers, dim={teacher_dim}, heads={teacher_num_heads}")

# Create projectors
projector_config = model_config["projector"]
projector_params = projector_config["params"].copy()
projector_params["dtype"] = dtype
projector_list = []

for _ in range(slm_num_layers):
    projector = create_projector(
        projector_config["type"],
        source_dim=teacher_dim,
        target_dim=base_dim,
        source_num_heads=teacher_num_heads,
        target_num_heads=base_num_heads,
        **projector_params
    )
    projector_list.append(projector.to(device))

# Create RosettaModel
rosetta_model = RosettaModel(
    model_list=[base_model, teacher_model],
    base_model_idx=0,
    projector_list=projector_list,
    aggregator_list=[]
).to(device)

# Setup mapping
source_target_mapping = last_aligned_sources(slm_num_layers, llm_num_layers, 1)
print(f"Mapping: {source_target_mapping}")

for target_layer_idx, src_list in source_target_mapping.items():
    for source_layer_idx in src_list:
        rosetta_model.set_projector_config(
            source_model_idx=1,
            source_model_layer_idx=source_layer_idx,
            target_model_idx=0,
            target_model_layer_idx=target_layer_idx,
            projector_idx=target_layer_idx,
        )

# Check trainable parameters
total_params = sum(p.numel() for p in rosetta_model.parameters())
trainable_params = sum(p.numel() for p in rosetta_model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

# Create simple input
prompt = "Hello, how are you?"
# Simple text without chat template for llama-160m
text = f"User: {prompt}\nAssistant:"
inputs = tokenizer(text, return_tensors="pt").to(device)

# Create kv_cache_index: first part needs projection, second part doesn't
seq_len = inputs['input_ids'].shape[1]
instruction_len = seq_len - 5  # Simul instruction + small label
instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(instruction_len - 1, 1).unsqueeze(0).to(device)
label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(seq_len - instruction_len + 1, 1).unsqueeze(0).to(device)
kv_cache_index = [instruction_index, label_index]

print(f"\nInput shape: {inputs['input_ids'].shape}")
print(f"Instruction tokens: {instruction_len}, Label tokens: {seq_len - instruction_len + 1}")
print(f"KV cache index sections: {len(kv_cache_index)}")

# Create labels (only compute loss on last few tokens)
labels = inputs['input_ids'].clone()
labels[:, :instruction_len] = -100  # Ignore instruction part

# Forward pass
print("\nRunning forward pass...")
rosetta_model.train()  # Set to training mode
outputs = rosetta_model.forward(
    kv_cache_index=kv_cache_index,
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    labels=labels,
    use_cache=True
)

print(f"Loss: {outputs.loss.item()}")
print(f"Loss requires_grad: {outputs.loss.requires_grad}")
print(f"Loss grad_fn: {outputs.loss.grad_fn}")

# Try backward
if outputs.loss.requires_grad:
    print("\nTrying backward...")
    outputs.loss.backward()
    print("✓ Backward successful!")
    
    # Check gradients
    proj_grads = [p.grad is not None for p in rosetta_model.projector_list[0].parameters()]
    print(f"Projector gradients: {sum(proj_grads)}/{len(proj_grads)} parameters have gradients")
else:
    print("\n✗ Loss does not require gradients!")
    print("\nInspecting gradient chain...")
    # Check if any projector parameters are in the computation graph
    for i, proj in enumerate(rosetta_model.projector_list):
        has_grad = any(p.requires_grad for p in proj.parameters())
        print(f"Projector {i}: requires_grad={has_grad}")
