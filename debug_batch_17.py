"""Debug why batch 17 has no gradients"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import create_projector
from rosetta.train.model_utils import last_aligned_sources
from rosetta.train.dataset_adapters import create_dataset, ChatDataset, RosettaDataCollator
from rosetta.utils.evaluate import set_default_chat_template

# Load config
with open("recipe/train_recipe/C2C_llama160m+llama2_13b.json") as f:
    config = json.load(f)

model_config = config["model"]
training_config = config["training"]
data_config = config["data"]
device = "cuda"
dtype = torch.bfloat16

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
set_default_chat_template(tokenizer, model_config["base_model"])

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

# Freeze
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
for target_layer_idx, src_list in source_target_mapping.items():
    for source_layer_idx in src_list:
        rosetta_model.set_projector_config(
            source_model_idx=1,
            source_model_layer_idx=source_layer_idx,
            target_model_idx=0,
            target_model_layer_idx=target_layer_idx,
            projector_idx=target_layer_idx,
        )

print("Loading dataset...")
instruct_ds = create_dataset(dataset_type=data_config["type"], **data_config["kwargs"])
full_dataset = ChatDataset(instruct_ds, tokenizer)

# Create collator
collator = RosettaDataCollator(
    slm_tokenizer=tokenizer,
    llm_tokenizer=tokenizer,
    max_length=training_config.get("max_length", 2048),
    do_alignment=False
)

# Test batches around index 17 (batch_size=1, so sample index = batch index)
print("\nTesting samples around batch 17...")
for sample_idx in range(15, 20):
    print(f"\n{'='*80}")
    print(f"Sample {sample_idx}:")
    try:
        sample = full_dataset[sample_idx]
        batch = collator([sample])
        
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Num valid labels: {(batch['labels'][0] != -100).sum().item()}")
        print(f"KV cache sections: {len(batch['kv_cache_index'])}")
        for i, kv_idx in enumerate(batch['kv_cache_index']):
            print(f"  Section {i}: shape={kv_idx.shape}, first_value={kv_idx[0,0,0]}")
        
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else [x.to(device) for x in v] for k, v in batch.items()}
        
        # Forward
        rosetta_model.train()
        with torch.set_grad_enabled(True):
            outputs = rosetta_model.forward(
                kv_cache_index=batch['kv_cache_index'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                position_ids=batch['position_ids'],
                labels=batch['labels'],
                use_cache=True
            )
            
            print(f"Loss: {outputs.loss.item():.4f}")
            print(f"Loss requires_grad: {outputs.loss.requires_grad}")
            print(f"Loss grad_fn: {outputs.loss.grad_fn}")
            
            if not outputs.loss.requires_grad:
                print("❌ NO GRADIENTS! This is the problematic sample!")
                print(f"Sample data: {sample}")
            else:
                print("✓ Has gradients")
                
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Analysis complete!")
