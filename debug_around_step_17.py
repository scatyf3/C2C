"""
Debug script to check samples around the 17th optimizer step (batch ~272).
With grad_accum_steps=16, step 17 corresponds to batches 256-271.
"""
import torch
import sys
sys.path.append("/home/yf3005/C2C")

from rosetta.train.data_utils import ChatDataset, RosettaDataCollator
from rosetta.model.wrapper import RosettaModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m")
    if tokenizer.chat_template is None:
        print("Model JackFram/llama-160m has no chat template, setting default template...")
        tokenizer.chat_template = """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"""
        print("Default chat template has been set.")
    
    print("Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "JackFram/llama-160m",
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-hf",
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    # Create projector config
    with open("/home/yf3005/C2C/recipe/train_recipe/C2C_llama160m+llama2_13b.json", "r") as f:
        config = json.load(f)
    
    projector_dict = {0: config["projector"]}
    
    model = RosettaModel(
        base_model=base_model,
        base_model_idx=0,
        teacher_models=[teacher_model],
        projector_dict=projector_dict,
        mode="train"
    )
    model = model.to("cuda")
    model.eval()
    
    print("Loading dataset...")
    dataset = ChatDataset("default", split="train")
    collator = RosettaDataCollator(
        tokenizer=tokenizer,
        mode="rosetta",
        max_length=1024
    )
    
    print(f"\nChecking batches around optimizer step 17 (batches 256-271)...")
    print("="*80)
    
    # Check batches 256-271 (step 17 = batches 16*16 to 16*17-1 = 256-271)
    for sample_idx in range(256, 272):
        raw_sample = dataset[sample_idx]
        batch = collator([raw_sample])
        
        # Move to GPU
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        print(f"\nSample {sample_idx}:")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        # Count valid labels
        valid_labels = (batch['labels'] != -100).sum().item()
        print(f"Num valid labels: {valid_labels}")
        
        # Check kv_cache_index structure
        if batch['kv_cache_index'] is not None:
            kv_idx = batch['kv_cache_index']
            print(f"KV cache sections: {len(kv_idx)}")
            for sec_idx, section in enumerate(kv_idx):
                print(f"  Section {sec_idx}: shape={section.shape}, first_value={section[0,0,0,0].item()}")
        
        # Try forward pass
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    kv_cache_index=batch['kv_cache_index']
                )
            loss = outputs.loss
            print(f"Loss: {loss.item():.4f}")
            
            # Now try with grad enabled
            model.train()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                kv_cache_index=batch['kv_cache_index']
            )
            loss = outputs.loss
            print(f"Loss requires_grad: {loss.requires_grad}")
            print(f"Loss grad_fn: {loss.grad_fn}")
            if loss.requires_grad:
                print("✓ Has gradients")
            else:
                print("✗ NO GRADIENTS - THIS IS THE PROBLEM BATCH!")
            model.eval()
            
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("-"*80)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
