"""
Speculative Decoding with KV Cache Fusion

This module implements speculative decoding where:
1. Draft model (small/base) generates K candidate tokens quickly
2. Target model (large/teacher) validates these tokens in parallel
3. KV cache from target model is fused into draft model for next iteration

The key innovation is using the projector to fuse target model's KV cache
into the draft model, allowing the draft to benefit from the target's knowledge.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import logging
from transformers.cache_utils import Cache, DynamicCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)  # Enable INFO logs for debugging


def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    """Clone a KV cache"""
    new_cache = DynamicCache()
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        new_cache.key_cache.append(k.clone().detach())
        new_cache.value_cache.append(v.clone().detach())
    return new_cache


def slice_kv_cache(kv_cache: DynamicCache, start: int, end: int) -> DynamicCache:
    """Slice a KV cache along the sequence dimension"""
    new_cache = DynamicCache()
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        # Shape: (B, H, N, D) -> slice N dimension
        new_cache.key_cache.append(k[:, :, start:end, :].clone())
        new_cache.value_cache.append(v[:, :, start:end, :].clone())
    return new_cache


def append_kv_cache(kv_cache: DynamicCache, new_kv: DynamicCache) -> DynamicCache:
    """Append new KV cache to existing cache"""
    if kv_cache is None:
        return clone_kv_cache(new_kv)
    
    result = DynamicCache()
    for layer_idx in range(len(kv_cache.key_cache)):
        k_old = kv_cache.key_cache[layer_idx]
        v_old = kv_cache.value_cache[layer_idx]
        k_new = new_kv.key_cache[layer_idx]
        v_new = new_kv.value_cache[layer_idx]
        
        # Concatenate along sequence dimension (dim=2 for shape B,H,N,D)
        result.key_cache.append(torch.cat([k_old, k_new], dim=2))
        result.value_cache.append(torch.cat([v_old, v_new], dim=2))
    
    return result


def fuse_target_kv_to_draft(
    draft_kv_cache: DynamicCache,
    target_kv_cache: DynamicCache,
    projector,
    start_pos: int,
    end_pos: int,
    target_layer_idx: int,
    source_layer_idx: int,
) -> DynamicCache:
    """
    Fuse target model's KV cache into draft model's KV cache using projector
    
    Args:
        draft_kv_cache: Draft model's KV cache (will be modified in-place)
        target_kv_cache: Target model's KV cache (source for fusion)
        projector: Projector module to transform target KV to draft KV space
        start_pos: Start position in sequence to fuse
        end_pos: End position in sequence to fuse
        target_layer_idx: Which layer of draft model to update
        source_layer_idx: Which layer of target model to use as source
    
    Returns:
        Updated draft_kv_cache
    """
    # Extract KV slices
    draft_key = draft_kv_cache.key_cache[target_layer_idx]
    draft_value = draft_kv_cache.value_cache[target_layer_idx]
    target_key = target_kv_cache.key_cache[source_layer_idx]
    target_value = target_kv_cache.value_cache[source_layer_idx]
    
    # Slice the region to fuse (B, H, N, D) -> (B, H, K, D) where K = end_pos - start_pos
    draft_key_slice = draft_key[:, :, start_pos:end_pos, :]
    draft_value_slice = draft_value[:, :, start_pos:end_pos, :]
    target_key_slice = target_key[:, :, start_pos:end_pos, :]
    target_value_slice = target_value[:, :, start_pos:end_pos, :]
    
    # Debug: print shapes
    logger.debug(f"KV fusion: start_pos={start_pos}, end_pos={end_pos}")
    logger.debug(f"draft_key shape: {draft_key.shape}, slice: {draft_key_slice.shape}")
    logger.debug(f"target_key shape: {target_key.shape}, slice: {target_key_slice.shape}")
    
    # Projector expects (B, H, N, D) format directly
    source_kv = (target_key_slice, target_value_slice)
    target_kv = (draft_key_slice, draft_value_slice)
    
    fused_key, fused_value = projector.forward(source_kv, target_kv)
    
    # Update draft cache in-place
    draft_kv_cache.key_cache[target_layer_idx][:, :, start_pos:end_pos, :] = fused_key
    draft_kv_cache.value_cache[target_layer_idx][:, :, start_pos:end_pos, :] = fused_value
    
    return draft_kv_cache


class SpeculativeDecoder:
    """
    Speculative Decoder with KV Cache Fusion
    
    This class implements speculative decoding where the draft model generates
    multiple candidate tokens, the target model verifies them, and the target's
    KV cache is fused back into the draft model to improve future predictions.
    """
    
    def __init__(
        self,
        draft_model,
        target_model,
        projector_list: List,
        projector_config: dict,
        gamma: int = 4,
        decode_fusion: bool = True,
        prefill_fusion = None,  # C2C wrapper for proper prefill
    ):
        """
        Args:
            draft_model: Small/fast model for speculation (base model)
            target_model: Large/accurate model for verification (teacher model)
            projector_list: List of projector modules for KV fusion
            projector_config: Configuration dict mapping layers and projectors
            gamma: Number of tokens to speculate (K in the paper)
            decode_fusion: Whether to fuse target KV cache into draft during generation
            prefill_fusion: C2C RosettaModel wrapper (optional, for prefill-stage multi-model fusion)
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.projector_list = projector_list
        self.projector_config = projector_config
        self.gamma = gamma
        self.decode_fusion = decode_fusion
        self.prefill_fusion = prefill_fusion
        # Statistics
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.draft_calls = 0
        self.target_calls = 0
        self.accepted_lengths = []  # 记录每次accepted长度
        # Timing statistics
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.draft_time = 0.0
        self.verify_time = 0.0
        self.fusion_time = 0.0
    # 记录接受长度，接受率之类的统计数据
    def reset_stats(self):
        """Reset statistics counters"""
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.draft_calls = 0
        self.target_calls = 0
        self.accepted_lengths = []
        self.prefill_time = 0.0
        self.decode_time = 0.0
        self.draft_time = 0.0
        self.verify_time = 0.0
        self.fusion_time = 0.0
    
    def get_stats(self):
        """Get current statistics"""
        acceptance_rate = self.accepted_tokens / self.total_tokens if self.total_tokens > 0 else 0
        average_accepted_length = sum(self.accepted_lengths) / len(self.accepted_lengths) if self.accepted_lengths else 0
        return {
            "total_tokens": self.total_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": acceptance_rate,
            "average_accepted_length": average_accepted_length,
            "draft_calls": self.draft_calls,
            "target_calls": self.target_calls,
            "prefill_time": self.prefill_time,
            "decode_time": self.decode_time,
            "draft_time": self.draft_time,
            "verify_time": self.verify_time,
            "fusion_time": self.fusion_time,
        }
    
    # 生成num_tokens 个 draft token
    # TLDR: draft
    @torch.no_grad()
    def generate_draft_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[DynamicCache],
        num_tokens: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], DynamicCache]:
        candidate_ids = []
        candidate_logits = []
        current_input = input_ids
        current_past = past_key_values
        current_attention_mask = attention_mask
        for _ in range(num_tokens):
            self.draft_calls += 1
            # Forward pass on draft model
            outputs = self.draft_model(
                input_ids=current_input,
                attention_mask=current_attention_mask,
                past_key_values=current_past,
                use_cache=True,
                return_dict=True,
            )
            # sample next token
            logits = outputs.logits[:, -1, :]
            candidate_logits.append(logits)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            candidate_ids.append(next_token)
            # Update inputs for next iteration 
            current_input = next_token
            current_past = outputs.past_key_values
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((current_attention_mask.shape[0], 1), 
                          device=current_attention_mask.device,
                          dtype=current_attention_mask.dtype)
            ], dim=1)
        # list to tensor
        candidate_ids = torch.cat(candidate_ids, dim=1)
        return candidate_ids, candidate_logits, current_past

    # target model verify candidate tokens
    @torch.no_grad()
    def verify_candidates(
        self,
        prefix_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[DynamicCache],
    ) -> Tuple[int, torch.Tensor, DynamicCache, torch.Tensor]:
        """
        Verify candidate tokens using target model.

        Args:
            prefix_ids (torch.Tensor): Tokens before candidates (batch, prefix_len)
            candidate_ids (torch.Tensor): Candidate tokens to verify (batch, K)
            attention_mask (torch.Tensor): Attention mask for full sequence
            past_key_values (Optional[DynamicCache]): Target model's past KV cache

        Returns:
            Tuple[int, torch.Tensor, DynamicCache, torch.Tensor]:
                num_accepted: Number of accepted tokens
                next_token: Next token to add (either from candidates or resampled)
                target_kv_cache: Updated target KV cache
                target_logits: Logits from target model for accepted region
        """
        
        self.target_calls += 1
        
        # Concatenate prefix and candidates for parallel verification
        # Shape: (batch, prefix_len + K)
        # 合并获得完整input token
        full_input = torch.cat([prefix_ids, candidate_ids], dim=1)
        
        # Forward pass on target model (processes all K tokens in parallel)
        # target model forward
        outputs = self.target_model(
            input_ids=full_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        target_logits = outputs.logits  # (batch, prefix_len + K, vocab_size)
        target_kv_cache = outputs.past_key_values
        
        # Verify each candidate token
        # Compare target's prediction at position i with draft's token at i+1
        num_accepted = 0
        batch_size = candidate_ids.shape[0]
        prefix_len = prefix_ids.shape[1]
        
        # 遍历draft token里的每一步
        for i in range(candidate_ids.shape[1]):
            # Target's prediction at position (prefix_len + i - 1)
            # corresponds to candidate at position i
            if prefix_len + i > 0: # 据说是为处理空前缀产生的问题
                # greedy sample 当前位置的token
                target_pred_logits = target_logits[:, prefix_len + i - 1, :]
                target_pred_token = torch.argmax(target_pred_logits, dim=-1)
                candidate_token = candidate_ids[:, i]
                
                # Check if candidate matches target's prediction
                # 判断从target model解的logsitics和draft model生成的token是否一致
                if torch.all(target_pred_token == candidate_token):
                    num_accepted += 1
                else:
                    # Rejection sampling: sample from target distribution
                    # This is a simplified version; full spec decoding uses
                    # more sophisticated rejection sampling
                    next_token = target_pred_token.unsqueeze(1)
                    break
        else:
            # All candidates accepted, sample one more token from target
            last_logits = target_logits[:, -1, :]
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            num_accepted = candidate_ids.shape[1]
        
        # Update statistics
        self.total_tokens += candidate_ids.shape[1]
        self.accepted_tokens += num_accepted
        self.accepted_lengths.append(num_accepted)
        
        return num_accepted, next_token, target_kv_cache, target_logits
    
    @torch.no_grad()
    def fuse_kv_caches(
        self,
        draft_kv_cache: DynamicCache,
        target_kv_cache: DynamicCache,
        start_pos: int,
        num_tokens: int,
    ) -> DynamicCache:
        """
        Fuse target model's KV cache into draft model's KV cache
        
        Args:
            draft_kv_cache: Draft model's KV cache
            target_kv_cache: Target model's KV cache
            start_pos: Starting position for fusion
            num_tokens: Number of tokens to fuse
        
        Returns:
            Updated draft_kv_cache
        """
        if not self.decode_fusion or not self.projector_config:
            return draft_kv_cache
        
        # 计算需要project的kv
        end_pos = start_pos + num_tokens
        
        # Iterate through projector configuration
        # projector_config structure: {target_idx: {source_idx: {target_layer: [(source_layer, proj_idx)]}}}
        # For speculative decoding, we assume:
        # - draft_model is at base_model_idx (typically 0)
        # - target_model is at index 1
        
        base_idx = 0  # draft model index
        target_idx = 1  # target model index
        
        # 打印信息，处理conifg
        logger.info(f"fuse_kv_caches called: start_pos={start_pos}, num_tokens={num_tokens}, end_pos={start_pos+num_tokens}")
        logger.info(f"Draft KV cache: {len(draft_kv_cache.key_cache) if draft_kv_cache else 0} layers")
        logger.info(f"Target KV cache: {len(target_kv_cache.key_cache) if target_kv_cache else 0} layers")
        
        if draft_kv_cache and len(draft_kv_cache.key_cache) > 0:
            logger.info(f"Draft cache first layer shape: {draft_kv_cache.key_cache[0].shape}")
        if target_kv_cache and len(target_kv_cache.key_cache) > 0:
            logger.info(f"Target cache first layer shape: {target_kv_cache.key_cache[0].shape}")
        
        if base_idx not in self.projector_config:
            logger.warning(f"No projector config for base model (idx={base_idx})")
            return draft_kv_cache
        
        if target_idx not in self.projector_config[base_idx]:
            logger.warning(f"No projector config from target model (idx={target_idx}) to base")
            return draft_kv_cache
        
        # Apply projections for each configured layer
        logger.info(f"Applying projections for layers: {list(self.projector_config[base_idx][target_idx].keys())}")
        # Iterate through each draft layer to update
        for draft_layer_idx, layer_config in self.projector_config[base_idx][target_idx].items():
            for target_layer_idx, projector_idx in layer_config:
                projector = self.projector_list[projector_idx]
                logger.info(f"Fusing: target_layer={target_layer_idx} -> draft_layer={draft_layer_idx} using projector {projector_idx}")
                
                draft_kv_cache = fuse_target_kv_to_draft(
                    draft_kv_cache=draft_kv_cache,
                    target_kv_cache=target_kv_cache,
                    projector=projector,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    target_layer_idx=draft_layer_idx,
                    source_layer_idx=target_layer_idx,
                )
        
        return draft_kv_cache
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Generate tokens using speculative decoding with KV fusion
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            return_stats: Whether to return generation statistics
        
        Returns:
            generated_ids: Generated token sequence (batch, seq_len + gen_len)
            stats: Generation statistics (if return_stats=True)
        """
        import time
        
        self.reset_stats()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # Initialize with prompt
        current_ids = input_ids
        current_attention_mask = attention_mask
        
        # Initialize KV caches with prefill
        logger.info("="*80)
        logger.info(f"Starting prefill with prefill_fusion={'YES' if self.prefill_fusion else 'NO'}")
        
        # Start timing prefill
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        
        if self.prefill_fusion is not None:
            # Use C2C wrapper for proper multi-model prefill with KV fusion
            # Create kv_cache_index for prefill
            seq_len = input_ids.shape[1]
            logger.info(f"Input sequence length: {seq_len}")
            
            instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(
                seq_len - 1, 1
            ).unsqueeze(0).to(device)
            label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
            kv_cache_index = [instruction_index, label_index]
            
            logger.info(f"Created kv_cache_index with {len(kv_cache_index)} sections")
            
            # Perform prefill through prefill_fusion wrapper to get draft model's cache
            logger.info("Calling prefill_fusion.forward() for draft model...")
            wrapper_outputs = self.prefill_fusion(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kv_cache_index=kv_cache_index,
                use_cache=True,
            )
            logger.info(f"Prefill fusion forward completed")
            
            # Get draft (base) model's cache from prefill_fusion wrapper
            base_idx = self.prefill_fusion.base_model_idx
            logger.info(f"Base model index: {base_idx}")
            logger.info(f"Prefill fusion kv_cache_dict keys: {list(self.prefill_fusion.kv_cache_dict.keys())}")
            
            if base_idx in self.prefill_fusion.kv_cache_dict:
                logger.info(f"kv_cache_dict[{base_idx}] keys: {list(self.prefill_fusion.kv_cache_dict[base_idx].keys())}")
                if base_idx in self.prefill_fusion.kv_cache_dict[base_idx]:
                    draft_kv_cache = self.prefill_fusion.kv_cache_dict[base_idx][base_idx]
                    if draft_kv_cache:
                        logger.info(f"✓ Got draft KV cache: {len(draft_kv_cache.key_cache)} layers, seq_len={draft_kv_cache.key_cache[0].shape[2] if len(draft_kv_cache.key_cache) > 0 else 0}")
                    else:
                        logger.warning("✗ Draft KV cache is None")
                        draft_kv_cache = None
                else:
                    logger.warning(f"✗ Key {base_idx} not in kv_cache_dict[{base_idx}]")
                    draft_kv_cache = None
            else:
                logger.warning(f"✗ Base index {base_idx} not in kv_cache_dict")
                draft_kv_cache = None
            
            if draft_kv_cache is None:
                logger.warning("Could not get draft KV cache from prefill_fusion wrapper, using direct call")
                draft_outputs = self.draft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                draft_kv_cache = draft_outputs.past_key_values
                logger.info(f"Generated draft KV cache via direct call: {len(draft_kv_cache.key_cache)} layers")
            
            # For target model: prefill_fusion wrapper only generates teacher cache for non-final sections
            # For speculative decoding inference, we need to explicitly generate it
            logger.info("Generating target model KV cache explicitly for speculative decoding...")
            target_outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            target_kv_cache = target_outputs.past_key_values
            if target_kv_cache and len(target_kv_cache.key_cache) > 0:
                logger.info(f"✓ Generated target KV cache: {len(target_kv_cache.key_cache)} layers, seq_len={target_kv_cache.key_cache[0].shape[2]}")
                logger.info(f"  Target cache first layer key shape: {target_kv_cache.key_cache[0].shape}")
                logger.info(f"  Target cache first layer value shape: {target_kv_cache.value_cache[0].shape}")
            else:
                logger.error("✗✗✗ FAILED to generate target KV cache!")
            
            logger.info("="*80)
        else:
            # Original implementation: direct model calls (may not work with C2C fusion)
            draft_outputs = self.draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            draft_kv_cache = draft_outputs.past_key_values
            
            target_outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            target_kv_cache = target_outputs.past_key_values
        
        # End timing prefill
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        self.prefill_time = prefill_end - prefill_start
        
        generated_tokens = 0
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Start timing decode phase
        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        
        while generated_tokens < max_new_tokens:
            # Step 1: Draft model generates K candidates
            num_candidates = min(self.gamma, max_new_tokens - generated_tokens)
            
            # Use last token as input for draft generation
            last_token = current_ids[:, -1:]
            
            torch.cuda.synchronize()
            draft_start = time.perf_counter()
            candidate_ids, candidate_logits, new_draft_kv = self.generate_draft_candidates(
                input_ids=last_token,
                attention_mask=current_attention_mask,
                past_key_values=draft_kv_cache,
                num_tokens=num_candidates,
            )
            torch.cuda.synchronize()
            draft_end = time.perf_counter()
            self.draft_time += draft_end - draft_start
            
            # Step 2: Target model verifies candidates
            torch.cuda.synchronize()
            verify_start = time.perf_counter()
            # When using past_key_values, attention_mask should cover both past and new tokens
            # past_kv has seq_len, new tokens are 1 (prefix) + num_candidates
            past_seq_len = target_kv_cache.key_cache[0].shape[2] if target_kv_cache else 0
            verify_attention_mask = torch.ones((batch_size, past_seq_len + 1 + num_candidates), device=device, dtype=torch.long)
            num_accepted, next_token, new_target_kv, target_logits = self.verify_candidates(
                prefix_ids=last_token,
                candidate_ids=candidate_ids,
                attention_mask=verify_attention_mask,
                past_key_values=target_kv_cache,
            )
            torch.cuda.synchronize()
            verify_end = time.perf_counter()
            self.verify_time += verify_end - verify_start
            
            # Step 3: Update sequences with accepted tokens
            accepted_candidates = candidate_ids[:, :num_accepted]
            new_tokens = torch.cat([accepted_candidates, next_token], dim=1)
            current_ids = torch.cat([current_ids, new_tokens], dim=1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((batch_size, new_tokens.shape[1]), device=device, dtype=torch.long)
            ], dim=1)
            
            # Step 4: Fuse target KV cache into draft KV cache
            # Only fuse the accepted portion
            if num_accepted > 0:
                # Fusion positions should be relative to the NEW caches
                # new_target_kv has: old_target_len + 1 (prefix) + num_candidates
                # new_draft_kv has: old_draft_len + num_candidates  
                # We want to fuse the accepted tokens from target to draft
                # The accepted tokens are at positions: old_len+1 to old_len+1+num_accepted in target cache
                old_target_len = target_kv_cache.key_cache[0].shape[2]
                # In new_target_kv, the tokens are at positions [old_target_len+1 : old_target_len+1+num_accepted]
                fusion_start = old_target_len + 1
                fusion_end = fusion_start + num_accepted
                
                logger.info(f"Fusion range: [{fusion_start}:{fusion_end}] (num_accepted={num_accepted})")
                logger.info(f"new_target_kv length: {new_target_kv.key_cache[0].shape[2]}")
                logger.info(f"new_draft_kv length: {new_draft_kv.key_cache[0].shape[2]}")
                
                torch.cuda.synchronize()
                fusion_start_time = time.perf_counter()
                draft_kv_cache = self.fuse_kv_caches(
                    draft_kv_cache=new_draft_kv,
                    target_kv_cache=new_target_kv,
                    start_pos=fusion_start,
                    num_tokens=num_accepted,
                )
                torch.cuda.synchronize()
                fusion_end_time = time.perf_counter()
                self.fusion_time += fusion_end_time - fusion_start_time
            else:
                # No acceptance, rollback to original length + prefix token
                old_draft_len = draft_kv_cache.key_cache[0].shape[2]
                draft_kv_cache = slice_kv_cache(new_draft_kv, 0, old_draft_len + 1)
            
            # Update target cache
            target_kv_cache = new_target_kv
            
            # Update generation counter
            generated_tokens += new_tokens.shape[1]
            
            # Check for EOS
            if eos_token_id is not None:
                finished |= (next_token.squeeze(1) == eos_token_id)
                if torch.all(finished):
                    break
        
        # End timing decode phase
        torch.cuda.synchronize()
        decode_end = time.perf_counter()
        self.decode_time = decode_end - decode_start
        
        if return_stats:
            stats = self.get_stats()
            stats['speedup'] = stats['accepted_tokens'] / stats['target_calls'] if stats['target_calls'] > 0 else 1.0
            return current_ids, stats
        
        return current_ids


# Baseline: SpeculativeDecoder without KV fusion
class SpeculativeDecoderNoKV(SpeculativeDecoder):
    """Baseline speculative decoder without KV cache fusion"""
    
    def __init__(self, *args, **kwargs):
        kwargs['decode_fusion'] = False
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def fuse_kv_caches(
        self,
        draft_kv_cache: DynamicCache,
        target_kv_cache: DynamicCache,
        start_pos: int,
        num_tokens: int,
    ) -> DynamicCache:
        """No-op fusion for baseline - just return draft cache unchanged"""
        return draft_kv_cache
