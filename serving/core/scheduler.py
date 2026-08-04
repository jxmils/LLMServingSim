import bisect
import pandas as pd
from time import time
import csv
import os

from .request import *
from .utils import *
from .controller import *
from .memory_model import *
from .graph_generator import *
from .trace_generator import *
from .logger import print_markup, print_rule
from .pim_model import *
import numpy as np

# class that shedules request of astra-sim
class Scheduler:
    def __init__(self, model, node_id, instance_id, max_num_seqs, max_num_batched_tokens,
                 num_npus, tp_size, pp_size, npu_mem, cpu_mem,
                 start_npu, pd_type, fp, block_size, req_num,
                 prioritize_prefill, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, enable_chunked_prefill=False,
                 long_prefill_token_threshold=0, cxl_mem=0, ep_size=1, kv_cache_dtype='auto'):
        self.model = model
        self.config = get_config(model)
        self.node_id = node_id
        self.instance_id = instance_id
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = min(max_num_batched_tokens, self.config['max_position_embeddings'])
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.req_num = req_num
        self.start_npu = start_npu
        self.pd_type = pd_type
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefix_storage = prefix_storage
        self.prioritize_prefill = prioritize_prefill
        # lists are sorted in arrival time manner
        self.request = []
        self.inflight = []
        self.done = []
        self.batch_ids = -1

        # memory model
        self.memory = MemoryModel(model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem, ep_size=ep_size, pp_size=pp_size, kv_cache_dtype=kv_cache_dtype)

        # logger
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)
    
 
    def schedule(self, current, sys, batch_id=-1):
        if self.enable_prefix_caching:
            return self.schedule_with_prefix(current, sys, batch_id)
        else:
            return self.schedule_base(current, sys, batch_id)

    def _get_reload_size(self, batch_req, batch_len):
        load_size = 0
        for req in batch_req[:batch_len]:
            if req.evict:
                load_size += self.memory.get_evict_kv(req)
        return load_size

    # batch the request scheduling method
    def schedule_base(self, current, sys, batch_id=-1):
        # first NPU to process new batch
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.pp_size:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]

            # max_num_seqs limits total running requests (vLLM behavior)
            running_reqs = sum(len(b.requests) for b in self.inflight)
            available_slots = max(0, int(self.max_num_seqs) - running_reqs)
            batch_len = min(len(batch_req), available_slots)

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            kv_size = 0
            evict_size = 0

            # Get decode requests for preemption decisions
            gen_req = [req for req in batch_req if not req.is_prefill()]
            
            if self.prioritize_prefill and not self.enable_chunked_prefill:
                prefill_req = [req for req in batch_req if req.is_prefill()]

                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = min(len(batch_req), available_slots)
                    batch_req = batch_req[:batch_len]
            
            # Chunked prefill: process decode requests first, then prefill requests
            if self.enable_chunked_prefill:
                prefills = [req for req in batch_req if req.is_prefill()]
                decodes = [req for req in batch_req if not req.is_prefill()]
                batch_req = decodes + prefills
                batch_len = len(batch_req)
            
            # ============ STEP 1: Token budget allocation (FIRST) ============
            # Build scheduled_tokens dict: req.id -> tokens to process this step
            scheduled_tokens = {}
            
            if self.enable_chunked_prefill:
                # vLLM-style chunked prefill: schedule running (decode + ongoing prefill)
                # first, then waiting (new prefill) requests. Token budget is the main
                # constraint; long_prefill_token_threshold caps per-request tokens per step.
                token_budget = self.max_num_batched_tokens
                new_batch_req = []
                threshold = self.long_prefill_token_threshold
                # Decode requests first (each decode request = 1 token)
                for req in batch_req:
                    if not req.is_prefill():
                        if token_budget <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                # Then prefill requests (chunked)
                for req in batch_req:
                    if req.is_prefill():
                        if token_budget <= 0:
                            break
                        remaining = req.original_input - req.num_computed_tokens
                        # Per-request cap: long_prefill_token_threshold
                        if 0 < threshold < remaining:
                            remaining = threshold
                        chunk = min(remaining, token_budget)
                        if chunk <= 0:
                            break
                        req.chunk_len = chunk
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = chunk
                        token_budget -= chunk
                batch_req = new_batch_req
                batch_len = len(batch_req)

            else:
                # Non-chunked: compute scheduled tokens for each request
                total_len = 0
                for req in batch_req:
                    if req.is_prefill():
                        scheduled_tokens[req.id] = req.input
                        total_len += req.input
                    else:
                        scheduled_tokens[req.id] = 1
                        total_len += 1

                while total_len > self.max_num_batched_tokens:
                    # print(f"[NON_CHUNKED] total_len({total_len} = sum([req 0 ~ {batch_len - 1}])) exceed 'max_num_batched_tokens'")
                    last_req = batch_req[-1]
                    total_len -= scheduled_tokens[last_req.id]
                    del scheduled_tokens[last_req.id]
                    batch_req = batch_req[:-1]
                    batch_len -= 1
                
                # DEBUG: Check if total_len reached max
                # if total_len >= self.max_num_batched_tokens * 0.9:
                #     print(f"[NON-CHUNKED] Near max tokens! total_len: {total_len}/{self.max_num_batched_tokens}")
                #     print(f"              Batch: {batch_len} reqs, scheduled_tokens: {scheduled_tokens}")
            
                # Early return due to max_num_batched_tokens limitation (It occurs only when No chunked-prefill)
                if batch_len == 0:
                    print("     [WARNNING] Cannot load the request to batch due to max_num_batched_tokens limitation")
                    return None
            # ============ STEP 2: KV size calculation (with scheduled_tokens) ============
            temp_len = batch_len
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                load_size = self._get_reload_size(batch_req, i)
                if self.memory.is_avail(kv_size + load_size, Device.NPU):
                    temp_len = i
                    break
            
            # ============ STEP 3: Eviction if needed ============
            while temp_len == 0:
                # print("Evict Request to CPU due to memory limitation")
                # preempt request one by one until there is enough space
                if len(gen_req) == 0:
                    return None
                
                # check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue

                # else
                req_to_evict = gen_req[-1]
                evicted_kv_size = self.memory.get_evict_kv(req_to_evict)
                evict_size += evicted_kv_size
                req_to_evict.evict = True
                self.logger.info("Eviction of the request #%d", req_to_evict.id)
                gen_req = gen_req[:-1]
                # spill to cpu (host) memory. get_evict_kv returns per-rank
                # bytes; cpu_used is tracked in full-cluster bytes (matches
                # MemoryModel.apply_kv_cache_events convention), so scale by
                # num_npus when crossing the NPU->CPU boundary.
                self.memory.free(evicted_kv_size, Device.NPU)
                self.memory.allocate(evicted_kv_size * self.num_npus, Device.CPU)

                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)

                # check if can batch
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                    load_size = self._get_reload_size(batch_req, i)
                    if self.memory.is_avail(kv_size + load_size, Device.NPU):
                        temp_len = i
                        break

            batch_len = temp_len
            batch_req = batch_req[:batch_len]

            # Recompute kv_size for final batch
            kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            load_size = self._get_reload_size(batch_req, batch_len)

            # delete from request queue
            for req in batch_req:
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                if req.evict:
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            # ============ STEP 4: Allocate memory ============
            if kv_size > 0:
                self.memory.allocate(kv_size, Device.NPU)

            # Reload evicted KV to NPU and remove the spilled copy from CPU.
            # load_size is per-rank, cpu_used is full-cluster.
            if load_size > 0:
                self.memory.allocate(load_size, Device.NPU)
                self.memory.free(load_size * self.num_npus, Device.CPU)
            
            # ============ STEP 5: Build batch with lists ============
            total_len = 0
            kv_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            for req in batch_req:
                if req.is_prefill():
                    # Use scheduled_tokens for chunk size
                    chunk_size = scheduled_tokens.get(req.id, req.original_input - req.num_computed_tokens)

                    total_len += chunk_size
                    if req.is_init:  # Only set queuing delay on first chunk
                        req.set_que_delay(current)
                    q_list.append(chunk_size)
                    prefill_q_list.append(chunk_size)
                    # prefill_k_list: already computed tokens (k_cache from previous chunks)
                    prefill_k_list.append(req.num_computed_tokens)
                    # k_list: total kv cache after this step (computed + new)
                    # k_list.append(req.num_computed_tokens + chunk_size)
                    num_prefill += 1

                else:
                    # Decode
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.num_computed_tokens
                    decode_k_list.append(req.num_computed_tokens)
                    # k_list.append(req.num_computed_tokens)

            # make batch, output doesn't matter here!! always one iteration
            # batch is also 1
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, load_size)
            # add already fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            # print(f"[BATCH DEBUG] Batch: {len(new_batch_req)} reqs, scheduled_tokens: {scheduled_tokens}")
            # batch.log()
            # add scheduled_tokens to batch for debugging
            batch.scheduled_tokens = scheduled_tokens
            return batch
        
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch == None:
                    return None
                # check if this has been runned in the system
                if sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
    
    def schedule_with_prefix(self, current, sys, batch_id=-1):
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.pp_size:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]

            # max_num_seqs limits total running requests (vLLM behavior)
            running_reqs = sum(len(b.requests) for b in self.inflight)
            available_slots = max(0, int(self.max_num_seqs) - running_reqs)
            batch_len = min(len(batch_req), available_slots)

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            # Prioritize prefill (without chunked prefill) or reorder for chunked prefill
            if self.prioritize_prefill and not self.enable_chunked_prefill:
                prefill_req = [req for req in batch_req if req.is_prefill()]
                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = min(len(batch_req), available_slots)
                    batch_req = batch_req[:batch_len]
            
            # Chunked prefill: process decode requests first, then prefill requests
            if self.enable_chunked_prefill:
                prefills = [req for req in batch_req if req.is_prefill()]
                decodes = [req for req in batch_req if not req.is_prefill()]
                batch_req = decodes + prefills
                batch_len = len(batch_req)

            # Get decode requests for preemption decisions
            gen_req = [req for req in batch_req if not req.is_prefill()]
            # gen_req = [req for req in batch_req if not (req.num_computed_tokens >= req.original_input)]
            
            # ============ STEP 0: Prefix Matching ============
            # Only match prefix for NEW prefill requests (first chunk)
            # Ongoing chunked prefills already have their prefix cache info
            # for req in batch_req:
            #     if req.is_prefill():
            #         self.memory.prefix_match(req)
            
            # ============ STEP 1: Token budget allocation ============
            scheduled_tokens = {}
            
            if self.enable_chunked_prefill:
                # Chunked prefill: assign token budget to requests
                token_budget = self.max_num_batched_tokens
                new_batch_req = []
                
                # Decode requests first (each decode request = 1 token)
                for req in batch_req:
                    if not req.is_prefill():
                        if token_budget <= 0:
                            break
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = 1
                        token_budget -= 1
                
                # Then prefill requests (chunked)
                threshold = self.long_prefill_token_threshold
                for req in batch_req:
                    if req.is_prefill():
                        if token_budget <= 0:
                            break
                        # Calculate remaining tokens without considering prefix cache
                        # because it is already considered in "self.memory.prefix_match(req)" -> req.num_computed_tokens
                        if req.num_computed_tokens == 0:
                            self.memory.prefix_match(req)
                        remaining = req.original_input - req.num_computed_tokens
                        # Per-request cap: long_prefill_token_threshold
                        if 0 < threshold < remaining:
                            remaining = threshold
                        chunk = min(remaining, token_budget)
                        if chunk <= 0:
                            break

                        req.chunk_len = chunk
                        new_batch_req.append(req)
                        scheduled_tokens[req.id] = chunk
                        token_budget -= chunk

                batch_req = new_batch_req
                batch_len = len(batch_req)
            else:
                # Non-chunked: compute scheduled tokens for each request
                total_len = 0
                for req in batch_req:
                    if req.is_prefill():
                        if req.num_computed_tokens == 0:
                            self.memory.prefix_match(req)
                        # Consider prefix cache hit for non-chunked prefill
                        prefix_hit = req.prefix_cache_hit
                        tokens_to_compute = max(req.original_input - prefix_hit, 1)
                        scheduled_tokens[req.id] = tokens_to_compute
                        req.chunk_len = tokens_to_compute  # Set chunk_len for add_done()
                        total_len += tokens_to_compute
                    else:
                        scheduled_tokens[req.id] = 1
                        total_len += 1

                while total_len > self.max_num_batched_tokens:
                    last_req = batch_req[-1]
                    total_len -= scheduled_tokens[last_req.id]
                    del scheduled_tokens[last_req.id]
                    batch_req = batch_req[:-1]
                    batch_len -= 1
            
            # ============ STEP 1.5: Lock prefix for scheduled requests ============
            newly_locked = set()
            for req in batch_req:
                # if req.is_prefill() and req.num_computed_tokens == 0:
                if req.is_prefill() and req.npu_last_node is not None and not req._prefix_locked:
                    self.memory.lock_prefix(req, Device.NPU)
                    req._prefix_locked = True
                    newly_locked.add(req.id)
            
            # ============ STEP 2: KV size calculation ============
            kv_size = 0
            evict_size = 0
            temp_len = batch_len
            total_useable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)
            
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                if total_useable_size >= kv_size:
                    temp_len = i
                    break
            
            # ============ STEP 3: Eviction if needed ============
            evicted_req = []
            while temp_len == 0:
                # print("eviction occurs!!")
                if len(gen_req) == 0:
                    # print("gen_req length == 0 (No decode) => return None (No Batch)")
                    # No request to evict but no memory - rollback prefix cache lock
                    for req in batch_req:
                        if req.is_prefill() and req._prefix_locked:
                            
                            self.memory.unlock_prefix(req, Device.NPU)
                            self.memory.erase_prefix_info(req)
                            req._prefix_locked = False
                    return None
                
                # Check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue
                
                # Evict the last decode request
                # (DEPRECATED) self.memory.unlock_prefix(gen_req[-1], Device.NPU)
                # (DEPRECATED) self.memory.erase_prefix_info(gen_req[-1])
                if gen_req[-1].is_prefill() and getattr(gen_req[-1], '_prefix_locked', False):
                    self.memory.unlock_prefix(gen_req[-1], Device.NPU)
                    # self.memory.erase_prefix_info(gen_req[-1])
                    gen_req[-1]._prefix_locked = False
                
                current_usable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)
                
                gen_req[-1].evict = True
                evicted_req.append(gen_req[-1])
                self.logger.info("Eviction of the request #%d", gen_req[-1].id)
                gen_req = gen_req[:-1]
                
                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)
                
                # Check if can batch now
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i, scheduled_tokens)
                    if current_usable_size >= kv_size:
                        temp_len = i
                        break

            # Unlock prefix for requests that didn't make it into the batch
            for req in batch_req[temp_len:]:
                if req.is_prefill() and req._prefix_locked:
                    self.memory.unlock_prefix(req, Device.NPU)
                    self.memory.erase_prefix_info(req)
                    req._prefix_locked = False

            batch_len = temp_len
            batch_req = batch_req[:batch_len]
            
            # Recompute kv_size for final batch
            kv_size = self.memory.get_block_kv(batch_req, batch_len, scheduled_tokens)
            evict_size = (kv_size - self.memory.avail_size(Device.NPU)) if kv_size > self.memory.avail_size(Device.NPU) else 0
            
            if evict_size > 0:
                self.memory.evict_prefix_cache(evict_size, Device.NPU)

            # ============ STEP 4: Allocate memory & handle evicted requests ============
            evict_load_size = 0
            prefix_load_size = 0
            
            for req in batch_req:
                # Remove from request queue
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                # Load prefix cache from storage if needed
                if req.is_prefill() and req.storage_cache_hit > req.npu_cache_hit:
                    prefix_load_size += (req.storage_cache_hit - req.npu_cache_hit) * self.memory.get_kv(1)

                # Handle evicted requests
                if req.evict:
                    self.memory.prefix_match(req)
                    self.memory.lock_prefix(req, Device.NPU)
                    if self.prefix_storage is not None:
                        self.memory.unlock_prefix(req, Device.CPU)
                    evict_load_size += self.memory.get_evict_kv(req)
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            # ============ STEP 5: Build batch with lists ============
            total_len = 0
            kv_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            
            # Evict storage prefix cache if needed
            total_size = 0
            for req in batch_req:
                total_size += self.memory.get_total_kv(req) * self.num_npus
            for req in evicted_req:
                total_size += self.memory.get_total_kv(req) * self.num_npus
            
            if self.prefix_storage is not None:
                storage_evict_size = (total_size - self.memory.avail_size(self.prefix_storage)) if total_size > self.memory.avail_size(self.prefix_storage) else 0
                if storage_evict_size > 0:
                    self.memory.evict_prefix_cache(storage_evict_size, self.prefix_storage)

            for req in batch_req:
                # Update the prefix cache for incoming batch
                # NOTE: Moved to add_done() to ensure prefix cache is updated after chunk computation
                # self.memory.cache_unfinished_req(req, Device.NPU)
                # if self.prefix_storage is not None:
                #     self.memory.cache_unfinished_req(req, self.prefix_storage)
                
                if req.is_prefill():
                    # Use scheduled_tokens for chunk size. num_computed_tokens
                    # already includes any prefix-cache hit (memory_model.py
                    # bumps it on first prefix_match), so chunk_size is already
                    # the count of tokens actually computed this iteration —
                    # no further prefix-hit subtraction is needed downstream.
                    chunk_size = scheduled_tokens.get(req.id, req.original_input - req.num_computed_tokens)
                    if chunk_size > self.max_num_batched_tokens:
                        raise Exception("Chunk length exceeds max num batched tokens")

                    total_len += chunk_size
                    if req.is_init:  # Only set queuing delay on first chunk
                        req.set_que_delay(current)

                    q_list.append(chunk_size)
                    num_prefill += 1
                    prefill_q_list.append(chunk_size)
                    # prefill_k_list: already computed tokens (k_cache from previous chunks)
                    prefill_k_list.append(req.num_computed_tokens)
                else:
                    # Decode: use num_computed_tokens (inevitable modification)
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.num_computed_tokens  # inevitable modification: was req.input
                    decode_k_list.append(req.num_computed_tokens)  # inevitable modification: was req.input
                
                k_list.append(req.num_computed_tokens)  # inevitable modification: was req.input
            
            # Storage needs to hold evicted cache
            if self.prefix_storage is not None:
                for req in evicted_req:
                    self.memory.storage_cache_evicted_req(req)

            
            # For debugging
            # self.memory.npu_prefix_cache.pretty_print()
            # self.memory.npu_prefix_cache.print_prefix_info()
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, evict_load_size + prefix_load_size)
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            # print(f"[BATCH DEBUG] Batch: {len(new_batch_req)} reqs, scheduled_tokens: {scheduled_tokens}")
            batch.scheduled_tokens = scheduled_tokens
            # batch.log()
            return batch
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch is None or sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
        
    # pop inflight, add to done
    def add_done(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        end_reqs = []
        if len(self.inflight) == 0:
            return prompt_t, gen_t, end_reqs
        batch = None
        # find batch
        id -= 1
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        # no batch return
        if batch == None:
            return prompt_t, gen_t, end_reqs
        # already done
        if sys in batch.end:
            return prompt_t, gen_t, end_reqs
        else:
            # add to done system
            batch.end.append(sys)
            # check all npus are done
            if self.pd_type != "prefill":
                if self.start_npu not in batch.end or (self.start_npu + self.num_npus - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
            else:
                if self.start_npu not in batch.end or (self.start_npu + self.num_npus * 2 - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
        self.logger.info(
            "Batch #%d is done",
            batch.batch_id,
        )
                
        pool = []
        for req in batch.requests:
            # For chunked prefill, use computed tokens to determine prefill vs decode
            # Use is_prefill() method which checks num_computed_tokens < original_input
            is_prefill_req = req.is_prefill()
            
            # change phase
            if is_prefill_req:
                # Get chunk_len from scheduling step
                chunk_len = req.chunk_len if req.chunk_len > 0 else (req.original_input - req.num_computed_tokens)
                if chunk_len > self.max_num_batched_tokens:
                    raise Exception("Chunk length exceeds max num batched tokens")

                # Update num_computed_tokens
                req.num_computed_tokens += chunk_len
                req.chunk_len = 0  # Reset for next step
                
                # Check if prefill is complete
                if req.num_computed_tokens >= req.original_input:
                    # Update prefix cache before clearing is_init (for stats tracking)
                    if self.enable_prefix_caching:
                        self.memory.cache_unfinished_req(req, Device.NPU)
                        if self.prefix_storage is not None:
                            self.memory.cache_unfinished_req(req, self.prefix_storage)
                    req.is_init = False
                    # Include prefix cache hit tokens in prompt throughput
                    prompt_t += chunk_len + req.prefix_cache_hit
                    req.set_ttft(finish)
                    
                    if self.pd_type == "prefill":
                        # Prefill instance: send to decode instance
                        self.logger.info("Request #%d is prefill done", req.id)
                        self.logger.info("Request #%d is sent to decode instance", req.id)
                        # req.num_computed_tokens += 1  # First decode token was generated
                        
                        # remove kv cache here
                        if self.enable_prefix_caching:
                            self.memory.unlock_prefix(req, Device.NPU)
                        else:
                            kv_size = self.memory.get_evict_kv(req)
                            self.memory.free(kv_size, Device.NPU)

                        end_reqs.append(req)
                        continue
                    else:
                        # Non-PD: prefill complete, first output token generated
                        # The last prefill token passing through lm_head generates the first output
                        gen_t += 1
                        # req.num_computed_tokens += 1  # Count the first generated token
                        # req.set_ttft(finish)
                        # pool.append(req)
                        # continue
                else:
                    # Prefill not complete, return to pool for next chunk
                    prompt_t += chunk_len
                    # pool.append(req)
                    # continue
            else:
                # Decode phase
                if req.is_init:
                    # Full prefix cache hit: all input tokens were cached, so the
                    # request never entered the prefill-complete path where is_init
                    # is cleared. Lock the prefix node (was skipped because
                    # is_prefill() returned False during scheduling), count prefix
                    # stats once, then clear is_init.
                    if self.enable_prefix_caching:
                        if req.npu_last_node is not None and not req._prefix_locked:
                            self.memory.lock_prefix(req, Device.NPU)
                            req._prefix_locked = True
                        self.memory.cache_unfinished_req(req, Device.NPU)
                        if self.prefix_storage is not None:
                            self.memory.cache_unfinished_req(req, self.prefix_storage)
                    req.is_init = False
                    req.set_ttft(finish)
                    # Full prefix hit: count all cached tokens as prompt throughput
                    prompt_t += req.prefix_cache_hit
                gen_t += 1
                req.add_itl(finish)
                req.num_computed_tokens += 1

            # Update computed tokens for decode
            # req.num_computed_tokens += 1

            # check done
            if req.output <= req.num_computed_tokens + 1:
                # print("Request #{} is done".format(req.id))
                self.logger.info("Request #%d is done", req.id)
                # remove kv cache here
                if self.enable_prefix_caching:
                    self.memory.cache_finished_req(req, Device.NPU) # insert happens here
                    if self.prefix_storage is not None:
                        self.memory.cache_finished_req(req, Device.CPU)
                else:
                    kv_size = self.memory.get_evict_kv(req)
                    self.memory.free(kv_size, Device.NPU)
                req.add_latency(finish)
                self.done.append(req)
                end_reqs.append(req)

            # return to pool
            else:
                # print("Request #{} is not finished => go to pool".format(req.id))
                # Update prefix cache after chunk completion (moved from schedule_with_prefix())
                if self.enable_prefix_caching:
                    self.memory.cache_unfinished_req(req, Device.NPU)
                    if self.prefix_storage is not None:
                        self.memory.cache_unfinished_req(req, self.prefix_storage)
                pool.append(req)
        # return to request pool, both are already sorted with arrival_time
        if self.prioritize_prefill:
            self.request = self._merge_by_arrival_id(pool, self.request)
        else:
            self.request = pool + self.request
        del self.inflight[idx]
        del batch

        return prompt_t, gen_t, end_reqs
    

    ##### Helper Functions ######
    # get new batch id
    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    # add a request
    def add_request(self, req, is_init=True):
        new_req = Request(*(req), is_init=is_init)
        # Maintain arrival-time sort order (required by schedule_base/schedule_with_prefix)
        bisect.insort(self.request, new_req, key=lambda r: (r.arrival, r.id))
        return
    
    # add decode request to decode instance from prefill instnace
    def add_decode(self, req):
        req.instance_id = self.instance_id
        self.request.append(req)
        if self.enable_prefix_caching:
            self.memory.prefix_match(req)
            kv_size = self.memory.get_evict_kv(req)
            evict_size = max(0, kv_size - self.memory.avail_size(Device.NPU))
            if evict_size > 0:
                self.memory.evict_prefix_cache(evict_size, Device.NPU)
            self.memory.cache_unfinished_req(req, Device.NPU)
        else:
            kv_size = self.memory.get_total_kv(req)
            self.memory.allocate(kv_size, Device.NPU)
    
    # get first request's arrival time
    def get_first_arrival_time(self):
        return self.first_arrival_time if self.first_arrival_time != 0 else 1 # need to add event handler at first
    
    # merge requests in the request pool, ensuring they are sorted by arrival time
    def _merge_by_arrival_id(self, left, right):
        if not left:  
            return right
        if not right: 
            return left

        # Fast path: if ranges don't overlap, just concatenate
        if (left[-1].arrival, left[-1].id) <= (right[0].arrival, right[0].id):
            return left + right
        if (right[-1].arrival, right[-1].id) <= (left[0].arrival, left[0].id):
            return right + left

        # General merge
        i = j = 0
        out = []
        while i < len(left) and j < len(right):
            li, rj = left[i], right[j]
            if (li.arrival, li.id) <= (rj.arrival, rj.id):
                out.append(li); i += 1
            else:
                out.append(rj); j += 1
        if i < len(left):  
            out.extend(left[i:])
        if j < len(right): 
            out.extend(right[j:])
        return out
    
    # print total system request metrics (TTFT, TPOT, ITL)
    def print_result(self):
        # Extract ttft, tpot, and itl values from the completed requests
        ttft_values = [req.ttft for req in self.done]
        tpot_values = [req.tpot for req in self.done]
        itl_values = [itl for req in self.done for itl in req.itl]

        def _render(title: str, values, num_space=0):
            print_rule(f"[sim.tagline]{title}[/]")
            if not values:
                print_markup(f"No {title.split()[0]} data available")
                return
            mean = np.mean(values) / 1_000_000
            median = np.median(values) / 1_000_000
            p99 = np.percentile(values, 99) / 1_000_000
            label = title.split()[-1] if title != "Time to First Token" else "TTFT"
            # Map to the metric short-name used in the detail rows.
            short = {
                "Time to First Token": "TTFT",
                "Time per Output Token (excl. 1st token)": "TPOT",
                "Inter-token Latency": "ITL",
            }[title]
            spacing = " " * num_space
            print_markup(f"Mean {short} (ms){spacing}:                                                     {mean:.2f}")
            print_markup(f"Median {short} (ms){spacing}:                                                   {median:.2f}")
            print_markup(f"P99 {short} (ms){spacing}:                                                      {p99:.2f}")

        _render("Time to First Token", ttft_values)
        _render("Time per Output Token (excl. 1st token)", tpot_values)
        _render("Inter-token Latency", itl_values, num_space=1)

    # print each request results
    def print_request_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # check all the request is done
    def is_request_empty(self):
        if len(self.request) == 0 and len(self.inflight) == 0:
            return True
        else:
            return False
        
    # save requests information to an output file
    def save_output(self, output_file, is_append=False):
        if not os.path.isabs(output_file):
            output_file = f'../{output_file}'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            if not is_append:
                writer.writerow(['instance id', 'request id', 'model', 'input', 'output', 
                                'arrival', 'end_time', 'latency', 
                                'queuing_delay', 'TTFT', 'TPOT', 'ITL'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.instance_id,
                    req.id,
                    req.model,
                    req.input,
                    req.output - req.input,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.ttft,
                    req.tpot,
                    req.itl
                ])


def main():
    pass

if __name__ == "__main__":
    main()
