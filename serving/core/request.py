# class that manages request of astra-sim
class Request:
    def __init__(self, id, model, input, output, arrival, instance_id, input_hash_ids=None, output_hash_ids=None, is_init=True):
        self.id = id
        self.model = model
        self.input = input  # Always keep original input length
        self.output = output
        self.arrival = arrival
        self.instance_id = instance_id
        self.is_init = is_init
        self.original_input = input
        self.num_computed_tokens = 0  # Tracks actual computed tokens (vLLM style)
        self.evict = False
        self.end_time = -1
        self.latency = -1
        self.queuing_delay = -1
        self.ttft = -1
        self.tpot = -1
        self.itl = []
        self.recent_end = 0

        # For chunked prefill
        self.chunk_len = 0  # tokens scheduled for this request in the current step

        # For prefix caching modeling
        self.input_hash_ids = input_hash_ids
        self.output_hash_ids = output_hash_ids
        self.prefix_cache_hit = 0
        self.npu_cache_hit = 0
        self.storage_cache_hit = 0
        self.npu_last_node = None
        self.cpu_last_node = None
        self.storage_last_node = None

        # For prefix cache lock tracking
        self._prefix_locked = False
        self._prefix_npu_stats_counted = False
        self._prefix_storage_stats_counted = False

        # For agentic session tracking (informational, does not drive scheduling)
        self.session_id = None
        self.sub_request_index = None

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def add_latency(self, end_time):
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        self.input = self.original_input
        if self.output == self.input + 1:
            self.tpot = 0
        else:
            self.tpot = (self.latency - self.ttft) // (self.output - self.input - 1)
    
    def add_itl(self, current): # 
        self.itl.append(current - self.recent_end)
        self.recent_end = current

    def set_que_delay(self, current):
        self.queuing_delay = current - self.arrival
    
    def set_ttft(self, current):
        self.ttft = current - self.arrival
        self.recent_end = current
    
    def log(self):
        print("         scheduled request : {}".format(self.__dict__))
    
    def is_prefill(self):
        """Check if request is still in prefill phase (has tokens left to compute)"""
        return self.num_computed_tokens < self.original_input

# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, batch_time, kv_size, evict=0, load=0):
        self.batch_id = batch_id
        self.model = model
        self.total_len = total_len
        self.kv_len = kv_len
        self.batch_time = batch_time
        self.fired = [] # systems that fired this batch
        self.requests = []
        self.end = []
        # vllm
        self.kv_size = kv_size
        self.evict = evict
        self.load = load
        # for attn prediction
        self.q_list = q_list
        self.k_list = k_list
        self.num_prefill = num_prefill
        self.num_decode = num_decode
        self.prefill_q_list = prefill_q_list
        self.prefill_k_list = prefill_k_list
        self.decode_k_list = decode_k_list

        # for debugging
        self.scheduled_tokens = None
    def log(self):
        print("-------------------------Batch Log------------------------")
        for key in self.__dict__.keys():
            if key == 'requests':
                continue
            print("         {} : {}".format(key, self.__dict__[key]))
        for req in self.requests:
            req.log()
        print("----------------------------------------------------------")
    