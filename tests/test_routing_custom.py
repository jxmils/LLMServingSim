"""CUSTOM routing honours a request row's instance_id (plan §9 B imbalance)."""
import json

from serving.core.router import Router


class _Sched:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.pd_type = None
        self.waiting, self.running = [], []
        self.max_num_seqs = 8


def test_custom_select_uses_row_instance_id():
    scheds = [_Sched(0), _Sched(1), _Sched(2)]
    r = Router(3, scheds, 0, routing_policy="CUSTOM")
    assert r._select_instance(scheds, "prefill", {"instance_id": 1}) == 1
    assert r._select_instance(scheds, "prefill", {"instance_id": 5}) == 2      # modulo
    a = r._select_instance(scheds, "prefill", {})                              # no field: round robin
    b = r._select_instance(scheds, "prefill", None)
    assert (a, b) == (0, 1)


def test_other_policies_ignore_the_field():
    scheds = [_Sched(0), _Sched(1)]
    r = Router(2, scheds, 0, routing_policy="RR")
    assert [r._select_instance(scheds, "prefill", {"instance_id": 1}) for _ in range(3)] == [0, 1, 0]


def test_loader_keeps_instance_id():
    rows = [{"input_toks": 4, "output_toks": 2, "arrival_time_ns": 0, "input_tok_ids": [1, 2, 3, 4], "instance_id": 1},
            {"input_toks": 4, "output_toks": 2, "arrival_time_ns": 1}]
    scheds = [_Sched(0), _Sched(1)]
    r = Router(2, scheds, 0, routing_policy="CUSTOM")
    for row in rows:
        r._load_flat_request(row, True)
    assert r._pending_requests[0]["instance_id"] == 1 and "instance_id" not in r._pending_requests[1]
