"""G5 fixture 3: capacity conservation and explicit block states."""
from serving.core.block_pool import BlockPool, BlockState, Device


def _counts(pool):
    return {s.name: n for s, n in pool.state_counts().items()}


def test_states_conserve_capacity_through_a_scripted_lifetime():
    pool = BlockPool(Device.NPU, 8, 16, 4096, enable_caching=True)
    pool.sample(0)
    assert _counts(pool)["FREE"] == 8 and pool.check_conservation()

    # reserve 3 blocks for a request: capacity charged, nothing resident
    blocks = pool.get_new_blocks(3)
    pool.sample(10)
    c = _counts(pool)
    assert c["RESERVED"] == 3 and c["FREE"] == 5 and pool.check_conservation()

    # one of them is a staging load; the batch completes -> all resident
    pool.mark_in_transfer(blocks[:1])
    assert _counts(pool)["IN_TRANSFER"] == 1
    pool.sample(20)
    pool.mark_written(blocks)
    pool.sample(30)
    c = _counts(pool)
    assert c["REFERENCED"] == 3 and c["IN_TRANSFER"] == 0 and c["RESERVED"] == 0

    # index two of them, release the request: indexed -> evictable, other -> free
    pool.cache_full_blocks([111, 222, 333], blocks, 0, 2)
    pool.free_blocks(reversed(blocks))
    pool.sample(40)
    c = _counts(pool)
    assert c["EVICTABLE"] == 2 and c["FREE"] == 6 and c["REFERENCED"] == 0
    assert pool.check_conservation()

    # a hit pins an evictable block again without charging new capacity
    hit = pool.get_cached_block(111)
    pool.touch([hit])
    assert hit.state is BlockState.REFERENCED and _counts(pool)["EVICTABLE"] == 1

    # allocating past the free blocks evicts the remaining indexed one
    fresh = pool.get_new_blocks(7)
    c = _counts(pool)
    assert c["RESERVED"] == 7 and c["REFERENCED"] == 1 and c["EVICTABLE"] == 0 and c["FREE"] == 0
    assert pool.check_conservation()
    pool.sample(50)

    # integrals: reserved bytes were 3 blocks for t in [10,20), 2 for [20,30)
    # (one went in transfer) and 7 for [50,60)
    integ = pool.ledger_integrals(end_ns=60)
    assert integ["reserved_byte_ns"] == 3 * 4096 * 10 + 2 * 4096 * 10 + 7 * 4096 * 10
    assert integ["in_transfer_byte_ns"] == 1 * 4096 * 10          # [20,30)
    assert integ["evictable_byte_ns"] == 2 * 4096 * 10             # [40,50)
    assert sum(pool.state_counts().values()) == 8


def test_write_ledger_files(tmp_path):
    from serving.core.kv_cache_manager import TieredKVCacheManager
    npu = BlockPool(Device.NPU, 4, 16, 1024, enable_caching=True)
    cpu = BlockPool(Device.CPU, 2, 256, 16 * 1024, enable_caching=True)
    m = TieredKVCacheManager(16, npu, [cpu], enable_caching=True)
    m.sample(0)
    npu.get_new_blocks(2)
    m.sample(100)
    summary = m.write_ledger(str(tmp_path), 0, end_ns=200)
    assert set(summary) == {"inst0_npu", "inst0_cpu", "exit_check"}
    assert summary["exit_check"]["inst0_npu"]["pinned_blocks"] == 2 and not summary["exit_check"]["inst0_npu"]["all_free"]
    assert summary["inst0_npu"]["reserved_byte_ns"] == 2 * 1024 * 100
    assert (tmp_path / "inst0_npu.csv").read_text().splitlines()[0].startswith("t_ns,free_blocks,reserved_blocks")
    assert m.check_conservation()
