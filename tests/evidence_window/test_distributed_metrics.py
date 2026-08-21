from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from evidence_window.metrics import reduce_sufficient_statistics


def _reduce_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        local_numerator = torch.tensor(float(2 + 2 * rank))
        reduced = reduce_sufficient_statistics(local_numerator, torch.tensor(1.0))
        assert reduced.numerator.item() == 6.0
        assert reduced.count.item() == 2.0
        assert reduced.mean.item() == 3.0
    finally:
        dist.destroy_process_group()


def test_sufficient_statistics_reduce_before_mean(tmp_path):
    """Ranks reduce numerator and count, avoiding an average-of-averages error."""
    init_file = Path(tmp_path) / "gloo-init"
    mp.spawn(_reduce_worker, args=(str(init_file),), nprocs=2, join=True)
