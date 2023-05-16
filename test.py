import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam


# On each spawned worker
def worker(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29499"
    dist.init_process_group("gloo", rank=rank, world_size=2)
    model1 = DDP(torch.nn.Linear(1, 1))
    model2 = DDP(torch.nn.Linear(1, 1))
    optim = Adam(list(model1.parameters()) + list(model2.parameters()), lr=0.01)
    # Rank 1 gets one more input than rank 0
    inputs = [torch.tensor([1.0]) for _ in range(10 + rank)]
    print(rank, "start")
    with model1.no_sync(), model2.no_sync():
        with Join([model1, model2], throw_on_early_termination=False):
            for input in inputs:
                model1.forward(input).sum()
                model2.forward(input).sum()
                # with model1.no_sync(), model2.no_sync():
                #     (loss1 + loss2).backward()
                optim.step()
    print(rank, "done")
    # All ranks reach here without hanging/erroring


def main():
    mp.spawn(worker, nprocs=2, join=True)


if __name__ == "__main__":
    main()
