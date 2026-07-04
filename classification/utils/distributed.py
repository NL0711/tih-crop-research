# --------------------------------------------------------
# Distributed Training Utilities
# Provides a proper abstraction layer for distributed execution.
# When torch.distributed IS initialized: uses real distributed ops.
# When torch.distributed is NOT initialized: falls back to single-GPU
# mode with no-op stubs (rank=0, world_size=1).
# --------------------------------------------------------

import os
import torch
import torch.distributed as torch_dist


def is_dist_avail_and_initialized():
    """Check if torch.distributed is available AND a process group is initialized."""
    if not torch_dist.is_available():
        return False
    if not torch_dist.is_initialized():
        return False
    return True


def get_world_size():
    """Return the world size. Falls back to 1 when not distributed."""
    if not is_dist_avail_and_initialized():
        return 1
    return torch_dist.get_world_size()


def get_rank():
    """Return the global rank. Falls back to 0 when not distributed."""
    if not is_dist_avail_and_initialized():
        return 0
    return torch_dist.get_rank()


def is_main_process():
    """Return True if this is rank 0 (or single-GPU mode)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes. No-op when not distributed."""
    if not is_dist_avail_and_initialized():
        return
    torch_dist.barrier()


def broadcast_object_list(obj_list, src=0):
    """Broadcast a list of picklable objects. No-op when not distributed."""
    if not is_dist_avail_and_initialized():
        return
    torch_dist.broadcast_object_list(obj_list, src=src)


def all_reduce(tensor, op=None):
    """All-reduce a tensor across processes. No-op when not distributed."""
    if not is_dist_avail_and_initialized():
        return
    if op is None:
        op = torch_dist.ReduceOp.SUM
    torch_dist.all_reduce(tensor, op=op)


def reduce_tensor(tensor):
    """Reduce a tensor by averaging across all processes.

    In single-GPU mode the tensor is returned unchanged.
    """
    if not is_dist_avail_and_initialized():
        return tensor
    rt = tensor.clone()
    torch_dist.all_reduce(rt, op=torch_dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def init_distributed_mode():
    """Initialize distributed training if launched via torchrun / torch.distributed.launch.

    Detects environment variables set by torchrun (RANK, WORLD_SIZE, LOCAL_RANK).
    If they are absent, configures single-GPU mode silently.

    Returns:
        tuple: (rank, world_size) after initialization.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

        torch.cuda.set_device(local_rank)

        # Use gloo on Windows, nccl on Linux
        backend = 'gloo' if os.name == 'nt' else 'nccl'

        torch_dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )
        torch_dist.barrier()
        return rank, world_size
    else:
        # Single-GPU mode — no process group needed
        print("Not using distributed mode (single GPU)")
        torch.cuda.set_device(0)
        return 0, 1
