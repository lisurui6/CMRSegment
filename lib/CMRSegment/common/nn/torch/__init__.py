import torch
from typing import Iterable


def prepare_tensors(tensors: Iterable[torch.Tensor], gpu: bool, device: int):
    if gpu:
        if isinstance(tensors, torch.Tensor):
            if not tensors.is_cuda:
                tensors = tensors.cuda(device=device)
            return tensors
        tensors = list(tensors)
        for idx, tensor in enumerate(tensors):
            if not tensor.is_cuda:
                tensor = tensor.cuda(device=device)
                tensors[idx] = tensor
    return tensors
