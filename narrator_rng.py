"""Thread-local RNG isolation for Kokoro's random excitation and noise."""
import torch
from torch.overrides import TorchFunctionMode


class NarratorRandomMode(TorchFunctionMode):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.rand, torch.randn):
            device = torch.device(kwargs.get("device") or "cpu")
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if device == self.generator.device and "generator" not in kwargs:
                return func(*args, **{**kwargs, "generator": self.generator})
        if func is torch.randn_like and args[0].device == self.generator.device:
            # randn_like has no generator argument; preserve its tensor options.
            return torch.empty_like(*args, **kwargs).normal_(generator=self.generator)
        return func(*args, **kwargs)
