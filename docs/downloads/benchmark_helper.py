"""Measure a precompiled kernel with preallocated inputs and outputs.

Example (replace this with your kernel and buffers):
    from benchmark_helper import measure
    result = measure(lambda: kernel(x, bias, y), useful_bytes=2*x.numel()*4)

Compile, allocate, initialize, and validate before calling measure(). The byte
count is an algorithmic convention, not a hardware DRAM traffic measurement.
"""

import torch
import tilelang
from tilelang.profiler import do_bench


def measure(fn, useful_bytes, warmup_ms=25, repeat_ms=100):
    if useful_bytes <= 0:
        raise ValueError("useful_bytes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch environment is required")
    # Warm up before the measurement helper's own warm-up stage.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ms = do_bench(
        fn, warmup=warmup_ms, rep=repeat_ms,
        backend="event", return_mode="median",
    )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tilelang": getattr(tilelang, "__version__", "unknown"),
        "latency_us": ms * 1000,
        "effective_gbps": useful_bytes / (ms * 1e-3) / 1e9,
        "useful_bytes": useful_bytes,
        "cache_policy": "TileLang event backend: flush L2 before each sample",
        "statistic": "median",
    }
    print(result)
    return result
