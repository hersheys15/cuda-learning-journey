import torch
from torch.utils.cpp_extension import load
import time
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

torch.set_grad_enabled(False)

# Load the compiled CUDA extension
relu_lib = load(name="relu_extension", sources=["relu.cu"])

def benchmark(fn, x, out=None, name="custom", iters=1000):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        if out is not None:
            fn(x, out)
        else:
            fn(x)
    torch.cuda.synchronize()
    end = time.time()
    print(f"{name:>12}: {(end - start)*1000/iters:.6f} ms")

def custom_relu_cpu(x):
    return torch.where(x > 0, x, torch.zeros_like(x))
# CPU benchmark
def run_cpu_benchmark(x_cpu, tag="cpu", iters=1000):
    start = time.time()
    for _ in range(iters):
        _ = custom_relu_cpu(x_cpu)
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    print(f"{tag:>12}: {mean_time:.6f} ms")
    return mean_time

# Inputs
x = torch.randn((2048, 2048), device="cuda").float()
y = torch.zeros_like(x)

# Run benchmarks
print("Benchmarking ReLU implementations:")
benchmark(relu_lib.relu_kernel, x, y, name="custom_relu")
benchmark(torch.relu, x, name="torch_relu")
x_cpu = x.detach().cpu().contiguous()
run_cpu_benchmark(x_cpu, tag="cpu")