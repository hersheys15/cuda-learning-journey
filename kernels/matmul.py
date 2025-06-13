import time
import torch
from torch.utils.cpp_extension import load

matmul_cuda = load(
    name="matmul_cuda",
    sources=["matmulWpytor.cu"],
    verbose=True
)

# Test
M = 2**11 # Is 1024
A = torch.randn(M, M, device="cuda", dtype=torch.float32)
B = torch.randn(M, M, device="cuda", dtype=torch.float32)

# Ensure the CUDA extension is loaded
if not matmul_cuda:
    raise RuntimeError("Failed to load the CUDA extension for matmul.")

# Measure performance for CUDA
start = time.time()
C = matmul_cuda.matmul(A, B)
torch.cuda.synchronize()
end = time.time()
print(f"CUDA matmul time: {end - start:.6f} seconds")
# Measure performance for CPU
start_cpu = time.time()
C_cpu = torch.matmul(A.cpu(), B.cpu())
end_cpu = time.time()
print(f"CPU matmul time: {end_cpu - start_cpu:.6f} seconds")

# Compare performance
print(f"CUDA matmul is {'faster' if (end - start) < (end_cpu - start_cpu) else 'slower'} than CPU matmul.")
