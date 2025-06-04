# 🗓️ Weekly Log — May 29th, 2025

## 🚧 Progress Summary

This week, I implemented and tested a basic CUDA kernel for matrix multiplication (`matrix_mult_naive.cu`). I also wrote a CPU equivalent to establish a timing baseline and started organizing this repository to track future development.

### ✅ Completed
- Set up GitHub repo and directory structure
- Wrote and tested naive CUDA kernel for square matrix multiplication
- Created CPU version for comparison
- Ran initial benchmarks on 512×512 and 1024×1024 matrices
- Drafted README and updated progress checklist

### ⚙️ Technical Notes
- Naive kernel uses global memory only, no tiling
- Kernel launch config: `<<<(N/32, N/32), (32, 32)>>` for N×N matrices
- GPU performs ~15–20x faster than CPU on 1024×1024 matrices, depending on system load

### 🧠 Key Takeaways
- Memory coalescing wasn't terrible in naive kernel, but shared memory should improve it further
- Writing even a basic kernel taught me a lot about thread indexing logic
- `nvprof` was useful but verbose—will look into `nvtx` for cleaner profiling

## 📊 Benchmark Snapshot

| Matrix Size | CPU Time (ms) | GPU Time (ms) |
|-------------|---------------|---------------|
| 512×512     | 143.4         | 10.8          |
| 1024×1024   | 876.2         | 57.2          |

> CPU: single-threaded baseline, not optimized  
> GPU: NVIDIA RTX 3070, compiled with `nvcc -O2`

## ❓ Questions / Challenges
- Should I manually handle edge cases where matrix size isn’t divisible by block size now or later?
- When profiling with `nvprof`, how do I interpret the "global memory throughput" in context of peak device bandwidth?

## 🔭 Goals for Next Week
- Implement shared memory tiling (following NVIDIA's guide)
- Compare shared memory kernel with naive version
- Automate benchmark script + logging
- Start writing `notes/2025-06-10-shared-memory.md`

---

