# 🗓️ Weekly Log — May 29th, 2025

## 🚧 Progress Summary

This week, I implemented and tested a basic CUDA kernel for matrix multiplication (`matrix_mult.cu`). I also wrote a CPU equivalent to establish a timing baseline.

### ✅ Completed
- Wrote and tested naive CUDA kernel for square matrix multiplication
- Created CPU version for comparison
- Ran initial benchmarks on 512×512 and 1024×1024 matrices

### ⚙️ Technical Notes
- GPU performs x faster than CPU on 1024×1024 matrices

### 🧠 Key Takeaways
- Writing even a basic kernel taught me a lot about thread indexing logic

## 📊 Benchmark Snapshot

| Matrix Size | CPU Time (ms) | GPU Time (ms) |
|-------------|---------------|---------------|
| 1024×1024   |               |               |

> CPU: single-threaded baseline, not optimized  
> GPU: NVIDIA RTX 4090

## 🔭 Goals for Next Week
- Understand how matrix mult works intuitively
- Read quantization paper
- Do simple symmetric quantization on a 1D Array FP16 Array [-128, 127]
- Pytorch Matmult
- Start writing `notes/2025-06-05-shared-memory.md`

---