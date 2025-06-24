# 🗓️ Weekly Log — June 12th, 2025

## 🚧 Progress Summary

This week, I implemented and benchmarked PyTorch-based matrix multiplication using both CPU and GPU (NVIDIA RTX 4090). I recorded detailed timing results across varying matrix sizes, highlighting significant GPU acceleration. I also deepened my understanding of low-level GPU execution by working with EPS (Equalization Power Scale) for range calibration and learning how binding ensures correct kernel parameter execution.

### ✅ Completed

- pytorch matrix multiplication

### ⚙️ Technical Notes

- EPS (Equalization Power Scale) helps stabilize computation during range calibration and scale calculation.
- Binding is used to ensure that the GPU kernel is executed with the correct parameters.

### 🧠 Key Takeaways

- kernel fusion is adding two kernels together to reduce memory access and improve performance.
- erasure coding is a method of data protection that allows for recovery of data even if some parts are lost or corrupted.

## 📊 Benchmark Snapshot

| Matrix Size | CPU Time (ms)    | GPU Time (ms)    |
|-------------|------------------|------------------|
|   64 x 64   | 0.000091 seconds | 0.000994 seconds |
| 128 x 128   | 0.001801 seconds | 0.000088 seconds |
| 256 x 256   | 0.002778 seconds | 0.000099 seconds |
| 1024 x 1024 | 0.007580 seconds | 0.000478 seconds |
| 4096 x 4096 | 0.112405 seconds | 0.027021 seconds |
|16384 x 16384| 5.676453 seconds | 2.535303 seconds |

> CPU: single-threaded baseline, not optimized  
> GPU: NVIDIA RTX 4090

## 🔭 Goals for Next Week

- Start writing `6-19-25-weeklylog.md`
- <https://github.com/xlite-dev/LeetCUDA/tree/main/kernels>
- flash-attn
- relu
- mat-transpose
- transformer
- softmax
- sgemv
- openai-triton
- nvidia-nsight
- turn into technical report on overleaf for companies and all (esp for overleaf continued familiarity)
- kernel fusions

---
