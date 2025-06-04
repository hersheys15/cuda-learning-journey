#include <iostream>
#include <cstdlib>
#include <chrono>

#define N 1024  // Matrix size N x N

// -------------------- CPU VERSION --------------------
void matrixMultiplyCPU(float *A, float *B, float *C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// -------------------- GPU VERSION --------------------
__global__ void matrixMultiplyGPU(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row in C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column in C

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// -------------------- MAIN --------------------
int main() {
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory, CPU side
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    // Initialize matrices with random values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU Time: " << duration_cpu.count() << " ms\n";

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Kernel launch config
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    // GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    matrixMultiplyGPU<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float duration_gpu;
    cudaEventElapsedTime(&duration_gpu, start_gpu, stop_gpu);
    std::cout << "GPU Time: " << duration_gpu << " ms\n";

    std::cout << "GPU Faster by percent of: " << (duration_cpu.count() / duration_gpu - 1) * 100 << "%\n";

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
