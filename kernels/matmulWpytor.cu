#include <torch/extension.h>
// Needed since we do not compile w nvcc, which does include the CUDA headers by default in cuda files.
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < M) 
    {
        float val = 0;
        for (int k = 0; k < M; ++k)
            val += A[row * M + k] * B[k * M + col];
        C[row * M + col] = val;
    }
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) 
{
    int M = A.size(0);
    auto C = torch::zeros({M, M}, A.options());

    const int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((M + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("matmul", &matmul, "Matrix multiplication (CUDA)");
}
