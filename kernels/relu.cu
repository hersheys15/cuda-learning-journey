#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* x, float* y, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}

void relu_kernel(torch::Tensor x, torch::Tensor y) 
{
    int N = x.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("relu_kernel", &relu_kernel, "ReLU kernel");
}
