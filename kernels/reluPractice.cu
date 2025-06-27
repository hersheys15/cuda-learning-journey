// Needed includes
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// =======================
// CUDA Kernel
// =======================
__global__ void relu_kernel(/* device pointers, size */) 
{
    // each thread gets a unique index
    // perform: y[i] = max(0, x[i]);
}

// =======================
// C++ Wrapper Function
// =======================
void relu(torch::Tensor x, torch::Tensor y) 
{
    // 1. Check that tensors are on CUDA and same shape/dtype
    // 2. Get raw pointers from x and y
    // 3. Compute number of elements
    // 4. Configure grid and block sizes
    // 5. Launch relu_kernel<<<grid, block>>>(...)
}

// =======================
// PyTorch Binding
// =======================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("relu", &relu, "ReLU kernel (CUDA)");
}
