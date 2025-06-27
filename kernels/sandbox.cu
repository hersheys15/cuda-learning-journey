// relu.cu

#include <cstdio>
#include <torch/extension.h>

// int main(void)
// {
//     printf("Hello GPU.");
//     return 0;
// }

PYBIND11_MODULE(module_name, m)
{
    m.def("hello", &hello_func)
    {
        std::string hello()
        {
            printf("%s", &hello);
            return hello;
        }
    }
}