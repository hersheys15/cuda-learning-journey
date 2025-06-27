# relu.py

# define modules
import subprocess
import torch, torch.utils
from torch.utils.cpp_extension import load

# load
torch.utils.cpp_extension.load(sources = ["relu.cu"], name = "reluuu")

# Runs relu.cu
subprocess.run(["nvcc", "relu.cu"])
subprocess.run(["./a.out", "relu.c"])