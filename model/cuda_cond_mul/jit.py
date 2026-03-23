from torch.utils.cpp_extension import load
cond_mul_cuda = load(
    'cond_mul_cuda', ['cond_mul_cuda.cpp', 'cond_mul_cuda_kernel.cu'], verbose=True)
help(cond_mul_cuda)
