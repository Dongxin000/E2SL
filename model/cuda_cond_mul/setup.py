from setuptools import setup
import torch
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# jit version of this!!!
from torch.utils.cpp_extension import load

# if there is more than one gpu type connected to this computer we need to provide the according architectures
# to nvcc
extra_cuda_cflags = []
if torch.cuda.device_count() > 1:
    arch_list = ""
    for device in range(0, torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(device)
        arch_list += f"{cap[0]}.{cap[1]} "

    extra_cuda_cflags = list(set(extra_cuda_cflags))
    extra_cuda_cflags = {'nvcc': extra_cuda_cflags,
                         'cxx': []}
    #set the cuda flag for architecture list so the code is not only compiled for only one of the installed GPUs
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list[:-1]
    setup(
        name='cond_mul_cuda',
        ext_modules=[
            CUDAExtension('cond_mul_cuda', ['cond_mul_cuda.cpp',
                                            'cond_mul_cuda_kernel.cu'])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    setup(
        name='cond_mul_cuda',
        ext_modules=[
            CUDAExtension('cond_mul_cuda', [
                'cond_mul_cuda.cpp',
                'cond_mul_cuda_kernel.cu',
            ]),
        ],
        cmdclass={
            'build_ext': BuildExtension
        })

