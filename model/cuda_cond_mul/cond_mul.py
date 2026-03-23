import math
import os
from torch import nn
from torch.autograd import Function
import torch

installed = False
if installed:
    import cond_mul_cuda
else:
    #jit version of this!!!
    from torch.utils.cpp_extension import load
    # if there is more than one gpu type connected to this computer we need to provide the according architectures
    # to nvcc
    extra_cuda_cflags = []
    if torch.cuda.device_count() > 1:
        arch_list = ""
        for device in range(0, torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(device)
            arch_list += f"{cap[0]}.{cap[1]} "
        # set the cuda flag for architecture list so the code is not only compiled for only one of the installed GPUs
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list[:-1]

        #for device in range(0, torch.cuda.device_count()):
        #    cap = torch.cuda.get_device_capability(device)
        #    extra_cuda_cflags.append(f"-arch=sm_{cap[0]}{cap[1]}")
        #extra_cuda_cflags.append(f"--gpu-code=sm_{cap[0]}{cap[1]}")
        #extra_cuda_cflags = list(set(extra_cuda_cflags))
        #print(extra_cuda_cflags)
        cond_mul_cuda = load(
            'cond_mul_cuda', ['model/cuda_cond_mul/cond_mul_cuda.cpp', 'model/cuda_cond_mul/cond_mul_cuda_kernel.cu'],
            verbose=True)#,
            #extra_cuda_cflags=extra_cuda_cflags)
    else:
        cond_mul_cuda = load(
            'cond_mul_cuda', ['model/cuda_cond_mul/cond_mul_cuda.cpp', 'model/cuda_cond_mul/cond_mul_cuda_kernel.cu'],
            verbose=True)

torch.manual_seed(42)


class Cond_Mul_Function(Function):
    @staticmethod
    def forward(ctx, input, inds, weights, bias):
        outputs = cond_mul_cuda.forward(input, inds, weights, bias)
        variables = [input] + [inds] + [weights]
        ctx.save_for_backward(*variables)

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_h):
        #TODO: remove if this turns out to be unnecessary
        # set the active device to the one that has our tensors
        #old_device = torch.cuda.current_device()
        #torch.cuda.set_device(grad_h.device)

        outputs = cond_mul_cuda.backward(
            grad_h.contiguous(), *ctx.saved_variables)

        #TODO: remove if this turns out to be unnecessary
        # restore the old state!
        #torch.cuda.set_device(old_device)

        grad_input, grad_weights, grad_bias = outputs
        return grad_input, None, grad_weights, grad_bias # what to do about the inds

# module takes input of shape(n, c_in) + indices (inds) of length n to generate the output (n, c_out)
class CondMul(nn.Module):
    def __init__(self, classes, input_features, output_features):
        super(CondMul, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        k = math.sqrt(1.0/float(input_features))
        self.register_parameter(name='w',
                                param=nn.Parameter(torch.rand(classes, input_features, output_features)*k*2.0-k))
        self.register_parameter(name='b',
                                param=nn.Parameter(torch.rand(classes, 1, output_features)*k*2.0-k))

    def forward(self, input, inds):
        assert len(input.shape) == 2, \
            "Expect input to be of shape (n, input_features)."
        assert input.shape[1] == self.input_features, \
            f"Expect the second dimension of input to be of size input_features {self.input_features}."
        assert len(inds.shape) == 1, \
            "Expecting inds to be of shape (n)"
        assert inds.shape[0] == input.shape[0], \
            "Expecting dim[0] of input and inds to be of same size."
        assert inds.device == self.w.device and input.device == self.w.device, \
            f"Expecting input tensors to be on device {self.w.device}."

        # set the active device to the one that has our tensors
        old_device = torch.cuda.current_device()
        torch.cuda.set_device(self.w.device)

        result = Cond_Mul_Function.apply(input, inds, self.w, self.b)

        # restore the old state!
        torch.cuda.set_device(old_device)
        return result
