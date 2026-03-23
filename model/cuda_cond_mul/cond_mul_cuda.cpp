#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> cond_mul_cuda_forward(
    torch::Tensor input,
    torch::Tensor inds,
    torch::Tensor weights,
    torch::Tensor bias);


std::vector<torch::Tensor> cond_mul_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor inds,
    torch::Tensor weights);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> cond_mul_forward(
    torch::Tensor input,
    torch::Tensor inds,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(input);
  CHECK_INPUT(inds);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return cond_mul_cuda_forward(input, inds, weights, bias);
}

std::vector<torch::Tensor> cond_mul_backward(
    torch::Tensor grad_output,//gradient of output
    torch::Tensor input,
    torch::Tensor inds,
    torch::Tensor weights) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  CHECK_INPUT(inds);
  CHECK_INPUT(weights);

  return cond_mul_cuda_backward(
      grad_output,
      input,
      inds,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cond_mul_forward, "cond_mul forward (CUDA)");
  m.def("backward", &cond_mul_backward, "cond_mul backward (CUDA)");
}
