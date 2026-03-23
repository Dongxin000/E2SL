import torch
import torch.nn as nn
import math

class RefCondMul(nn.Module):

    def __init__(self, classes, input_features, output_features):
        super(RefCondMul, self).__init__()
        m=input_features
        n=output_features
        self.m = m
        self.n = n
        k = math.sqrt(1.0/float(input_features))
        self.register_parameter(name='w',
                                param=nn.Parameter(torch.rand(classes, m, n)*k*2.0-k))
        self.register_parameter(name='b',
                                param=nn.Parameter(torch.rand(classes, 1, n)*k*2.0-k))


    # x should be in shape ( n, input_features) transformed to (n, input_features, 1, 1)
    def forward(self, x, inds):
        x = x.transpose(1, len(x.shape)-1)
        old_shape_2 = list(x.shape)
        x = x.reshape((-1, 1, self.m))
        inds = inds.flatten().type(torch.int64)
        b = self.b.index_select(0, inds)
        w = self.w.index_select(0, inds)
        x = torch.matmul(x, w) + b

        old_shape_2[-1] = self.n
        x = x.reshape(old_shape_2)
        x = x.transpose(1, len(x.shape)-1)

        return x

class RefCondMulConv(nn.Module):

    def __init__(self, classes, input_features, output_features):
        super(RefCondMulConv, self).__init__()
        m = input_features
        n = output_features
        self.m = m
        self.n = n
        self.input_features = input_features
        self.output_features = output_features
        self.classes = classes
        self.conv = nn.Conv2d(input_features, classes * output_features, 1)


    def forward(self, x, inds):
        inds = inds.type(torch.int64)
        # x should be in shape ( n, input_features) transformed to (n, input_features, 1, 1)
        x = x.reshape((-1, self.input_features, 1, 1))
        x = self.conv(x)

        # x should now be in shape(n, output_features * classes , 1, 1) so we convert to
        # (n, classes, output_features)
        x = x.reshape((-1, self.classes, self.output_features))
        x = torch.gather(x, 1, inds.reshape((-1, 1, 1)).repeat(1, 1, self.output_features))

        # x should nwo be in shape(n, 1, output_features)
        x = x.reshape((-1, self.output_features))
        return x


class Ref2CondMulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inds, weights, bias):
        device = weights.device
        outputs = torch.zeros((input.shape[0], bias.shape[-1]), device=device)
        variables = [input] + [inds] + [weights] + [bias]
        ctx.save_for_backward(*variables)
        #ctx.mark_non_differentiable(inds)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        # outputs = my_linear_cpp.backward(
        #    grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        # d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs

        # do it in python cause it is easier for now
        input, inds, weights, bias = ctx.saved_variables
        d_input = torch.zeros_like(input)
        d_weights = torch.zeros_like(weights)
        d_bias = torch.zeros_like(bias)
        d_inds = torch.zeros_like(inds)
        return d_input, inds, d_weights, d_bias  # the input of one is the output of the other...so or similar

class Ref2CondMul(nn.Module):

    def __init__(self, classes, m, n):
        super(Ref2CondMul, self).__init__()
        self.m = m
        self.n = n
        self.register_parameter(name='w', param=nn.Parameter(torch.randn((classes, m, n))))
        self.register_parameter(name='b', param=nn.Parameter(torch.randn((classes, 1, n))))

    def forward(self, x, inds):
        return Ref2CondMulFunc.apply(x, inds, self.w, self.b)

