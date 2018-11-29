import torch


class SoftMaxAndLogSoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        max, _ = input.max(dim=dim, keepdim=True)
        input_minus_max = input - max
        exp = torch.exp(input_minus_max)
        sum = exp.sum(dim=dim, keepdim=True)
        softmax = exp / sum
        log_softmax = input_minus_max - torch.log(sum)
        ctx.dim = dim
        ctx.save_for_backward(softmax)
        return softmax, log_softmax

    @staticmethod
    def backward(ctx, grad_softmax, grad_log_softmax):
        dim = ctx.dim
        softmax, = ctx.saved_tensors
        gx = softmax * grad_softmax
        gx -= softmax * gx.sum(dim=dim, keepdim=True)
        gx += grad_log_softmax - softmax * grad_log_softmax.sum(dim=dim, keepdim=True)
        return gx, None


def softmax_and_log_softmax(input, dim):
    return SoftMaxAndLogSoftMax.apply(input, dim)


_very_large_value = 1e34


class MaskedSoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, invalid_target_mask, dim):
        masked_input = input.clone()
        masked_input[invalid_target_mask] -= _very_large_value
        max, _ = masked_input.max(dim=dim, keepdim=True)
        masked_input -= max
        exp = torch.exp(masked_input)
        exp[invalid_target_mask] = 0
        sum = exp.sum(dim=dim, keepdim=True)
        softmax = exp / sum
        ctx.dim = dim
        ctx.save_for_backward(softmax)
        return softmax

    @staticmethod
    def backward(ctx, grad_softmax):
        dim = ctx.dim
        softmax, = ctx.saved_tensors
        gx = softmax * grad_softmax
        gx -= softmax * gx.sum(dim=dim, keepdim=True)
        return gx, None, None


def masked_softmax(input, invalid_target_mask, dim):
    return MaskedSoftMax.apply(input, invalid_target_mask, dim)


class MaskedLogSoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, invalid_target_mask, dim):
        masked_input = input.clone()
        masked_input[invalid_target_mask] -= _very_large_value
        max, _ = masked_input.max(dim=dim, keepdim=True)
        masked_input -= max
        exp = torch.exp(masked_input)
        exp[invalid_target_mask] = 0
        sum = exp.sum(dim=dim, keepdim=True)
        softmax = exp / sum
        log_softmax = masked_input - torch.log(sum)
        ctx.dim = dim
        ctx.save_for_backward(softmax)
        return log_softmax

    @staticmethod
    def backward(ctx, grad_log_softmax):
        dim = ctx.dim
        softmax, = ctx.saved_tensors
        gx = grad_log_softmax - softmax * grad_log_softmax.sum(dim=dim, keepdim=True)
        return gx, None, None


def masked_log_softmax(input, invalid_target_mask, dim):
    return MaskedLogSoftMax.apply(input, invalid_target_mask, dim)


class MaskedSoftMaxAndLogSoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, invalid_target_mask, dim):
        masked_input = input.clone()
        masked_input[invalid_target_mask] -= _very_large_value
        max, _ = masked_input.max(dim=dim, keepdim=True)
        masked_input -= max
        exp = torch.exp(masked_input)
        exp[invalid_target_mask] = 0
        sum = exp.sum(dim=dim, keepdim=True)
        softmax = exp / sum
        log_softmax = masked_input - torch.log(sum)
        ctx.dim = dim
        ctx.save_for_backward(softmax)
        return softmax, log_softmax

    @staticmethod
    def backward(ctx, grad_softmax, grad_log_softmax):
        dim = ctx.dim
        softmax, = ctx.saved_tensors
        gx = softmax * grad_softmax
        gx -= softmax * gx.sum(dim=dim, keepdim=True)
        gx += grad_log_softmax - softmax * grad_log_softmax.sum(dim=dim, keepdim=True)
        return gx, None, None


def masked_softmax_and_log_softmax(input, invalid_target_mask, dim):
    return MaskedSoftMaxAndLogSoftMax.apply(input, invalid_target_mask, dim)
