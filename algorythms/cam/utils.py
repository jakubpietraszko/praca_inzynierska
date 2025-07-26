def forward_hook(module, input, output):
    nonlocal activations
    activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]