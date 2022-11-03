import torch


def mod(tensor, modulo, offset):
    """
    Returns the elementwise modulo with offset of the input tensor.

    Args:
        tensor (torch.Tensor): input tensor.
        modulo (torch.Tensor): modulo.
        offset (torch.Tensor): offset.

    Returns:
        (torch.Tensor): elementwise modulo with offset tensor.
    """
    # modulo with offset
    return tensor - modulo * torch.div(tensor + offset, modulo, rounding_mode="floor")


def indices_to_start_end(indices):
    indices_step = torch.argwhere(torch.diff(indices) > 1).flatten()
    start = torch.cat((torch.tensor([0]), indices_step + 1))
    end = torch.cat((indices_step, torch.tensor([indices.size(0) - 1])))
    return indices[start], indices[end]
