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
    return tensor - modulo * torch.div(tensor + offset, modulo, rounding_mode="floor")

def rand_symm_matrix(n, device=None, dtype=None):
    """
    Returns a random symmetric matrix.

    Args:
        n (int): matrix size.
        device (torch.device, optional): device to use. Defaults to None.
        dtype (torch.dtype, optional): data type to use. Defaults to None.

    Returns:
        (torch.Tensor): random symmetric matrix.
    """
    A = torch.rand(n, n, device=device, dtype=dtype)
    return (A + A.T) / 2

def indices_to_start_end(indices):
    indices_step = torch.argwhere(torch.diff(indices) > 1).flatten()
    start = torch.cat((torch.tensor([0]), indices_step + 1))
    end = torch.cat((indices_step, torch.tensor([indices.size(0) - 1])))
    return indices[start], indices[end]
