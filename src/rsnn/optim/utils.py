import torch


def all_close_to_one_of(tensor, values, rtol=1e-5, atol=1e-8):
    """
    Check if all elements of a tensor are close to one of the values.

    Args:
        array (torch.tensor): the constrained arrays.
        values (torch.tensor): the allowed values.
        rtol (float, optional): the relative tolerance. Defaults to 1e-5.
        atol (float, optional): the absolute tolerance. Defaults to 1e-8.

    Returns:
        (bool): the check result.
    """
    return torch.isclose(tensor[..., None], values, rtol=rtol, atol=atol).any(dim=-1).all()

def obs_block_(mxf, Vxf, myb, Vyb, C):
    """
    Gaussian message passing through a (scalar) observation block.
    Inplace operation.

    Args:
        mxf (torch.tensor): the forward input mean vector with shape (K).
        Vxf (torch.tensor): the forward input covariance matrix with shape (K, K).
        myb (torch.tensor): the backward observation mean with shape (1).
        Vyb (torch.tensor): the backward observation variance with shape (1).
        C (torch.tensor): the observation matrix with shape (K).
    """
    CVxf = C @ Vxf # (K)
    g = 1 / (Vyb + torch.inner(CVxf,C))
    mxf.add_(CVxf, alpha=(myb - torch.inner(C,mxf)) * g)
    Vxf.sub_(torch.outer(CVxf, CVxf), alpha=g)
