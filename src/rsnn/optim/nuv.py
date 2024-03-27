import torch


def box_nuv(mx, xb, gamma=1.0):
    """
    Update the NUV means and variances for a box constraints of the form |x| <= xb.

    Args:
        mx (torch.tensor): the posterior mean.
        xb (torch.tensor): the box constraints.
        gamma (float): the NUV slope parameter.

    Returns:
        (torch.tensor): the NUV means.
        (torch.tensor): the NUV variances.
    """
    Vxlf = torch.abs(mx + xb)
    Vxrl = torch.abs(mx - xb)
    return xb * (Vxlf - Vxrl) / (Vxlf + Vxrl), Vxlf * Vxrl / (Vxlf + Vxrl) / gamma


def right_half_space_nuv(mx, xb, gamma=1.0):
    """
    Update the NUV means and variances for a right half space constraints of the form x >= xb.

    Args:
        mx (torch.tensor): the posterior mean.
        xb (torch.tensor): the half-space constraints.
        gamma (float): the NUV slope parameter.

    Returns:
        (torch.tensor): the NUV means.
        (torch.tensor): the NUV variances.
    """
    return xb + torch.abs(mx - xb), torch.abs(mx - xb) / gamma


def left_half_space_nuv(mx, xb, gamma=1.0):
    """
    Update the NUV means and variances for a left half space constraints of the form x <= xb.

    Args:
        mx (torch.tensor): the posterior mean.
        xb (torch.tensor): the half-space constraints.
        gamma (float): the NUV slope parameter.

    Returns:
        (torch.tensor): the NUV means.
        (torch.tensor): the NUV variances.
    """
    return xb - torch.abs(mx - xb), torch.abs(mx - xb) / gamma


def binary_nuv(mx, Vx, xb):
    """
    Update the NUV means and variances for a binary constraints of the form x in {-xb, xb}.

    Args:
        mx (torch.tensor): the posterior mean.
        xb (torch.tensor): the binary constraints.

    Returns:
        (torch.tensor): the NUV means.
        (torch.tensor): the NUV variances.
    """
    Vxlf = Vx + torch.square(mx + xb)
    Vxrf = Vx + torch.square(mx - xb)

    return xb * (Vxlf - Vxrf) / (Vxlf + Vxrf), Vxlf * Vxrf / (Vxlf + Vxrf)
