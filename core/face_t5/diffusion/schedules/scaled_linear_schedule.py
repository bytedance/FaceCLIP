import numpy as np
import torch


def scaled_linear_schedule(
    timesteps: int,
    beta_0: float = 0.0001,
    beta_1: float = 0.0200,
    zero_terminal: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    NOTE: "alphas_cumprod" listed here is listed as "alpha" in Eq. (4) of [1].
          This can be verified by pattern matching the assigning of "x_t" in
          the "shared_step" method of the DDPM class in
          "models/diffusion/ddpm.py".
    References:
        [1] Song et al. Denoising Diffusion Implicit Models. ICLR 2021.
    """
    assert 0.0 < beta_0 < beta_1 < 1.0, "beta_0 and beta_1 must be in (0, 1)"
    betas = np.linspace(beta_0 ** 0.5, beta_1 ** 0.5, timesteps, dtype=np.float64) ** 2
    betas = torch.from_numpy(betas).float()

    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_cumprod_sqrt = alphas_cumprod.sqrt()

    if zero_terminal:
        """
        Based on:
        https://arxiv.org/abs/2305.08891
        """
        # store old value
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
        # shift so last timestep is zero
        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        # scale so first timestep is back to original value
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (
            alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T
        )
        # convert alpha_cumprod_sqrt back to betas
        alphas_cumprod = alphas_cumprod_sqrt.square()
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = torch.cat([alphas_cumprod[0:1], alphas])
        betas = 1.0 - alphas

    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
    alphas_cumprod_m1_sqrt = (1 - alphas_cumprod).sqrt()

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    logvar = torch.log(torch.cat((posterior_variance[1:2], betas[1:]), 0))

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_sqrt": alphas_cumprod_sqrt,
        "alphas_cumprod_m1_sqrt": alphas_cumprod_m1_sqrt,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "posterior_variance": posterior_variance,
        "logvar": logvar,
    }
