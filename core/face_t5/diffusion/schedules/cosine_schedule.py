import numpy as np
import torch


def cosine_schedule(
    timesteps: int, max_beta: float = 0.999
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

    def alpha_bar(t):
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    betas = torch.from_numpy(np.array(betas)).float().to(device)

    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_cumprod_sqrt = alphas_cumprod.sqrt()
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
