

import torch


def sample_sigmas(
    img_batch,
    P_mean=-1.2,
    P_std=1.2,
):
    rnd_normal = torch.randn(
        [img_batch.shape[0], 1, 1, 1], device=img_batch.device
    )
    sigmas = (rnd_normal * P_std + P_mean).exp()
    return sigmas


def edm_loss(
    model,
    img_batch,
    sigma_data=0.5,
    P_mean=-1.2,
    P_std=1.2,
    sigmas=None,
    noise=None,
    class_labels=None,
    return_output=False,
    mean=True,
):

    # Set noise vector
    if noise is not None:
        assert sigmas is not None, (
            "If noise is provided, sigmas must be provided."
        )
        n = noise

    else:
        sigmas = sigmas or sample_sigmas(img_batch, P_mean, P_std)
        n = torch.randn_like(img_batch) * sigmas

    # Weight coefficient for loss, as introduced in EDM paper
    weight = (
        (sigmas**2 + sigma_data**2) / (sigmas * sigma_data)**2
    )

    # Compute denoised image with forward model pass
    D_yn = model(img_batch + n, sigmas, class_labels=class_labels)

    # Compute loss
    loss = weight * (D_yn - img_batch)**2
    if mean:
        loss = loss.mean()

    return (loss, D_yn) if return_output else loss
