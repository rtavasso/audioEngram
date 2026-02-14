"""
Adversarial and feature matching losses for GAN training.

- Hinge loss for generator and discriminator
- L1 feature matching loss across discriminator layers
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def adversarial_g_loss(
    disc_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor:
    """
    Generator hinge loss: encourage discriminator to classify fakes as real.

    Args:
        disc_outputs: list of (logits, features) from MultiScaleSTFTDiscriminator on fake audio.

    Returns:
        Scalar loss.
    """
    loss = torch.tensor(0.0, device=disc_outputs[0][0].device)
    for logits, _ in disc_outputs:
        loss = loss + F.relu(1.0 - logits).mean()
    return loss / len(disc_outputs)


def adversarial_d_loss(
    disc_real_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    disc_fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor:
    """
    Discriminator hinge loss.

    Args:
        disc_real_outputs: discriminator outputs on real audio.
        disc_fake_outputs: discriminator outputs on fake audio.

    Returns:
        Scalar loss.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0][0].device)
    for (real_logits, _), (fake_logits, _) in zip(disc_real_outputs, disc_fake_outputs):
        loss = loss + F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    return loss / len(disc_real_outputs)


def feature_matching_loss(
    disc_real_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    disc_fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor:
    """
    L1 feature matching loss between real and fake intermediate features.

    Args:
        disc_real_outputs: discriminator outputs on real audio.
        disc_fake_outputs: discriminator outputs on fake audio.

    Returns:
        Scalar loss (averaged over scales and layers).
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0][0].device)
    n_layers = 0
    for (_, real_feats), (_, fake_feats) in zip(disc_real_outputs, disc_fake_outputs):
        for rf, ff in zip(real_feats, fake_feats):
            loss = loss + F.l1_loss(ff, rf.detach())
            n_layers += 1
    return loss / max(n_layers, 1)
