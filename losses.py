"""
Loss functions for JukeDrummer language model training.

Available losses:
  - focal_loss      : focal cross-entropy, focuses on hard tokens
  - perceptual_loss : soft codebook embedding MSE
  - fad_loss        : Fréchet distance in VQ codebook embedding space
  - onset_contour_loss : temporally-tolerant onset alignment loss
  - mel_detail_loss    : mel-space reconstruction via frozen VQVAE decoder
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Focal loss
# ─────────────────────────────────────────────────────────────────────────────


def focal_loss(pred, targets, gamma=2.0):
    """
    FL = (1 - p_t)^gamma * CE.  gamma=0 → standard CE.

    pred:    (N, T, bins)  logits
    targets: (N, T)        integer token ids
    returns: scalar loss in bits
    """
    bins = pred.shape[-1]
    ce_nats = F.cross_entropy(
        pred.reshape(-1, bins), targets.reshape(-1), reduction="none"
    )
    if gamma > 0:
        pt = torch.exp(-ce_nats)
        loss = ((1 - pt) ** gamma * ce_nats).mean()
    else:
        loss = ce_nats.mean()
    return loss / np.log(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual loss
# ─────────────────────────────────────────────────────────────────────────────


def perceptual_loss(pred, targets, codebook, tau=0.5):
    """
    Soft codebook lookup MSE.
    Differentiable proxy for "how far in VQ embedding space is the prediction
    from the ground-truth codebook vector".

    pred:      (N, T, codebook_size)  logits
    targets:   (N, T)                 integer token ids
    codebook:  (codebook_size, d)     frozen VQ buffer
    tau:       softmax temperature (lower = sharper)
    returns:   scalar MSE loss
    """
    soft_w = torch.softmax(pred / tau, dim=-1)  # (N, T, C)
    soft_emb = soft_w @ codebook  # (N, T, d)
    target_emb = codebook[targets]  # (N, T, d)
    return F.mse_loss(soft_emb, target_emb)


def perceptual_loss_advanced(
    pred, targets, codebook, tau=0.5, alpha=0.5, use_cosine=True
):
    """
    Enhanced perceptual loss combining MSE and cosine similarity.
    Ensures both magnitude and directional alignment in embedding space.

    pred:      (N, T, codebook_size)  logits
    targets:   (N, T)                 integer token ids
    codebook:  (codebook_size, d)     frozen VQ buffer
    tau:       softmax temperature (lower = sharper)
    alpha:     weight for cosine component (0 = MSE only, 1 = cosine only)
    returns:   scalar combined loss
    """
    soft_w = torch.softmax(pred / tau, dim=-1)  # (N, T, C)
    soft_emb = soft_w @ codebook  # (N, T, d)
    target_emb = codebook[targets]  # (N, T, d)

    mse_loss = F.mse_loss(soft_emb, target_emb)

    if use_cosine:
        cos_loss = 1 - F.cosine_similarity(soft_emb, target_emb, dim=-1).mean()
        return (1 - alpha) * mse_loss + alpha * cos_loss

    return mse_loss


def perceptual_loss_multiscale(
    pred, targets, codebook, taus=[0.1, 0.5, 1.0], weights=None
):
    """
    Multi-scale perceptual loss using multiple softmax temperatures.
    Captures both sharp (confident) and soft (uncertain) distributions.

    pred:      (N, T, codebook_size)  logits
    targets:   (N, T)                 integer token ids
    codebook:  (codebook_size, d)     frozen VQ buffer
    taus:      list of temperatures to average over
    weights:   optional weights for each tau scale
    returns:   scalar combined MSE loss
    """
    if weights is None:
        weights = [1.0] * len(taus)

    total_loss = 0.0
    for tau, w in zip(taus, weights):
        soft_w = torch.softmax(pred / tau, dim=-1)
        soft_emb = soft_w @ codebook
        target_emb = codebook[targets]
        total_loss += w * F.mse_loss(soft_emb, target_emb)

    return total_loss / sum(weights)


# ─────────────────────────────────────────────────────────────────────────────
# FAD loss helpers
# ─────────────────────────────────────────────────────────────────────────────


def _matrix_sqrt(A):
    """
    Differentiable symmetric matrix square root via eigendecomposition.
    A: (d, d) symmetric positive semi-definite matrix.
    """
    A = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype) * 1e-6
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues = eigenvalues.clamp(min=0.0)
    return eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.mT


def _frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real):
    """
    FD = ||mu_r - mu_g||^2 + Tr(Σ_r + Σ_g - 2 * sqrt(Σ_r @ Σ_g))
    All inputs are differentiable w.r.t. the generated distribution.
    """
    mean_sq_diff = (mu_gen - mu_real).pow(2).sum()
    sqrt_prod = _matrix_sqrt(sigma_real @ sigma_gen)
    trace_term = torch.trace(sigma_gen + sigma_real - 2.0 * sqrt_prod)
    return mean_sq_diff + trace_term


# ─────────────────────────────────────────────────────────────────────────────
# FAD loss
# ─────────────────────────────────────────────────────────────────────────────


def fad_loss(pred, targets, codebook, tau=0.5, diagonal=False):
    """
    Fréchet Audio Distance in VQ codebook embedding space.

    Uses soft codebook embeddings (differentiable) for the generated side and
    hard codebook lookups (detached) for the real side.  Computes batch-level
    mean and covariance, then the Fréchet distance between the two Gaussians.

    pred:      (N, T, codebook_size)  logits
    targets:   (N, T)                 integer token ids
    codebook:  (codebook_size, d)     frozen VQ buffer  (d = 64)
    tau:       softmax temperature for soft lookup
    diagonal:  if True, use diagonal covariance (variance only) — faster,
               more stable, less expressive.  Good for initial experiments.
    returns:   scalar Fréchet distance (lower = generated distribution closer
               to real distribution in embedding space)
    """
    d = codebook.shape[-1]

    # Generated embeddings (differentiable)
    soft_w = torch.softmax(pred / tau, dim=-1)
    gen_emb = (soft_w @ codebook).reshape(-1, d)  # (N*T, d)

    # Real embeddings (no gradient — we match the real distribution, not move it)
    real_emb = codebook[targets].reshape(-1, d).detach()  # (N*T, d)

    mu_gen = gen_emb.mean(dim=0)
    mu_real = real_emb.mean(dim=0)

    n = gen_emb.shape[0]
    gen_c = gen_emb - mu_gen
    real_c = real_emb - mu_real

    if diagonal:
        # Variance-only approximation — avoids matrix sqrt, very stable
        var_gen = gen_c.pow(2).mean(dim=0)  # (d,)
        var_real = real_c.pow(2).mean(dim=0)  # (d,)
        mean_sq = (mu_gen - mu_real).pow(2).sum()
        var_term = (var_gen.sqrt() - var_real.sqrt()).pow(2).sum()
        return mean_sq + var_term
    else:
        sigma_gen = (gen_c.T @ gen_c) / (n - 1)  # (d, d)
        sigma_real = (real_c.T @ real_c) / (n - 1)  # (d, d)
        return _frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)


# ─────────────────────────────────────────────────────────────────────────────
# Onset contour loss
# ─────────────────────────────────────────────────────────────────────────────


def _gaussian_kernel_1d(sigma, device):
    """Create a 1D Gaussian smoothing kernel (truncated at 3*sigma)."""
    radius = int(3 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _energy_and_onsets(emb):
    """Compute energy envelope and onset strength from embeddings (N, T, d)."""
    energy = emb.norm(dim=-1)                                    # (N, T)
    onset = F.relu(energy[:, 1:] - energy[:, :-1])              # (N, T-1)
    return energy, onset


def _smooth(signal, sigma, device):
    """Gaussian-smooth a (N, T) signal along time axis."""
    kernel = _gaussian_kernel_1d(sigma, device).view(1, 1, -1)
    pad = kernel.shape[-1] // 2
    return F.conv1d(signal.unsqueeze(1), kernel, padding=pad).squeeze(1)


def onset_contour_loss(pred, targets, codebook, tau=0.5, sigma=3.0,
                       sigmas=None, peak_weight=2.0):
    """
    Multi-scale onset contour loss with peak presence penalty.

    Compares onset profiles at multiple Gaussian smoothing scales:
      - Fine scale (sigma=1): tight timing — penalizes shifted hits
      - Medium scale (sigma=3): moderate tolerance
      - Coarse scale (sigma=6): overall rhythmic pattern

    Plus a peak presence term that upweights timesteps where the target
    has strong onsets, penalizing the model harder for *missing* drum hits
    than for adding extra ones.

    pred:         (N, T, codebook_size)  logits
    targets:      (N, T)                 integer token ids
    codebook:     (codebook_size, d)     frozen VQ buffer
    tau:          softmax temperature for soft lookup
    sigma:        backward-compatible single scale (used if sigmas is None)
    sigmas:       list of scales for multi-scale comparison
    peak_weight:  extra weight on target peak locations (>1 = penalize missing hits)
    returns:      scalar onset contour loss
    """
    if sigmas is None:
        sigmas = [1.0, sigma, sigma * 2]

    # Embeddings
    soft_w = torch.softmax(pred / tau, dim=-1)          # (N, T, C)
    pred_emb = soft_w @ codebook                         # (N, T, d)
    target_emb = codebook[targets]                       # (N, T, d)

    _, pred_onset = _energy_and_onsets(pred_emb)         # (N, T-1)
    _, target_onset = _energy_and_onsets(target_emb)

    # Multi-scale onset comparison
    total_loss = 0.0
    for s in sigmas:
        pred_s = _smooth(pred_onset, s, pred.device)
        target_s = _smooth(target_onset, s, pred.device)
        total_loss = total_loss + F.mse_loss(pred_s, target_s)
    total_loss = total_loss / len(sigmas)

    # Peak presence loss — weight errors higher where target has strong onsets
    # This specifically penalizes missing drum hits
    if peak_weight > 1.0:
        target_smooth = _smooth(target_onset, sigmas[0], pred.device)
        pred_smooth = _smooth(pred_onset, sigmas[0], pred.device)

        # Soft peak mask: normalize target onset strength to [0, 1]
        t_max = target_smooth.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        peak_mask = target_smooth / t_max                # (N, T-1) in [0, 1]

        # Weighted MSE: base weight 1 + (peak_weight-1) at peak locations
        weights = 1.0 + (peak_weight - 1.0) * peak_mask
        weighted_err = weights * (pred_smooth - target_smooth) ** 2
        total_loss = total_loss + weighted_err.mean()

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Mel detail loss
# ─────────────────────────────────────────────────────────────────────────────


def mel_detail_loss(pred, targets, codebook, decoder, tau=0.5, log_weight=0.5):
    """
    Mel-space reconstruction loss via frozen VQVAE decoder.

    Decodes both the soft-predicted embeddings and the hard target tokens
    through the VQVAE decoder to full mel spectrograms, then compares
    with L1 + log-scale L1 (to capture quiet spectral detail).

    pred:       (N, T, codebook_size)  logits
    targets:    (N, T)                 integer token ids
    codebook:   (codebook_size, d)     frozen VQ buffer (d=64)
    decoder:    VQVAE decoder module (frozen, no grad)
    tau:        softmax temperature for soft lookup
    log_weight: blend between linear L1 and log-mag L1 (0=linear only, 1=log only)
    returns:    scalar mel reconstruction loss
    """
    # Soft predicted embeddings → decoder → mel
    soft_w = torch.softmax(pred / tau, dim=-1)          # (N, T, C)
    pred_emb = soft_w @ codebook                         # (N, T, d=64)
    pred_mel = decoder(pred_emb.permute(0, 2, 1))       # (N, 80, T_mel)

    # Hard target embeddings → decoder → mel (detached)
    target_emb = codebook[targets]                       # (N, T, d=64)
    target_mel = decoder(target_emb.permute(0, 2, 1)).detach()  # (N, 80, T_mel)

    # Linear L1 — overall magnitude
    l1 = F.l1_loss(pred_mel, target_mel)

    # Log-magnitude L1 — emphasizes fine spectral detail in quiet regions
    eps = 1e-5
    log_l1 = F.l1_loss(
        torch.log(pred_mel.abs() + eps),
        torch.log(target_mel.abs() + eps),
    )

    return (1 - log_weight) * l1 + log_weight * log_l1
