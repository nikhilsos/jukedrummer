"""
Loss functions for JukeDrummer language model training.

Available losses:
  - focal_loss      : focal cross-entropy, focuses on hard tokens
  - perceptual_loss : soft codebook embedding MSE
  - fad_loss        : Fréchet distance in VQ codebook embedding space
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
        pred.reshape(-1, bins), targets.reshape(-1), reduction='none'
    )
    if gamma > 0:
        pt   = torch.exp(-ce_nats)
        loss = ((1 - pt) ** gamma * ce_nats).mean()
    else:
        loss = ce_nats.mean()
    return loss / np.log(2.)


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
    soft_w     = torch.softmax(pred / tau, dim=-1)   # (N, T, C)
    soft_emb   = soft_w @ codebook                    # (N, T, d)
    target_emb = codebook[targets]                    # (N, T, d)
    return F.mse_loss(soft_emb, target_emb)


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
    sqrt_prod    = _matrix_sqrt(sigma_real @ sigma_gen)
    trace_term   = torch.trace(sigma_gen + sigma_real - 2.0 * sqrt_prod)
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
    soft_w  = torch.softmax(pred / tau, dim=-1)
    gen_emb = (soft_w @ codebook).reshape(-1, d)          # (N*T, d)

    # Real embeddings (no gradient — we match the real distribution, not move it)
    real_emb = codebook[targets].reshape(-1, d).detach()  # (N*T, d)

    mu_gen  = gen_emb.mean(dim=0)
    mu_real = real_emb.mean(dim=0)

    n = gen_emb.shape[0]
    gen_c  = gen_emb  - mu_gen
    real_c = real_emb - mu_real

    if diagonal:
        # Variance-only approximation — avoids matrix sqrt, very stable
        var_gen  = gen_c.pow(2).mean(dim=0)    # (d,)
        var_real = real_c.pow(2).mean(dim=0)   # (d,)
        mean_sq  = (mu_gen - mu_real).pow(2).sum()
        var_term = (var_gen.sqrt() - var_real.sqrt()).pow(2).sum()
        return mean_sq + var_term
    else:
        sigma_gen  = (gen_c.T  @ gen_c)  / (n - 1)   # (d, d)
        sigma_real = (real_c.T @ real_c) / (n - 1)    # (d, d)
        return _frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
