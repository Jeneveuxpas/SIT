import inspect
import torch
import torch.nn.functional as F
from typing import Callable, Dict

# =========================================
# Registry / Factory with kwarg filtering
# =========================================

_LOSS_REGISTRY: Dict[str, Callable[..., "ProjectionLoss"]] = {}

def register_loss(name: str):
    def deco(cls):
        _LOSS_REGISTRY[name] = cls
        cls.__loss_name__ = name
        return cls
    return deco

def available_losses():
    return sorted(_LOSS_REGISTRY.keys())

def _apply_aliases(cls, kwargs: dict) -> dict:
    # Optional per-class alias map, e.g. {"temperature": "tau", "t": "tau"}
    aliases = getattr(cls, "KWARG_ALIASES", None) or {}
    out = dict(kwargs)
    for a, target in aliases.items():
        if a in out and target not in out:
            out[target] = out.pop(a)
    return out

def make_projection_loss(name: str, strict: bool = False, **kwargs) -> "ProjectionLoss":
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {available_losses()}")
    cls = _LOSS_REGISTRY[name]
    kw = _apply_aliases(cls, kwargs)
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kw.items() if k in sig.parameters}
    unused = {k: v for k, v in kw.items() if k not in sig.parameters}
    if strict and unused:
        raise TypeError(f"Unused kwargs for loss '{name}': {sorted(unused)}")
    return cls(**valid)

# =========================================
# Base
# =========================================

class ProjectionLoss:
    """All projection losses implement __call__(zs, zs_tilde, **kwargs) with tensors shaped [B, T, D]."""
    def _check(self, zs, zs_tilde):
        if zs.ndim != 3 or zs_tilde.ndim != 3:
            raise ValueError(f"zs and zs_tilde must be [B,T,D]; got {zs.shape=} {zs_tilde.shape=}")
        if zs.shape != zs_tilde.shape:
            raise ValueError(f"Shape mismatch: {zs.shape=} vs {zs_tilde.shape=}")

    def __call__(self, zs, zs_tilde, **kwargs):
        raise NotImplementedError

# =========================================
# Cosine
# =========================================

@register_loss("cosine")
class CosineProjectionLoss(ProjectionLoss):
    def __init__(self, spnorm_method: str = "zscore", zscore_alpha: float = 1.0, eps: float = 1e-6, **kwargs):
        self.spnorm_method = spnorm_method
        self.zscore_alpha = zscore_alpha
        self.eps = eps

    def _apply_spnorm(self, feat: torch.Tensor) -> torch.Tensor:
        if self.spnorm_method == "none":
            return feat
        elif self.spnorm_method == "zscore":
            return zscore_norm(feat, dim=1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "zscore_token":
            return zscore_norm(feat, dim=-1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "layernorm":
            return F.layer_norm(feat, normalized_shape=(feat.shape[-1],), eps=self.eps)
        else:
            raise ValueError(f"Unknown spnorm_method: {self.spnorm_method}")

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)
        # cast to float32
        zs = zs.float()
        zs_tilde = zs_tilde.float()
        
        # Step 1: Spatial zscore normalization (per REPA paper)
        zs = self._apply_spnorm(zs)
        
        # Step 2: L2 normalize for cosine similarity
        zs = F.normalize(zs, dim=-1)
        zs_tilde = F.normalize(zs_tilde, dim=-1)
        
        # compute cosine similarity
        cos_sim = (zs * zs_tilde).sum(dim=-1)    # [B,T]
        loss = -cos_sim
        return loss.mean()


# =========================================
# MSE with Spatial Normalization
# =========================================

from utils import zscore_norm  # Import from utils to avoid code duplication


@register_loss("mse")
class MSEProjectionLoss(ProjectionLoss):
    """
    MSE loss with optional SpatialNormalization (zscore) applied to both inputs.
    
    Args:
        spnorm_method: "none" or "zscore" (default: "zscore")
        zscore_alpha: scaling factor for zscore normalization (default: 1.0)
        eps: small constant for numerical stability (default: 1e-6)
    """
    KWARG_ALIASES = {"spnorm": "spnorm_method"}
    
    def __init__(self, spnorm_method: str = "zscore", zscore_alpha: float = 1.0, eps: float = 1e-6, **kwargs):
        self.spnorm_method = spnorm_method
        self.zscore_alpha = zscore_alpha
        self.eps = eps

    def _apply_spnorm(self, feat: torch.Tensor) -> torch.Tensor:
        if self.spnorm_method == "none":
            return feat
        elif self.spnorm_method == "zscore":
            # feat is (B, T, D), we normalize over spatial dim T (dim=1)
            return zscore_norm(feat, dim=1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "zscore_token":
            # Normalize over feature dim D (dim=-1)
            return zscore_norm(feat, dim=-1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "layernorm":
            # Standard LayerNorm over feature dim D
            return F.layer_norm(feat, normalized_shape=(feat.shape[-1],), eps=self.eps)
        else:
            raise ValueError(f"Unknown spnorm_method: {self.spnorm_method}")

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)
        # cast to float32
        zs = zs.float()
        zs_tilde = zs_tilde.float()
        
        # Apply spatial normalization to both
        zs_norm = self._apply_spnorm(zs)
        # Compute MSE loss
        loss = F.mse_loss(zs_norm, zs_tilde)
        return loss


# =========================================
# MSE-V: Velocity prediction in feature space
# =========================================

@register_loss("mse_v")
class MSEVelocityProjectionLoss(ProjectionLoss):
    """
    MSE loss for velocity prediction in feature space.
    
    Computes: MSE(z_tilde, d_alpha_t * zs + d_sigma_t * noise_feat)
    
    This is analogous to v-prediction in latent space, but applied to
    encoder features. Requires passing d_alpha_t and d_sigma_t through kwargs.
    
    Args:
        spnorm_method: "none" or "zscore" (default: "zscore")
        zscore_alpha: scaling factor for zscore normalization (default: 1.0)
        eps: small constant for numerical stability (default: 1e-6)
    """
    KWARG_ALIASES = {"spnorm": "spnorm_method"}
    
    def __init__(self, spnorm_method: str = "zscore", zscore_alpha: float = 1.0, eps: float = 1e-6, **kwargs):
        self.spnorm_method = spnorm_method
        self.zscore_alpha = zscore_alpha
        self.eps = eps

    def _apply_spnorm(self, feat: torch.Tensor) -> torch.Tensor:
        if self.spnorm_method == "none":
            return feat
        elif self.spnorm_method == "zscore":
            return zscore_norm(feat, dim=1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "zscore_token":
            return zscore_norm(feat, dim=-1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "layernorm":
            return F.layer_norm(feat, normalized_shape=(feat.shape[-1],), eps=self.eps)
        else:
            raise ValueError(f"Unknown spnorm_method: {self.spnorm_method}")

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        # cast to float32
        zs = zs.float()
        zs_tilde = zs_tilde.float()
        
        # Get d_alpha_t and d_sigma_t from kwargs (passed from loss function)
        d_alpha_t = kwargs.get('d_alpha_t', None)
        d_sigma_t = kwargs.get('d_sigma_t', None)
        
        if d_alpha_t is None or d_sigma_t is None:
            raise ValueError("mse_v loss requires d_alpha_t and d_sigma_t in kwargs. "
                             "Make sure the loss function passes these values.")
        
        # Generate noise in feature space (same shape as zs)
        noise_feat = torch.randn_like(zs)
        
        # Reshape d_alpha_t and d_sigma_t for broadcasting with (B, T, D)
        if isinstance(d_alpha_t, torch.Tensor):
            d_alpha_t = d_alpha_t.view(d_alpha_t.shape[0], 1, 1).float()
            d_sigma_t = d_sigma_t.view(d_sigma_t.shape[0], 1, 1).float()

        # Normalize encoder features for stable target scale
        zs_norm = self._apply_spnorm(zs)
        
        # Compute velocity target in feature space: v = d_alpha_t * zs + d_sigma_t * noise
        z_target = d_alpha_t * zs_norm + d_sigma_t * noise_feat
        
        # Compute MSE loss (zs_tilde not normalized - projector learns direct mapping)
        loss = F.mse_loss(zs_tilde, z_target)
        return loss



@register_loss("mse_v_norm")
class MSEVelocityNormProjectionLoss(ProjectionLoss):
    """
    MSE loss for velocity prediction in feature space with normalization.

    Computes: MSE(zscore(z_tilde), d_alpha_t * zscore(zs) + d_sigma_t * noise_feat)

    This is analogous to v-prediction in latent space, but applied to
    encoder features with normalization on both sides.
    Requires passing d_alpha_t and d_sigma_t through kwargs.

    Args:
        spnorm_method: "none" or "zscore" (default: "zscore")
        zscore_alpha: scaling factor for zscore normalization (default: 1.0)
        eps: small constant for numerical stability (default: 1e-6)

    Enhanced numerical stability for multi-GPU training.
    """
    KWARG_ALIASES = {"spnorm": "spnorm_method"}

    def __init__(self, spnorm_method: str = "zscore", zscore_alpha: float = 1.0, eps: float = 1e-6, **kwargs):
        self.spnorm_method = spnorm_method
        self.zscore_alpha = zscore_alpha
        self.eps = eps

    def _apply_spnorm(self, feat: torch.Tensor) -> torch.Tensor:
        if self.spnorm_method == "none":
            return feat
        elif self.spnorm_method == "zscore":
            return zscore_norm(feat, dim=1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "zscore_token":
            return zscore_norm(feat, dim=-1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "layernorm":
            return F.layer_norm(feat, normalized_shape=(feat.shape[-1],), eps=self.eps)
        else:
            raise ValueError(f"Unknown spnorm_method: {self.spnorm_method}")

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        # Force float32 computation for numerical stability (even if model uses bf16)
        zs = zs.float()
        zs_tilde = zs_tilde.float()

        # Get d_alpha_t and d_sigma_t from kwargs (passed from loss function)
        d_alpha_t = kwargs.get('d_alpha_t', None)
        d_sigma_t = kwargs.get('d_sigma_t', None)

        if d_alpha_t is None or d_sigma_t is None:
            raise ValueError("mse_v_norm loss requires d_alpha_t and d_sigma_t in kwargs. "
                             "Make sure the loss function passes these values.")

        # Generate noise in feature space (same shape as zs)
        noise_feat = torch.randn_like(zs)

        # Reshape d_alpha_t and d_sigma_t for broadcasting with (B, T, D)
        if isinstance(d_alpha_t, torch.Tensor):
            d_alpha_t = d_alpha_t.view(d_alpha_t.shape[0], 1, 1).float()
            d_sigma_t = d_sigma_t.view(d_sigma_t.shape[0], 1, 1).float()

        # Normalize encoder features for stable target scale
        zs_norm = self._apply_spnorm(zs)

        # Normalize model output zs_tilde (core of the method)
        zs_tilde_norm = self._apply_spnorm(zs_tilde)

        # Compute velocity target in feature space: v = d_alpha_t * zs + d_sigma_t * noise
        z_target = d_alpha_t * zs_norm + d_sigma_t * noise_feat

        # Compute MSE loss between normalized model output and target
        loss = F.mse_loss(zs_tilde_norm, z_target)

        # Safety net: detect and handle abnormal loss values
        # Lower threshold from 100 to 10 for earlier detection
        if torch.isnan(loss) or torch.isinf(loss) or loss > 10.0:
            # This batch has numerical issues, skip it to prevent training collapse
            print(f"[WARNING] Detected abnormal projection loss: {loss.item():.4f}, skipping this batch")
            loss = torch.tensor(0.0, device=zs.device, dtype=torch.float32)

        return loss
# =========================================
# MSE-Noisy: Noisy interpolation loss in feature space
# =========================================

@register_loss("mse_noisy")
class MSENoisyProjectionLoss(ProjectionLoss):
    """
    MSE loss for noisy interpolation in feature space.
    
    Computes: MSE(alpha_t * zs + sigma_t * noise_feat, zs_tilde)
    
    This aligns the encoder's representation for noisy states.
    Target is the interpolation between clean features and noise,
    matching the noisy latent construction in diffusion/flow matching.
    
    Requires passing alpha_t and sigma_t through kwargs.
    
    Args:
        spnorm_method: "none" or "zscore" (default: "zscore")
        zscore_alpha: scaling factor for zscore normalization (default: 1.0)
        eps: small constant for numerical stability (default: 1e-6)
    """
    KWARG_ALIASES = {"spnorm": "spnorm_method"}
    
    def __init__(self, spnorm_method: str = "zscore", zscore_alpha: float = 1.0, eps: float = 1e-6, **kwargs):
        self.spnorm_method = spnorm_method
        self.zscore_alpha = zscore_alpha
        self.eps = eps

    def _apply_spnorm(self, feat: torch.Tensor) -> torch.Tensor:
        if self.spnorm_method == "none":
            return feat
        elif self.spnorm_method == "zscore":
            return zscore_norm(feat, dim=1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "zscore_token":
            return zscore_norm(feat, dim=-1, alpha=self.zscore_alpha, eps=self.eps)
        elif self.spnorm_method == "layernorm":
            return F.layer_norm(feat, normalized_shape=(feat.shape[-1],), eps=self.eps)
        else:
            raise ValueError(f"Unknown spnorm_method: {self.spnorm_method}")

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)
        
        # cast to float32
        zs = zs.float()
        zs_tilde = zs_tilde.float()

        # Get alpha_t and sigma_t from kwargs (passed from loss function)
        alpha_t = kwargs.get('alpha_t', None)
        sigma_t = kwargs.get('sigma_t', None)
        
        if alpha_t is None or sigma_t is None:
            raise ValueError("mse_noisy loss requires alpha_t and sigma_t in kwargs. "
                             "Make sure the loss function passes these values.")
        
        # Generate noise in feature space (same shape as zs)
        noise_feat = torch.randn_like(zs)
        
        # Reshape alpha_t and sigma_t for broadcasting with (B, T, D)
        # They are typically (B, 1, 1, 1) shaped for image latents
        if isinstance(alpha_t, torch.Tensor):
            alpha_t = alpha_t.view(alpha_t.shape[0], 1, 1).float()
            sigma_t = sigma_t.view(sigma_t.shape[0], 1, 1).float()
        
        # Normalize zs FIRST, then add noise (for better scale matching with gaussian noise)
        zs_norm = self._apply_spnorm(zs)
        z_noisy = alpha_t * zs_norm + sigma_t * noise_feat
        
        # Compute MSE loss (zs_tilde not normalized - projector learns direct mapping)
        loss = F.mse_loss(z_noisy, zs_tilde)
        return loss
