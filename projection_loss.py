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
    # accepts only these kwargs; others will be ignored by factory unless strict=True
    def __init__(self, **kwargs):
        pass

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)
        # normalize zs and zs_tilde
        zs = F.normalize(zs, dim=-1)
        zs_tilde = F.normalize(zs_tilde, dim=-1)
        # compute cosine similarity
        cos_sim = (zs * zs_tilde).sum(dim=-1)    # [B,T]
        loss = -cos_sim
        return loss.mean()


# =========================================
# MSE with Spatial Normalization
# =========================================

def spatial_zscore(feat: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score normalization along spatial dimension.

    Args:
        feat: (B, T, D) patch tokens
        alpha: scaling factor for mean subtraction (default 1.0)
        eps: small constant for numerical stability

    Returns:
        Normalized features (B, T, D)
    """
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True)
    return (feat - alpha * mean) / (std + eps)


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
            return spatial_zscore(feat, alpha=self.zscore_alpha, eps=self.eps)
        else:
            raise ValueError(f"Unknown spnorm_method: {self.spnorm_method}")

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)
        # Apply spatial normalization to both
        zs_norm = self._apply_spnorm(zs)
        zs_tilde_norm = self._apply_spnorm(zs_tilde)
        # Compute MSE loss
        loss = F.mse_loss(zs_norm, zs_tilde_norm)
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
            return spatial_zscore(feat, alpha=self.zscore_alpha, eps=self.eps)
        else:
            raise ValueError(f"Unknown spnorm_method: {self.spnorm_method}")

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)
        
        # Get d_alpha_t and d_sigma_t from kwargs (passed from loss function)
        d_alpha_t = kwargs.get('d_alpha_t', None)
        d_sigma_t = kwargs.get('d_sigma_t', None)
        
        if d_alpha_t is None or d_sigma_t is None:
            raise ValueError("mse_v loss requires d_alpha_t and d_sigma_t in kwargs. "
                           "Make sure the loss function passes these values.")
        
        # Generate noise in feature space (same shape as zs)
        noise_feat = torch.randn_like(zs)
        
        # Compute velocity target in feature space: v = d_alpha_t * zs + d_sigma_t * noise
        # d_alpha_t and d_sigma_t are scalars or (B, 1, 1, 1) shaped
        # We need to reshape them for broadcasting with (B, T, D)
        if isinstance(d_alpha_t, torch.Tensor):
            # Reshape from (B, 1, 1, 1) to (B, 1, 1) for broadcasting
            d_alpha_t = d_alpha_t.view(d_alpha_t.shape[0], 1, 1)
            d_sigma_t = d_sigma_t.view(d_sigma_t.shape[0], 1, 1)
        
        z_target = d_alpha_t * zs + d_sigma_t * noise_feat
        
        # Apply spatial normalization to both
        z_target_norm = self._apply_spnorm(z_target)
        zs_tilde_norm = self._apply_spnorm(zs_tilde)
        
        # Compute MSE loss
        loss = F.mse_loss(zs_tilde_norm, z_target_norm)
        return loss
