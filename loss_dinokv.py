"""
Loss function for SiT with DINO-KV Distillation.

Extends iREPA's SILoss with attention distillation loss support.
"""
import torch
import numpy as np
import torch.nn.functional as F
import projection_loss as pl


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SILossWithDINOKV:
    """
    Loss function for SiT with DINO-KV distillation.
    
    Combines:
    1. Denoising loss (v-prediction)
    2. REPA projection loss (cosine, nt-xent, etc.)
    3. Attention distillation loss (from model output, Stage 2 only)
    """
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            projection_loss_type="cosine",
            projection_loss_kwargs={},
            proj_coeff=[0.5],
            distill_coeff=1.0,
        ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.distill_coeff = distill_coeff

        # parse projection loss type and coeff
        self.projection_loss_type = [elem.strip() for elem in projection_loss_type.split(",") if elem.strip()]
        self.proj_coeff = [float(elem.strip()) for elem in proj_coeff.split(",") if elem.strip()]
        assert len(self.projection_loss_type) == len(self.proj_coeff), \
            f"len(self.projection_loss_type) - {len(self.projection_loss_type)} != len(self.proj_coeff) - {len(self.proj_coeff)}"
        self.projection_loss_kwargs = projection_loss_kwargs
        # create projection loss
        self.projection_loss = [
            pl.make_projection_loss(projection_loss_type, **projection_loss_kwargs)
            for projection_loss_type in self.projection_loss_type
        ]
        assert len(self.projection_loss) == len(self.proj_coeff), \
            f"len(self.projection_loss) - {len(self.projection_loss)} != len(self.proj_coeff) - {len(self.proj_coeff)}"
        

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        """
        Compute loss for SiT with DINO-KV distillation.
        
        Args:
            model: SiTWithDINOKV model
            images: Latent images (B, C, H, W)
            model_kwargs: Dict with 'y', 'dino_kv_list', 'stage'
            zs: List of encoder features for REPA projection loss
            
        Returns:
            denoising_loss: Per-sample denoising loss
            proj_loss: Scalar projection loss
            distill_loss: Scalar attention distillation loss
            loss_dict: Dict with individual loss values
        """
        if model_kwargs is None:
            model_kwargs = {}
        if zs is None:
            zs = []

        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError()
        
        # Forward pass - model returns (output, zs_tilde, zs_original, distill_loss)
        model_output, zs_tilde, zs_tilde_original, distill_loss = model(
            model_input, time_input.flatten(), **model_kwargs
        )
        
        # Denoising loss
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # Projection loss (REPA)
        total_proj_loss = 0.
        proj_loss_dict = {}
        for proj_loss_name, proj_loss_fn, coeff in zip(self.projection_loss_type, self.projection_loss, self.proj_coeff):
            proj_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)
            if len(zs) > 0 and zs_tilde is not None and len(zs_tilde) > 0:
                for z, z_tilde, z_tilde_original in zip(zs, zs_tilde, zs_tilde_original):
                    proj_loss = proj_loss + proj_loss_fn(z, z_tilde, z_tilde_original, 
                                                         alpha_t=alpha_t, sigma_t=sigma_t,
                                                         d_alpha_t=d_alpha_t, d_sigma_t=d_sigma_t)
                proj_loss /= len(zs)
            proj_loss_dict[proj_loss_name] = proj_loss.detach().item()
            proj_loss_dict[f"{proj_loss_name}_weighted"] = proj_loss.detach().item() * coeff
            total_proj_loss = total_proj_loss + coeff * proj_loss
        
        # Handle distillation loss
        if distill_loss is None or not isinstance(distill_loss, torch.Tensor):
            distill_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)
        
        # Aggregate loss dict
        loss_dict = proj_loss_dict.copy()
        loss_dict["distill_loss"] = distill_loss.detach().item() if isinstance(distill_loss, torch.Tensor) else distill_loss
        
        return denoising_loss, total_proj_loss, distill_loss * self.distill_coeff, loss_dict
