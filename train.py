"""
Training script for SiT with Encoder KV Distillation (sota iREPA version).

Two-stage training:
- Stage 1: Q_SiT @ K_Enc, V_Enc (linear projection trainable)
- Stage 2: Q_SiT @ K_SiT, V_SiT + logits distillation (projection detached)

Based on sota/iREPA train.py with Encoder-KV integration.
"""
import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import math


import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.autoencoder import VAE_F8D4
from models.sit_encoder import SiT_EncoderKV_models 
from models.encoder_adapter import EncoderKVExtractor
from loss import SILossWithEncoderKV
from vision_encoder import load_encoders

from dataset import HFImgLatentDataset, HFLatentDataset, ImageFolderLatentDataset
import wandb
from torchvision.utils import make_grid
from utils import ALL_SPNORM_METHODS
    
#################################################################################
#                                  Utils                                       #
#################################################################################
def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def safe_unwrap_model(model):
    """
    Safely unwrap model from DDP and torch.compile wrappers.
    This avoids issues with accelerate's unwrap_model causing KeyErrors on _orig_mod.
    """
    # Unwrap DDP first
    if hasattr(model, 'module'):
        model = model.module
    # Unwrap torch.compile (Iteratively check for _orig_mod to handle multiple compilations if any)
    while hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def get_current_lr(optimizer) -> float:
    """Return current LR from the first optimizer parameter group."""
    if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    # Fallback for wrapped optimizer objects.
    inner_optimizer = getattr(optimizer, "optimizer", None)
    if inner_optimizer is not None and hasattr(inner_optimizer, "param_groups") and len(inner_optimizer.param_groups) > 0:
        return float(inner_optimizer.param_groups[0].get("lr", 0.0))
    return 0.0

def parse_layer_indices(indices_str: str) -> list:
    """Parse comma-separated layer indices string to list of ints (1-based -> 0-based)."""
    return [int(x.strip()) - 1 for x in indices_str.split(',')]


def get_loss_stop_multiplier(step: int, stop_step: int = None, fade_steps: int = 0) -> float:
    """
    Compute loss multiplier for stage-wise termination.

    Args:
        step: Current global training step.
        stop_step: Absolute step to begin turning off this loss branch.
            None (or negative) keeps the branch always on.
        fade_steps: If >0, use cosine fade from 1->0 after stop_step.
            If 0, hard-stop at stop_step.

    Returns:
        Scalar multiplier in [0, 1].
    """
    if stop_step is None or stop_step < 0:
        return 1.0
    if step < stop_step:
        return 1.0
    if fade_steps <= 0:
        return 0.0

    progress = min(1.0, (step - stop_step) / max(1, fade_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def get_sit_block_probe_params(model, layer_idx_1based: int):
    """
    Select lightweight probe parameters from a specific SiT block.
    We use attention projection weights to keep monitoring cost low.
    """
    if not hasattr(model, "blocks"):
        return []

    block_idx = layer_idx_1based - 1
    if block_idx < 0 or block_idx >= len(model.blocks):
        return []

    block = model.blocks[block_idx]
    selected = []
    for name, p in block.named_parameters():
        if not p.requires_grad:
            continue
        if name in {"attn.qkv.weight", "attn.proj.weight"}:
            selected.append(p)

    # Fallback in case module naming differs.
    if len(selected) == 0:
        for name, p in block.named_parameters():
            if p.requires_grad and name.startswith("attn."):
                selected.append(p)
    return selected


def compute_grad_cosine(loss_main: torch.Tensor, loss_aux: torch.Tensor, params):
    """
    Compute cosine similarity between gradients of two losses
    on a fixed probe parameter set.
    """
    if len(params) == 0:
        return None
    if not isinstance(loss_main, torch.Tensor) or not isinstance(loss_aux, torch.Tensor):
        return None
    if not loss_main.requires_grad or not loss_aux.requires_grad:
        return None

    grads_main = torch.autograd.grad(
        loss_main, params, retain_graph=True, allow_unused=True
    )
    grads_aux = torch.autograd.grad(
        loss_aux, params, retain_graph=True, allow_unused=True
    )

    device = params[0].device
    dot = torch.tensor(0.0, device=device)
    main_norm_sq = torch.tensor(0.0, device=device)
    aux_norm_sq = torch.tensor(0.0, device=device)
    used = 0

    for g_main, g_aux in zip(grads_main, grads_aux):
        if g_main is None or g_aux is None:
            continue
        gm = g_main.detach().float().reshape(-1)
        ga = g_aux.detach().float().reshape(-1)
        dot = dot + torch.dot(gm, ga)
        main_norm_sq = main_norm_sq + torch.dot(gm, gm)
        aux_norm_sq = aux_norm_sq + torch.dot(ga, ga)
        used += 1

    if used == 0:
        return None
    if main_norm_sq.item() <= 0 or aux_norm_sq.item() <= 0:
        return None

    main_norm = torch.sqrt(main_norm_sq)
    aux_norm = torch.sqrt(aux_norm_sq)
    cosine = dot / (main_norm * aux_norm + 1e-12)
    return cosine, main_norm, aux_norm


def reduce_scalar_mean(accelerator: Accelerator, value, device: torch.device):
    """
    Reduce a scalar metric across all processes with nan-safe mean.
    Returns None when all processes report NaN.
    """
    if value is None:
        t = torch.tensor(float("nan"), device=device, dtype=torch.float32)
    elif isinstance(value, torch.Tensor):
        t = value.detach().float()
        if t.ndim > 0:
            t = t.mean()
    else:
        t = torch.tensor(float(value), device=device, dtype=torch.float32)

    gathered = accelerator.gather(t.view(1))
    valid = ~torch.isnan(gathered)
    if valid.any():
        return gathered[valid].mean().item()
    return None

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    
    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = f"{save_dir}/checkpoints"

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Parse layer indices
    enc_layer_indices = parse_layer_indices(args.enc_layer_indices)
    sit_layer_indices = parse_layer_indices(args.sit_layer_indices)
    assert len(enc_layer_indices) == len(sit_layer_indices), \
        "Encoder and SiT layer indices must have same length"

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    in_channels = 4
    # Load encoders (generic)
    # We use the first encoder for K/V distillation if available
    encoders = load_encoders(
        args.enc_type, device, args.resolution, accelerator=accelerator
    )
    
    # Create Encoder K/V extractor first to detect dimension and heads (only if KV distillation enabled)
    encoder_kv_extractor = None
    enc_dim = None
    enc_heads = None
    if args.use_kv and len(encoders) > 0:
        encoder_kv_extractor = EncoderKVExtractor(encoders[0].model, enc_layer_indices)
        encoder_kv_extractor.eval()
        
        # Auto-detect enc_dim and enc_heads from the encoder layer
        if len(enc_layer_indices) > 0:
            enc_dim = encoder_kv_extractor.get_layer_dim(enc_layer_indices[0])
            enc_heads = encoder_kv_extractor.get_layer_heads(enc_layer_indices[0])
            
            # Fallback to model embedding dim if detection failed
            if enc_dim == 0:
                enc_dim = encoders[0].embed_dim
            
            # Fallback for heads: try enc_dim // 64 (common head_dim)
            if enc_heads == 0:
                enc_heads = enc_dim // 64 if enc_dim >= 64 else 1
            
            if accelerator.is_main_process:
                logger.info(f"Auto-detected encoder config: enc_dim={enc_dim}, enc_heads={enc_heads}")
                    
    # z_dims reflects the TARGET REPA dimension (encoder final output)
    z_dims = [encoder.embed_dim for encoder in encoders]
    
    block_kwargs = {
        "fused_attn": args.fused_attn, 
        "qk_norm": args.qk_norm,
    }
    
    # Create SiT with Encoder KV
    model = SiT_EncoderKV_models[args.model](
        path_type=args.path_type,
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        projection_layer_type=args.projection_layer_type,
        proj_kwargs_kernel_size=args.proj_kwargs_kernel_size,
        enc_layer_indices=enc_layer_indices,
        sit_layer_indices=sit_layer_indices,
        enc_dim=enc_dim,
        enc_heads=enc_heads,
        kv_proj_type=args.kv_proj_type,
        kv_proj_hidden_dim=args.kv_proj_hidden_dim,
        kv_proj_kernel_size=args.kv_proj_kernel_size,
        kv_norm_type=args.kv_norm_type,
        kv_zscore_alpha=args.kv_zscore_alpha,
        distill_temperature=args.distill_temperature,
        kv_distill_snr_gamma=args.kv_distill_snr_gamma,
        kv_distill_min_weight=args.kv_distill_min_weight,
        attn_loss_weight=args.attn_loss_weight,
        kv_loss_weight=args.kv_loss_weight,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Load the VAE weights correctly, load the BN stats (sota version style)
    vae = VAE_F8D4().to(device).eval()
    vae_state_dict = torch.load("pretrained_models/sdvae-ft-mse-f8d4.pt", map_location=device, weights_only=False)
    vae.load_state_dict(vae_state_dict)

    latents_stats = torch.load("pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt", map_location=device, weights_only=False)
    latents_scale = latents_stats['latents_scale'].to(device).view(1, -1, 1, 1)
    latents_bias = latents_stats['latents_bias'].to(device).view(1, -1, 1, 1)

    projection_loss_kwargs = {
        'spnorm_method': args.spnorm_method,
        'zscore_alpha': args.zscore_alpha,
    }

    loss_fn = SILossWithEncoderKV(
        prediction=args.prediction,
        path_type=args.path_type, 
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        projection_loss_type=args.projection_loss_type,
        projection_loss_kwargs=projection_loss_kwargs,
        proj_coeff=args.proj_coeff,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Encoder KV: {len(enc_layer_indices)} layer pairs")
        logger.info(f"Encoder layers: {enc_layer_indices} -> SiT layers: {sit_layer_indices}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup scheduler
    lr_scheduler = None
    if args.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        def cosine_schedule_with_warmup(step):
            if step < args.lr_warmup_steps:
                return float(step) / float(max(1, args.lr_warmup_steps))
            
            decay_start = args.lr_decay_start_step if args.lr_decay_start_step is not None else args.lr_warmup_steps
            if step < decay_start:
                return 1.0
                
            progress = float(step - decay_start) / float(max(1, args.max_train_steps - decay_start))
            
            # Calculate cosine decay multiplier (starts at 1.0, ends at 0.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Scale range from [min_lr, base_lr] instead of [0, base_lr]
            # Multiplier = min_ratio + (1 - min_ratio) * cosine_decay
            min_ratio = args.min_lr / args.learning_rate
            return min_ratio + (1.0 - min_ratio) * cosine_decay
            
        lr_scheduler = LambdaLR(optimizer, lr_lambda=cosine_schedule_with_warmup)
    elif args.lr_scheduler == "constant":
        pass  # Default behavior
    
    # Setup data
    if args.repa_loss:
        try:
            # We can preprocess ImageNet 256/512 here, and directly load from disk
            train_dataset = HFImgLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, split="train")
        except Exception as e:
            print(f"Error loading HFImgLatentDataset: {e}")
            print("Falling back to ImageFolderLatentDataset")
            train_dataset = ImageFolderLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, resolution=args.resolution, split="train")
    else:
        train_dataset = HFLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, split="train")
    print(train_dataset)

    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    grad_norm = 0.0  # Initialize grad_norm to avoid undefined variable error
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{checkpoint_dir}/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        
        # Load scheduler state if available
        if lr_scheduler is not None and 'scheduler' in ckpt:
            lr_scheduler.load_state_dict(ckpt['scheduler'])
            
        if args.resume_override_lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = float(args.resume_override_lr)
            if accelerator.is_main_process:
                logger.info(
                    f"Resumed from step {global_step} and override learning rate to {args.resume_override_lr:.6g}"
                )
        elif accelerator.is_main_process:
            logger.info(
                f"Resumed from step {global_step} with optimizer learning rate {get_current_lr(optimizer):.6g}"
            )

    if args.compile:
        # Allow larger cache size for DYNAMO compilation
        torch._dynamo.config.cache_size_limit = 64
        # accumulated_cache_size_limit may not exist in older PyTorch versions
        if hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
            torch._dynamo.config.accumulated_cache_size_limit = 512
        model = torch.compile(model, backend="inductor", mode="default")
        # encoders = [torch.compile(encoder, backend="inductor", mode="default") for encoder in encoders]

        @torch.compile(fullgraph=False)
        def optim_step_fn():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    else:
        def optim_step_fn():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    if lr_scheduler is not None:
        lr_scheduler = accelerator.prepare(lr_scheduler)

    # Optional gradient-direction monitoring on fixed SiT probe layers.
    kv_probe_params = []
    repa_probe_params = []
    grad_window_active = False
    grad_window_remaining = 0
    grad_window_start_step = None
    grad_window_stats = {}
    grad_window_stat_keys = [
        f"grad_cos_main_kv_l{args.grad_monitor_kv_layer}",
        f"grad_norm_main_kv_l{args.grad_monitor_kv_layer}",
        f"grad_norm_kv_l{args.grad_monitor_kv_layer}",
        f"grad_cos_main_repa_l{args.grad_monitor_repa_layer}",
        f"grad_norm_main_repa_l{args.grad_monitor_repa_layer}",
        f"grad_norm_repa_l{args.grad_monitor_repa_layer}",
    ]
    if args.grad_monitor:
        probe_model = safe_unwrap_model(model)
        kv_probe_params = get_sit_block_probe_params(probe_model, args.grad_monitor_kv_layer)
        repa_probe_params = get_sit_block_probe_params(probe_model, args.grad_monitor_repa_layer)
        if accelerator.is_main_process:
            logger.info(
                f"Grad monitor enabled (interval={args.grad_monitor_interval}, "
                f"window_steps={args.grad_monitor_window_steps}): "
                f"KV probe=layer{args.grad_monitor_kv_layer} ({len(kv_probe_params)} params), "
                f"REPA probe=layer{args.grad_monitor_repa_layer} ({len(repa_probe_params)} params)"
            )
            if len(kv_probe_params) == 0:
                logger.warning("Grad monitor: KV probe parameter set is empty.")
            if len(repa_probe_params) == 0:
                logger.warning("Grad monitor: REPA probe parameter set is empty.")

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="iREPA-ENCODERKV", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with:
    sample_batch_size = args.n_samples // accelerator.num_processes
    batch = next(iter(train_dataloader))
    if len(batch) == 3:
        raw_image_sample, gt_xs, gt_labels = batch
    else:
        gt_xs, gt_labels = batch
        raw_image_sample = None
    gt_xs = gt_xs[:sample_batch_size]
    gt_labels = gt_labels[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
    )
    ys = gt_labels.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)

    # No longer need separate processors - encoders handle their own preprocessing

    # define spatial normalization class object
    
    for epoch in range(args.epochs):

        model.train()
        for batch in train_dataloader:
            grad_monitor_logs = {}
            if len(batch) == 3:
                raw_image, x, y = batch
                raw_image = raw_image.to(device)
            else:
                x, y = batch
                raw_image = None
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)

            # Auto-stage switching (by step count)
            current_stage = 1 if global_step < args.stage1_steps else 2
            current_align_mode = args.align_mode

            # Stage-wise termination gates
            # REPA: controls projection loss branch
            repa_gate = get_loss_stop_multiplier(
                step=global_step,
                stop_step=args.repa_stop_step,
                fade_steps=args.repa_stop_fade_steps,
            )
            # KV gate only affects Stage 2 distillation; Stage 1 KV guidance stays enabled.
            kv_gate = get_loss_stop_multiplier(
                step=global_step,
                stop_step=args.kv_stop_step,
                fade_steps=args.kv_stop_fade_steps,
            )
            repa_active = args.repa_loss and repa_gate > 0.0
            kv_active = args.use_kv and (current_stage == 1 or kv_gate > 0.0)

            # Log stage periodically
            if accelerator.is_main_process and global_step % 1000 == 0:
                logger.info(
                    f"Step {global_step}: stage = {current_stage} (switch at step {args.stage1_steps}), "
                    f"align_mode = {current_align_mode}, repa_gate = {repa_gate:.3f}, kv_gate = {kv_gate:.3f}"
                )

            labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                
                # Extract Encoder K/V and CLS token (only if KV distillation is enabled)
                enc_kv_list = None
                # Extract encoder features for REPA projection loss
                zs = []
                
                # Fast path: single encoder + both losses active -> one encoder forward.
                single_encoder_joint_path = (
                    kv_active and repa_active and len(encoders) == 1 and encoder_kv_extractor is not None
                )

                if single_encoder_joint_path:
                    if raw_image is None:
                        raise ValueError("active REPA/KV requires raw images, but dataset did not return them.")
                    raw_image_enc = encoders[0].preprocess(raw_image)
                    with accelerator.autocast():
                        encoder_kv_extractor.reset_cache()
                        features = encoders[0].forward_features(raw_image_enc)
                        z = features['x_norm_patchtokens']
                        zs.append(z)
                        try:
                            enc_kv_list = encoder_kv_extractor.get_captured_kv_list()
                        except RuntimeError:
                            # Fallback for encoders whose wrapper path does not trigger registered hooks.
                            enc_kv_list, enc_cls = encoder_kv_extractor(raw_image_enc)
                else:
                    if kv_active:
                        if raw_image is None:
                             raise ValueError("use_kv requires raw images, but dataset did not return them (check repa_loss arg).")
                        raw_image_enc = encoders[0].preprocess(raw_image)
                        with accelerator.autocast():
                            enc_kv_list, enc_cls = encoder_kv_extractor(raw_image_enc)
                    
                    if repa_active:
                        with accelerator.autocast():
                            for encoder in encoders:
                                # Preprocess the image using encoder's built-in method
                                raw_image_ = encoder.preprocess(raw_image)

                                # Encode the features
                                # outputs dictionary with keys: 'x_norm_patchtokens' and 'x_norm_clstoken'
                                features = encoder.forward_features(raw_image_) 

                                # normalize spatial features
                                # normalize spatial features (handled in loss now)
                                # Legacy Note: args.cls_token_weight was previously passed but ignored by SpatialNormalization.
                                # We keep it ignored here to match original behavior.
                                z = features['x_norm_patchtokens']

                                # append to list
                                zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(
                    y=labels,
                    enc_kv_list=enc_kv_list if kv_active else None,
                    stage=current_stage if kv_active else 2,  # Skip stage 1 if KV branch is off
                    align_mode=current_align_mode,
                )
                
                denoising_loss, proj_loss_raw, distill_loss_raw, loss_dict = loss_fn(model, x, model_kwargs, zs=zs)
                denoising_loss_mean = denoising_loss.mean()
                proj_loss = proj_loss_raw * repa_gate
                
                # Distillation coefficient: 0 in Stage 1, fixed coefficient in Stage 2.
                base_distill_coeff = args.distill_coeff if current_stage == 2 else 0.0
                current_distill_coeff = base_distill_coeff * kv_gate
                distill_loss = distill_loss_raw * current_distill_coeff

                should_start_grad_window = (
                    args.grad_monitor
                    and accelerator.sync_gradients
                    and global_step > 0
                    and (global_step % args.grad_monitor_interval == 0)
                    and (not grad_window_active)
                )
                if should_start_grad_window:
                    grad_window_active = True
                    grad_window_remaining = args.grad_monitor_window_steps
                    grad_window_start_step = global_step
                    grad_window_stats = {
                        key: {"sum": 0.0, "sum_sq": 0.0, "count": 0}
                        for key in grad_window_stat_keys
                    }

                should_monitor_grad = (
                    args.grad_monitor
                    and accelerator.sync_gradients
                    and grad_window_active
                    and grad_window_remaining > 0
                )
                if should_monitor_grad:
                    kv_grad_stats = compute_grad_cosine(
                        denoising_loss_mean, distill_loss, kv_probe_params
                    )
                    repa_grad_stats = compute_grad_cosine(
                        denoising_loss_mean, proj_loss, repa_probe_params
                    )

                    step_metrics = {}
                    if kv_grad_stats is not None:
                        kv_cos, kv_main_norm, kv_aux_norm = kv_grad_stats
                        step_metrics.update({
                            f"grad_cos_main_kv_l{args.grad_monitor_kv_layer}": kv_cos,
                            f"grad_norm_main_kv_l{args.grad_monitor_kv_layer}": kv_main_norm,
                            f"grad_norm_kv_l{args.grad_monitor_kv_layer}": kv_aux_norm,
                        })
                    if repa_grad_stats is not None:
                        repa_cos, repa_main_norm, repa_aux_norm = repa_grad_stats
                        step_metrics.update({
                            f"grad_cos_main_repa_l{args.grad_monitor_repa_layer}": repa_cos,
                            f"grad_norm_main_repa_l{args.grad_monitor_repa_layer}": repa_main_norm,
                            f"grad_norm_repa_l{args.grad_monitor_repa_layer}": repa_aux_norm,
                        })

                    # Aggregate metrics across all processes before window accumulation.
                    for key in grad_window_stat_keys:
                        value = step_metrics.get(key, None)
                        reduced_value = reduce_scalar_mean(accelerator, value, device)
                        if reduced_value is None:
                            continue
                        stat = grad_window_stats[key]
                        stat["sum"] += reduced_value
                        stat["sum_sq"] += reduced_value * reduced_value
                        stat["count"] += 1

                    grad_window_remaining -= 1
                    if grad_window_remaining == 0:
                        grad_window_active = False
                        window_end_step = global_step
                        grad_monitor_logs["grad_monitor_window_start"] = float(grad_window_start_step)
                        grad_monitor_logs["grad_monitor_window_end"] = float(window_end_step)

                        for key in grad_window_stat_keys:
                            stat = grad_window_stats[key]
                            if stat["count"] <= 0:
                                continue
                            mean_val = stat["sum"] / stat["count"]
                            var_val = max(0.0, stat["sum_sq"] / stat["count"] - mean_val * mean_val)
                            std_val = math.sqrt(var_val)
                            grad_monitor_logs[key] = mean_val
                            grad_monitor_logs[f"{key}_std"] = std_val
                            grad_monitor_logs[f"{key}_n"] = float(stat["count"])

                        if accelerator.is_main_process and len(grad_monitor_logs) > 0:
                            logger.info(
                                f"Step {window_end_step}: grad monitor window {grad_monitor_logs}"
                            )
                
                # Total loss
                loss = denoising_loss_mean + proj_loss + distill_loss
                loss = loss.float()
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optim_step_fn()

                if accelerator.sync_gradients and global_step % args.ema_update_freq == 0:
                    original_model = safe_unwrap_model(model)
                    update_ema(
                        ema,
                        original_model,
                        decay=args.ema_decay
                    )
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        original_model = safe_unwrap_model(model)

                        checkpoint = {
                            "model": original_model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                            "args": args,
                            "steps": global_step,
                        }
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                    from samplers import euler_sampler
                    with torch.no_grad():
                        samples = euler_sampler(
                            model,
                            xT,
                            ys,
                            num_steps=50,
                            cfg_scale=1.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            path_type=args.path_type,
                            heun=False,
                        ).to(torch.float32)
                        samples = vae.decode(samples / latents_scale + latents_bias).sample
                        gt_samples = vae.decode(gt_xs / latents_scale + latents_bias).sample
                        samples = (samples + 1) / 2.
                        gt_samples = (gt_samples + 1) / 2.
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    gt_samples = accelerator.gather(gt_samples.to(torch.float32))

                    stage_name = f"stage{current_stage}"
                    accelerator.log({
                        f"samples_{stage_name}": wandb.Image(array2grid(out_samples)),
                        "gt_samples": wandb.Image(array2grid(gt_samples))
                    })
                    if accelerator.is_main_process:
                        logger.info(f"Generated samples: Stage {current_stage}")

                # Include the loss and grad norms in the logging
                logs = {
                    "loss": loss.detach().item(),
                    "denoising_loss": denoising_loss_mean.detach().item(),
                    "proj_loss": proj_loss.detach().item() if isinstance(proj_loss, torch.Tensor) else proj_loss,
                    "proj_loss_raw": proj_loss_raw.detach().item() if isinstance(proj_loss_raw, torch.Tensor) else proj_loss_raw,
                    "distill_loss": distill_loss.detach().item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
                    "distill_loss_raw": distill_loss_raw.detach().item() if isinstance(distill_loss_raw, torch.Tensor) else distill_loss_raw,
                    "distill_coeff": current_distill_coeff,
                    "distill_coeff_base": base_distill_coeff,
                    "repa_gate": repa_gate,
                    "kv_gate": kv_gate,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "stage": current_stage,
                    "lr": get_current_lr(optimizer),
                }
                logs.update(loss_dict)
                logs.update(grad_monitor_logs)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Cleanup
    if encoder_kv_extractor is not None:
        encoder_kv_extractor.remove_hooks()
    
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training SiT with Encoder KV")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--resume-override-lr", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=256)

    # model
    parser.add_argument("--model", type=str, default="SiT-B/2-EncoderKV")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    # add arg for type of projection layer: allowed mlp | linear | conv
    parser.add_argument("--projection-layer-type", type=str, default="conv", choices=["mlp", "linear", "conv"])
    parser.add_argument("--proj-kwargs-kernel-size", type=int, default=1, choices=[1, 3, 5, 7])

    # Encoder KV specific
    parser.add_argument("--enc-layer-indices", type=str, default="11",
                        help="Comma-separated Encoder layer indices for K/V extraction (1-based, e.g. 1-12)")
    parser.add_argument("--sit-layer-indices", type=str, default="10",
                        help="Comma-separated SiT layer indices for K/V injection (1-based, e.g. 1-12)")
    parser.add_argument("--stage1-steps", type=int, default=30000,
                        help="Number of steps for Stage 1 (e.g., 30000 for 100k total)")
    parser.add_argument("--distill-coeff", type=float, default=1.0,
                        help="Coefficient for distillation loss (0 in Stage 1, this value in Stage 2)")
    parser.add_argument("--align-mode", type=str, default="attn_mse",
                        choices=["logits", "attn_mse", "snr_attn_mse", "attn_kl", "kv_mse", "attn_hybrid"],
                        help="Alignment mode: logits, attn_mse, snr_attn_mse, attn_kl, kv_mse, attn_hybrid")
    parser.add_argument("--distill-temperature", type=float, default=1.0,
                        help="Temperature for attn_kl mode (<1 sharpens distributions, making alignment harder)")
    parser.add_argument("--attn-loss-weight", type=float, default=1.0,
                        help="Weight for attention-output loss in attn_hybrid mode")
    parser.add_argument("--kv-loss-weight", type=float, default=1.0,
                        help="Weight for KV loss term in attn_hybrid mode")
    parser.add_argument("--kv-distill-snr-gamma", type=float, default=1.0,
                        help="Power for SNR-based KV weighting (>1 focuses more on low-noise steps)")
    parser.add_argument("--kv-distill-min-weight", type=float, default=0.0,
                        help="Lower bound for SNR-based KV weighting")
    parser.add_argument("--kv-stop-step", type=int, default=None,
                        help="Absolute step to begin turning off Stage-2 KV distillation (None keeps it on)")
    parser.add_argument("--kv-stop-fade-steps", type=int, default=0,
                        help="Cosine fade length for KV distillation after kv-stop-step (0 = hard stop)")
    parser.add_argument("--kv-proj-type", type=str, default="linear",
                        choices=["linear", "mlp", "conv"],
                        help="Projection type for Encoder K/V: linear, mlp, or conv")
    parser.add_argument("--kv-proj-hidden-dim", type=int, default=None,
                        help="Hidden dimension for MLP projection (default: max(enc_dim, sit_dim))")
    parser.add_argument("--kv-proj-kernel-size", type=int, default=1,
                        help="Kernel size for conv projection (default: 1)")
    parser.add_argument("--kv-norm-type", type=str, default="none",
                        choices=["none", "layernorm", "zscore", "zscore_token", "batchnorm"],
                        help="Normalization type for K/V: zscore=per-spatial, zscore_token=per-token")
    parser.add_argument("--kv-zscore-alpha", type=float, default=1.0, 
                        help="Alpha for z-score normalization: (x - alpha * mean) / std")
    # enc-dim and enc-heads are now auto-detected from encoder
    # dataset
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=100000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr-scheduler", type=str, default="constant", choices=["constant", "cosine"], help="Learning rate scheduler.")
    parser.add_argument("--lr-warmup-steps", type=int, default=0, help="Number of warmup steps for LR scheduler.")
    parser.add_argument("--lr-decay-start-step", type=int, default=None, help="Step to start LR decay (default: after warmup).")
    parser.add_argument("--min-lr", type=float, default=0.0, help="Minimum learning rate for cosine scheduler.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-update-freq", type=int, default=1)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=12)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=str, default="1.0")
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--repa-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-kv", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable KV distillation from encoder (use --no-use-kv to disable)")
    # add loss type
    parser.add_argument("--projection-loss-type", type=str, default="cosine", help="Should be a comma-separated list of projection loss types")
    parser.add_argument("--repa-stop-step", type=int, default=None,
                        help="Absolute step to begin turning off REPA projection loss (None keeps it on)")
    parser.add_argument("--repa-stop-fade-steps", type=int, default=0,
                        help="Cosine fade length for REPA projection loss after repa-stop-step (0 = hard stop)")
    parser.add_argument("--grad-monitor", action=argparse.BooleanOptionalAction, default=False,
                        help="Periodically monitor gradient cosine between denoising and auxiliary losses")
    parser.add_argument("--grad-monitor-interval", type=int, default=5000,
                        help="Gradient monitor interval in steps")
    parser.add_argument("--grad-monitor-window-steps", type=int, default=10,
                        help="Number of consecutive steps to aggregate for each grad monitor window")
    parser.add_argument("--grad-monitor-kv-layer", type=int, default=4,
                        help="1-based SiT block index for KV-loss gradient probe")
    parser.add_argument("--grad-monitor-repa-layer", type=int, default=10,
                        help="1-based SiT block index for REPA-loss gradient probe")

    # whether to normalize spatial features
    parser.add_argument("--spnorm-method", type=str, default="zscore", choices=["none", "zscore", "zscore_token", "layernorm"])
    parser.add_argument("--cls-token-weight", type=float, default=0.2)
    parser.add_argument("--zscore-alpha", type=float, default=0.6)
    parser.add_argument("--zscore-proj-skip-std", action=argparse.BooleanOptionalAction, default=False)

    # config file (YAML)
    parser.add_argument("--config", type=str, default=None,
        help="Path to YAML config file (e.g., configs/irepa.yaml)")

    # First parse to get config file path
    if input_args is not None:
        args, remaining = parser.parse_known_args(input_args)
    else:
        args, remaining = parser.parse_known_args()

    # Load config file if provided (CLI args will override config values)
    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        # Set config values as new defaults (CLI args will override these)
        for key, value in config.items():
            key_underscore = key.replace('-', '_')
            for action in parser._actions:
                if action.dest == key_underscore:
                    action.default = value
                    break

    # Re-parse with updated defaults so CLI args take precedence
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.kv_distill_snr_gamma <= 0:
        parser.error("--kv-distill-snr-gamma must be > 0")
    if not 0.0 <= args.kv_distill_min_weight <= 1.0:
        parser.error("--kv-distill-min-weight must be in [0, 1]")
    if args.kv_stop_fade_steps < 0:
        parser.error("--kv-stop-fade-steps must be >= 0")
    if args.repa_stop_fade_steps < 0:
        parser.error("--repa-stop-fade-steps must be >= 0")
    if args.grad_monitor_interval <= 0:
        parser.error("--grad-monitor-interval must be > 0")
    if args.grad_monitor_window_steps <= 0:
        parser.error("--grad-monitor-window-steps must be > 0")
    if args.grad_monitor_kv_layer <= 0:
        parser.error("--grad-monitor-kv-layer must be >= 1")
    if args.grad_monitor_repa_layer <= 0:
        parser.error("--grad-monitor-repa-layer must be >= 1")

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
