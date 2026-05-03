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

from dataset import HFImgLatentDataset, HFLatentDataset, ImageFolderLatentDataset, EDM2ImgLatentDataset
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

def parse_layer_indices(indices_str: str) -> list:
    """Parse comma-separated layer indices string to list of ints (1-based -> 0-based)."""
    return [int(x.strip()) - 1 for x in indices_str.split(',')]


def parse_layer_weights(weights_str: str, expected_len: int) -> list:
    """Parse and normalize comma-separated per-layer weights."""
    weights = [float(x.strip()) for x in weights_str.split(',') if x.strip()]
    if len(weights) != expected_len:
        raise ValueError(
            f"Expected {expected_len} layer weights, but got {len(weights)} from '{weights_str}'"
        )
    if any(weight < 0 for weight in weights):
        raise ValueError("Layer weights must be non-negative")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("At least one layer weight must be positive")
    return [weight / total_weight for weight in weights]


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


def get_three_stage_loss_multiplier(
    step: int,
    decay_start_step: int = None,
    decay_end_step: int = None,
    decay_end_scale: float = 1.0,
    final_scale: float = None,
) -> float:
    """
    Compute a 3-stage scalar multiplier:

    1) [0, decay_start_step): multiplier = 1.0
    2) [decay_start_step, decay_end_step): linearly decay 1.0 -> decay_end_scale
    3) [decay_end_step, end): multiplier = final_scale (or decay_end_scale if omitted)

    Passing None or negative steps disables this schedule and returns 1.0.
    """
    if (
        decay_start_step is None or decay_end_step is None
        or decay_start_step < 0 or decay_end_step < 0
    ):
        return 1.0
    if step < decay_start_step:
        return 1.0
    if step < decay_end_step:
        progress = (step - decay_start_step) / max(1, decay_end_step - decay_start_step)
        return 1.0 + progress * (decay_end_scale - 1.0)
    if final_scale is None:
        return decay_end_scale
    return final_scale


def get_stage1_transition_alpha(
    step: int,
    stage1_steps: int,
    transition_steps: int = 0,
    schedule: str = "cosine",
) -> float:
    """
    Smoothly hand off Stage-1 encoder-guided attention to native SiT attention.

    Returns alpha in [0, 1], where:
      0 -> pure encoder-guided attention
      1 -> pure SiT self-attention
    """
    if stage1_steps <= 0:
        return 1.0
    if transition_steps <= 0:
        return 0.0 if step < stage1_steps else 1.0
    if step >= stage1_steps:
        return 1.0

    transition_start = max(0, stage1_steps - transition_steps)
    if step < transition_start:
        return 0.0

    progress = (step - transition_start) / max(1, stage1_steps - transition_start)
    progress = min(1.0, max(0.0, progress))
    if schedule == "linear":
        return progress
    if schedule == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * progress))
    raise ValueError(f"Unknown transition schedule: {schedule}")


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
    sit_layer_loss_weights = None
    if args.sit_layer_loss_weights is not None:
        sit_layer_loss_weights = parse_layer_weights(
            args.sit_layer_loss_weights,
            expected_len=len(sit_layer_indices),
        )
    assert len(enc_layer_indices) == len(sit_layer_indices), \
        "Encoder and SiT layer indices must have same length"

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    in_channels = 4
    # Load encoders only when a raw-image branch is active.
    # KV distillation and REPA both require encoder features from raw images.
    needs_raw_images = args.use_kv or args.repa_loss
    if needs_raw_images:
        encoders = load_encoders(
            args.enc_type, device, args.resolution, accelerator=accelerator
        )
    else:
        encoders = []
    
    # Create Encoder K/V extractor first to detect dimension and heads (only if KV distillation enabled)
    encoder_kv_extractor = None
    enc_dim = None
    enc_heads = None
    if args.use_kv and len(encoders) > 0:
        encoder_kv_extractor = EncoderKVExtractor(encoders[0].model, enc_layer_indices)
        encoder_kv_extractor.eval()
        # Set target token count for spatial interpolation (SAM2 windowed attention etc.)
        encoder_kv_extractor._target_num_patches = (latent_size // 2) ** 2  # SiT patches: (32/2)^2 = 256
        
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
        sit_layer_loss_weights=sit_layer_loss_weights,
        enc_dim=enc_dim,
        enc_heads=enc_heads,
        kv_proj_type=args.kv_proj_type,
        kv_proj_hidden_dim=args.kv_proj_hidden_dim,
        kv_proj_kernel_size=args.kv_proj_kernel_size,
        kv_norm_type=args.kv_norm_type,
        kv_zscore_alpha=args.kv_zscore_alpha,
        kv_replace_mode=args.kv_replace_mode,
        kv_use_adaln=args.kv_use_adaln,
        train_kv_proj_in_stage2=args.train_kv_proj_stage2,
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
        logger.info(f"Encoder KV: {len(enc_layer_indices)} layer pairs, replace_mode={args.kv_replace_mode}")
        logger.info(f"Encoder layers: {enc_layer_indices} -> SiT layers: {sit_layer_indices}")
        if args.stage1_steps > 0 and args.transition_steps > 0:
            logger.info(
                "Stage-1 attention handoff: "
                f"last {args.transition_steps} steps with {args.transition_schedule} schedule"
            )
        if sit_layer_loss_weights is not None:
            logger.info(f"Stage-2 per-layer distill weights (normalized): {sit_layer_loss_weights}")
        if args.repa_decay_start_step is not None and args.repa_decay_start_step >= 0:
            logger.info(
                "REPA 3-stage schedule: "
                f"{args.repa_decay_start_step}->{args.repa_decay_end_step}, "
                f"end_scale={args.repa_decay_end_scale}, final_scale="
                f"{args.repa_final_scale if args.repa_final_scale is not None else args.repa_decay_end_scale}"
            )
        if args.train_kv_proj_stage2:
            logger.info("KV projection trainable in Stage 2 (no-Teacher-detach mode)")
        if args.kv_decay_start_step is not None and args.kv_decay_start_step >= 0:
            logger.info(
                "KV/distill 3-stage schedule: "
                f"{args.kv_decay_start_step}->{args.kv_decay_end_step}, "
                f"end_scale={args.kv_decay_end_scale}, final_scale="
                f"{args.kv_final_scale if args.kv_final_scale is not None else args.kv_decay_end_scale}"
            )
    
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
    
    # Setup data:
    # - If KV or REPA is active, we need raw images + latents.
    # - Otherwise, latent-only training can use the lighter HFLatentDataset path.
    if needs_raw_images:
        try:
            train_dataset = HFImgLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, split="train")
        except Exception as e:
            print(f"Error loading HFImgLatentDataset: {e}")
            try:
                print("Trying EDM2ImgLatentDataset (REPA preprocessing format)...")
                train_dataset = EDM2ImgLatentDataset(args.data_dir)
            except Exception as e2:
                print(f"Error loading EDM2ImgLatentDataset: {e2}")
                print("Falling back to ImageFolderLatentDataset")
                train_dataset = ImageFolderLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, resolution=args.resolution, split="train")
    else:
        print("Raw-image branches disabled; using HFLatentDataset")
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

    # Ablation: track whether stage-transition re-init has been done
    _stage_reinit_done = False

    for epoch in range(args.epochs):

        model.train()
        for batch in train_dataloader:
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
            transition_alpha = get_stage1_transition_alpha(
                step=global_step,
                stage1_steps=args.stage1_steps,
                transition_steps=args.transition_steps,
                schedule=args.transition_schedule,
            )

            # ── Ablation: re-initialize SiT body at Stage 1→2 transition ──
            if (current_stage == 2 and not _stage_reinit_done
                    and getattr(args, 'reinit_sit', False)):
                _stage_reinit_done = True
                if accelerator.is_main_process:
                    logger.info(f"[Ablation] Stage 1→2 transition at step {global_step}: "
                                "re-initializing SiT body (keeping KV projections)")
                # Unwrap model for direct parameter access
                raw_model = safe_unwrap_model(model)
                # Save kv_proj state dicts before re-init
                kv_proj_states = {}
                for i, block in enumerate(raw_model.blocks):
                    if hasattr(block, 'kv_proj'):
                        kv_proj_states[i] = {k: v.clone() for k, v in block.kv_proj.state_dict().items()}
                # Re-initialize the entire model
                raw_model.initialize_weights()
                # Restore kv_proj weights
                for i, block in enumerate(raw_model.blocks):
                    if i in kv_proj_states and hasattr(block, 'kv_proj'):
                        block.kv_proj.load_state_dict(kv_proj_states[i])
                # Reset EMA to match re-initialized model
                update_ema(ema, raw_model, decay=0)
                # Reset optimizer state (stale momentum from pre-reinit params)
                optimizer.zero_grad(set_to_none=True)
                for state in optimizer.state.values():
                    state.clear()
                if accelerator.is_main_process:
                    logger.info(f"[Ablation] SiT body re-initialized; preserved kv_proj in "
                                f"{len(kv_proj_states)} blocks; optimizer state & EMA reset")

            # Stage-wise termination gates
            # REPA: controls projection loss branch
            repa_stop_gate = get_loss_stop_multiplier(
                step=global_step,
                stop_step=args.repa_stop_step,
                fade_steps=args.repa_stop_fade_steps,
            )
            repa_decay_gate = get_three_stage_loss_multiplier(
                step=global_step,
                decay_start_step=args.repa_decay_start_step,
                decay_end_step=args.repa_decay_end_step,
                decay_end_scale=args.repa_decay_end_scale,
                final_scale=args.repa_final_scale,
            )
            repa_gate = repa_stop_gate * repa_decay_gate
            # KV gate only affects Stage 2 distillation; Stage 1 KV guidance stays enabled.
            kv_stop_gate = get_loss_stop_multiplier(
                step=global_step,
                stop_step=args.kv_stop_step,
                fade_steps=args.kv_stop_fade_steps,
            )
            kv_decay_gate = get_three_stage_loss_multiplier(
                step=global_step,
                decay_start_step=args.kv_decay_start_step,
                decay_end_step=args.kv_decay_end_step,
                decay_end_scale=args.kv_decay_end_scale,
                final_scale=args.kv_final_scale,
            )
            kv_gate = kv_stop_gate * kv_decay_gate
            repa_active = args.repa_loss and repa_gate > 0.0
            kv_active = args.use_kv and (current_stage == 1 or kv_gate > 0.0)

            # Log stage periodically
            if accelerator.is_main_process and global_step % 1000 == 0:
                logger.info(
                    f"Step {global_step}: stage = {current_stage} (switch at step {args.stage1_steps}), "
                    f"align_mode = {current_align_mode}, transition_alpha = {transition_alpha:.3f}, "
                    f"repa_gate = {repa_gate:.3f}, kv_gate = {kv_gate:.3f}"
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
                        raise ValueError("Active REPA/KV branch requires raw images, but the dataset returned latents only.")
                    raw_image_enc = encoders[0].preprocess(raw_image)
                    with accelerator.autocast():
                        encoder_kv_extractor.reset_cache()
                        encoder_kv_extractor._batch_size = raw_image_enc.shape[0]  # For SAM2 un-windowing
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
                             raise ValueError("use_kv requires raw images, but the dataset returned latents only.")
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
                    transition_alpha=transition_alpha if kv_active else 1.0,
                )
                
                denoising_loss, proj_loss_raw, distill_loss_raw, loss_dict = loss_fn(model, x, model_kwargs, zs=zs)
                denoising_loss_mean = denoising_loss.mean()
                proj_loss = proj_loss_raw * repa_gate
                
                # Distillation coefficient: 0 in Stage 1, fixed coefficient in Stage 2.
                base_distill_coeff = args.distill_coeff if current_stage == 2 else 0.0
                current_distill_coeff = base_distill_coeff * kv_gate
                distill_loss = distill_loss_raw * current_distill_coeff

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

                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        original_model = safe_unwrap_model(model)

                        checkpoint = {
                            "model": original_model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
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
                    # "proj_loss_raw": proj_loss_raw.detach().item() if isinstance(proj_loss_raw, torch.Tensor) else proj_loss_raw,
                    "distill_loss": distill_loss.detach().item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
                    # "distill_loss_raw": distill_loss_raw.detach().item() if isinstance(distill_loss_raw, torch.Tensor) else distill_loss_raw,
                    # "distill_coeff": current_distill_coeff,
                    # "distill_coeff_base": base_distill_coeff,
                    "repa_gate": repa_gate,
                    "repa_decay_gate": repa_decay_gate,
                    "kv_gate": kv_gate,
                    "kv_decay_gate": kv_decay_gate,
                    "transition_alpha": transition_alpha,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "stage": current_stage,
                }
                logs.update(loss_dict)
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
    parser.add_argument("--sit-layer-loss-weights", type=str, default=None,
                        help="Comma-separated Stage-2 distillation weights for sit-layer-indices; "
                             "weights are normalized to sum to 1")
    parser.add_argument("--stage1-steps", type=int, default=30000,
                        help="Number of steps for Stage 1 (e.g., 30000 for 100k total)")
    parser.add_argument("--transition-steps", type=int, default=5000,
                        help="Number of final Stage-1 steps used to cosine hand off "
                             "encoder-guided attention to native SiT attention. "
                             "Use 0 to keep a hard stage switch.")
    parser.add_argument("--transition-schedule", type=str, default="cosine",
                        choices=["cosine", "linear"],
                        help="Schedule for the Stage-1 attention handoff.")
    parser.add_argument("--distill-coeff", type=float, default=1.0,
                        help="Coefficient for distillation loss (0 in Stage 1, this value in Stage 2)")
    parser.add_argument("--align-mode", type=str, default="attn_mse",
                        choices=["logits", "attn_mse", "attn_cosine", "snr_attn_mse", "attn_kl", "kv_mse", "attn_hybrid"],
                        help="Alignment mode: logits, attn_mse, attn_cosine, snr_attn_mse, attn_kl, kv_mse, attn_hybrid")
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
    parser.add_argument("--kv-decay-start-step", type=int, default=None,
                        help="Start step for optional 3-stage Stage-2 KV/distill multiplier")
    parser.add_argument("--kv-decay-end-step", type=int, default=None,
                        help="End step for optional 3-stage Stage-2 KV/distill multiplier")
    parser.add_argument("--kv-decay-end-scale", type=float, default=1.0,
                        help="Multiplier reached at kv-decay-end-step during 3-stage KV/distill decay")
    parser.add_argument("--kv-final-scale", type=float, default=None,
                        help="Multiplier used after kv-decay-end-step (defaults to kv-decay-end-scale)")
    parser.add_argument("--kv-proj-type", type=str, default="linear",
                        choices=["linear", "mlp", "conv", "head_gate"],
                        help="Projection type for Encoder K/V: linear, mlp, conv, or head_gate")
    parser.add_argument("--kv-proj-hidden-dim", type=int, default=None,
                        help="Hidden dimension for MLP projection (default: max(enc_dim, sit_dim))")
    parser.add_argument("--kv-proj-kernel-size", type=int, default=1,
                        help="Kernel size for conv projection (default: 1)")
    parser.add_argument("--kv-norm-type", type=str, default="none",
                        choices=["none", "layernorm", "rmsnorm", "zscore", "zscore_token", "batchnorm", "k_rms_v_layer"],
                        help="Normalization type for K/V: zscore=per-spatial, zscore_token=per-token, k_rms_v_layer=K RMSNorm + V LayerNorm")
    parser.add_argument("--kv-zscore-alpha", type=float, default=1.0, 
                        help="Alpha for z-score normalization: (x - alpha * mean) / std")
    parser.add_argument("--kv-replace-mode", type=str, default="kv",
                        choices=["kv", "k", "v", "qkv", "qk", "q"],
                        help="Which attention components to replace from encoder in Stage 1: "
                             "kv (default), k-only, v-only, qkv (all), qk, q-only")
    parser.add_argument("--kv-use-adaln", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply AdaLN t-conditioning to KV projection output (default: False)")
    parser.add_argument("--train-kv-proj-stage2", action=argparse.BooleanOptionalAction, default=False,
                        help="Allow gradient flow through kv_proj during Stage 2 (use for no-Stage-1 ablation)")
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
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
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
    parser.add_argument("--enc-type", type=str, default='dinov2-b')
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
    parser.add_argument("--repa-decay-start-step", type=int, default=None,
                        help="Start step for optional 3-stage REPA multiplier")
    parser.add_argument("--repa-decay-end-step", type=int, default=None,
                        help="End step for optional 3-stage REPA multiplier")
    parser.add_argument("--repa-decay-end-scale", type=float, default=1.0,
                        help="Multiplier reached at repa-decay-end-step during 3-stage REPA decay")
    parser.add_argument("--repa-final-scale", type=float, default=None,
                        help="Multiplier used after repa-decay-end-step (defaults to repa-decay-end-scale)")
    # whether to normalize spatial features
    parser.add_argument("--spnorm-method", type=str, default="zscore", choices=["none", "zscore", "zscore_token", "layernorm"])
    parser.add_argument("--cls-token-weight", type=float, default=0.2)
    parser.add_argument("--zscore-alpha", type=float, default=0.6)
    parser.add_argument("--zscore-proj-skip-std", action=argparse.BooleanOptionalAction, default=False)

    # Ablation: selective re-initialization after checkpoint loading
    parser.add_argument("--reinit-sit", action=argparse.BooleanOptionalAction, default=False,
                        help="Re-initialize SiT body after loading checkpoint (keep KV projections). "
                             "Use with --resume-step to test Stage 1's contribution to SiT.")
    parser.add_argument("--reinit-kv-proj", action=argparse.BooleanOptionalAction, default=False,
                        help="Re-initialize KV projection after loading checkpoint (keep SiT body). "
                             "Use with --resume-step to test Stage 1's contribution to KV projection.")

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

    enc_layer_indices = parse_layer_indices(args.enc_layer_indices)
    sit_layer_indices = parse_layer_indices(args.sit_layer_indices)
    if len(enc_layer_indices) != len(sit_layer_indices):
        parser.error("Encoder and SiT layer indices must have same length")
    if args.sit_layer_loss_weights is not None:
        try:
            parse_layer_weights(args.sit_layer_loss_weights, expected_len=len(sit_layer_indices))
        except ValueError as exc:
            parser.error(str(exc))

    if args.kv_distill_snr_gamma <= 0:
        parser.error("--kv-distill-snr-gamma must be > 0")
    if not 0.0 <= args.kv_distill_min_weight <= 1.0:
        parser.error("--kv-distill-min-weight must be in [0, 1]")
    if args.kv_stop_fade_steps < 0:
        parser.error("--kv-stop-fade-steps must be >= 0")
    if args.repa_stop_fade_steps < 0:
        parser.error("--repa-stop-fade-steps must be >= 0")
    if args.transition_steps < 0:
        parser.error("--transition-steps must be >= 0")

    for prefix in ("kv", "repa"):
        decay_start = getattr(args, f"{prefix}_decay_start_step")
        decay_end = getattr(args, f"{prefix}_decay_end_step")
        decay_end_scale = getattr(args, f"{prefix}_decay_end_scale")
        final_scale = getattr(args, f"{prefix}_final_scale")

        decay_start_set = decay_start is not None and decay_start >= 0
        decay_end_set = decay_end is not None and decay_end >= 0
        if decay_start_set != decay_end_set:
            parser.error(
                f"--{prefix}-decay-start-step and --{prefix}-decay-end-step must both be set "
                "for the 3-stage schedule"
            )
        if decay_start_set and decay_end <= decay_start:
            parser.error(f"--{prefix}-decay-end-step must be greater than --{prefix}-decay-start-step")
        if decay_end_scale < 0:
            parser.error(f"--{prefix}-decay-end-scale must be >= 0")
        if final_scale is not None and final_scale < 0:
            parser.error(f"--{prefix}-final-scale must be >= 0")

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
