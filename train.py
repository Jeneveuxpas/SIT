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

from dataset import CustomDataset
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
    
    # Create Encoder K/V extractor first to detect dimension
    encoder_kv_extractor = None
    if len(encoders) > 0:
        encoder_kv_extractor = EncoderKVExtractor(encoders[0].model, enc_layer_indices)
        encoder_kv_extractor.eval()
        
        # Auto-detect real dimension of the extracted layer
        # This is critical for hierarchical models (SAM2/Hiera) where dims change across stages
        if len(enc_layer_indices) > 0:
            real_dim = encoder_kv_extractor.get_layer_dim(enc_layer_indices[0])
            if real_dim > 0 and args.enc_dim != real_dim:
                if accelerator.is_main_process:
                    logger.info(f"Overwriting args.enc_dim {args.enc_dim} -> {real_dim} based on extracted layer {enc_layer_indices[0]}.")
                args.enc_dim = real_dim
            elif real_dim == 0:
                 # Fallback to model embedding dim
                 z_dims = [encoder.embed_dim for encoder in encoders]
                 if len(z_dims) > 0 and args.enc_dim != z_dims[0]:
                    if accelerator.is_main_process:
                        logger.info(f"Overwriting args.enc_dim {args.enc_dim} -> {z_dims[0]} based on encoder.embed_dim.")
                    args.enc_dim = z_dims[0]
                    
    # z_dims should reflect the TARGET REPA dimension (typically final output)
    # args.enc_dim should reflect the K/V SOURCE dimension (intermediate layer)
    # So we do NOT overwrite z_dims with args.enc_dim here.
    # z_dims = [encoder.embed_dim for encoder in encoders] is already correct (set at line 172).
    # We ensure z_dims comes from the encoder object, not the potentially modified args.enc_dim.
    z_dims = [encoder.embed_dim for encoder in encoders]
    
    block_kwargs = {
        "fused_attn": args.fused_attn, 
        "qk_norm": args.qk_norm,
    }
    
    # Create SiT with Encoder KV
    model = SiT_EncoderKV_models[args.model](
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
        enc_dim=args.enc_dim,
        enc_heads=args.enc_heads,
        kv_proj_type=args.kv_proj_type,
        kv_proj_hidden_dim=args.kv_proj_hidden_dim,
        kv_proj_kernel_size=args.kv_proj_kernel_size,
        kv_norm_type=args.kv_norm_type,
        kv_zscore_alpha=args.kv_zscore_alpha,
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
        distill_coeff=args.distill_coeff,
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
    
    # Setup data (using sota CustomDataset)
    train_dataset = CustomDataset(args.data_dir)
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
    raw_image_sample, gt_xs, gt_labels = batch
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
            raw_image, x, y = batch
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)

            # Auto-stage switching
            progress = global_step / max(args.max_train_steps, 1)
            current_stage = 1 if progress < args.stage1_ratio else 2

            # Log stage periodically
            if accelerator.is_main_process and global_step % 1000 == 0:
                logger.info(f"Step {global_step}: stage = {current_stage} (progress = {progress:.2%})")

            labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                
                # Extract Encoder K/V and CLS token
                # Use encoder's built-in preprocess
                raw_image_enc = encoders[0].preprocess(raw_image)
                with accelerator.autocast():
                    enc_kv_list, enc_cls = encoder_kv_extractor(raw_image_enc)
                
                # Extract encoder features for REPA projection loss
                zs = []
                if args.repa_loss:
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
                    enc_kv_list=enc_kv_list,
                    stage=current_stage,
                    align_mode=args.align_mode,
                    kv_mode=args.kv_mode,
                )
                denoising_loss, proj_loss, distill_loss, loss_dict = loss_fn(model, x, model_kwargs, zs=zs)
                denoising_loss_mean = denoising_loss.mean()
                
                # Total loss
                loss = denoising_loss_mean + proj_loss + distill_loss
                # Ensure loss is float32 for scaler
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
                logging.info(f"Generated samples: Stage {current_stage}")

            # Include the loss and grad norms in the logging
            if accelerator.sync_gradients:
                logs = {
                    "loss": loss.detach().item(),
                    "denoising_loss": denoising_loss_mean.detach().item(),
                    "proj_loss": proj_loss.detach().item() if isinstance(proj_loss, torch.Tensor) else proj_loss,
                    "distill_loss": distill_loss.detach().item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
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
    parser.add_argument("--stage1-ratio", type=float, default=0.3,
                        help="Ratio of training for Stage 1. e.g., 0.5 = first 50%")
    parser.add_argument("--distill-coeff", type=float, default=2.0,
                        help="Coefficient for attention distillation loss (Stage 2 only)")
    parser.add_argument("--align-mode", type=str, default="attn_mse",
                        choices=["logits", "attn_mse", "kv_mse"],
                        help="Alignment mode: logits, attn_mse,kv_mse")
    parser.add_argument("--kv-proj-type", type=str, default="linear",
                        choices=["linear", "mlp", "conv"],
                        help="Projection type for Encoder K/V: linear, mlp, or conv")
    parser.add_argument("--kv-proj-hidden-dim", type=int, default=None,
                        help="Hidden dimension for MLP projection (default: max(enc_dim, sit_dim))")
    parser.add_argument("--kv-proj-kernel-size", type=int, default=1,
                        help="Kernel size for conv projection (default: 1)")
    parser.add_argument("--kv-norm-type", type=str, default="layernorm",
                        choices=["none", "layernorm", "zscore", "zscore_spatial", "zscore_token", "batchnorm"],
                        help="Normalization type for K/V: zscore=per-token, zscore_spatial=per-feature")
    parser.add_argument("--kv-zscore-alpha", type=float, default=1.0, 
                        help="Alpha for z-score normalization: (x - alpha * mean) / std")
    parser.add_argument("--enc-dim", type=int, default=768,
                        help="Encoder model embedding dimension")
    parser.add_argument("--enc-heads", type=int, default=12,
                        help="Encoder model number of attention heads")
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
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=str, default="1.0")
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--repa-loss", action=argparse.BooleanOptionalAction, default=True)
    # add loss type
    parser.add_argument("--projection-loss-type", type=str, default="cosine", help="Should be a comma-separated list of projection loss types")

    # whether to normalize spatial features
    parser.add_argument("--spnorm-method", type=str, default="zscore", choices=["none", "zscore", "zscore_spatial", "zscore_token", "layernorm"])
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

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
