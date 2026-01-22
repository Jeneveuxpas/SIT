# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples images from a trained SiT-DINOKV model using the original SiT architecture.
This script loads DINOKV checkpoints but uses sit.py for inference.
The DINO-KV specific weights are ignored via strict=False loading.
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import datetime
from models.autoencoder import VAE_F8D4
from samplers import euler_sampler, euler_maruyama_sampler


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """Builds a single .npz file from a folder of .png samples."""
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """Run sampling."""
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling requires GPU"
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model (using original SiT)
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    in_channels = 4

    z_dims_list = [int(elem) for elem in args.z_dims.split(',')] if args.z_dims else []
    
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=z_dims_list,
        encoder_depth=args.encoder_depth,
        eval_mode=True,
        **block_kwargs,
    ).to(device)
    
    # Load DINOKV checkpoint with strict=False (ignores DINO-KV specific weights)
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)['ema']
    
    # Filter out DINO-KV specific keys
    filtered_state_dict = {}
    ignored_keys = []
    for k, v in state_dict.items():
        if 'kv_proj' in k:  # DINO K/V projection weights
            ignored_keys.append(k)
        else:
            filtered_state_dict[k] = v
    
    if rank == 0 and ignored_keys:
        print(f"Ignored {len(ignored_keys)} DINO-KV specific keys")
    
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if rank == 0:
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    model.eval()

    # Load VAE
    vae = VAE_F8D4().to(device).eval()
    vae_ckpt = torch.load("pretrained_models/sdvae-ft-mse-f8d4.pt", map_location=f'cuda:{device}', weights_only=False)
    vae.load_state_dict(vae_ckpt)
    latents_stats = torch.load("pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt", map_location=f'cuda:{device}', weights_only=False)
    latents_scale = latents_stats["latents_scale"].to(device).view(1, -1, 1, 1)
    latents_bias = latents_stats["latents_bias"].to(device).view(1, -1, 1, 1)

    assert args.cfg_scale >= 1.0, "cfg_scale should be >= 1.0"

    # Create output folder
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    cfg_intv = "" if args.guidance_low == 0. and args.guidance_high == 1. else f"_{args.guidance_low}_{args.guidance_high}"
    hparams = f"cfg{args.cfg_scale}{cfg_intv}-seed{args.global_seed}-mode{args.mode}-steps{args.num_steps}_{ckpt_string_name}"
    exp_name = os.path.basename(os.path.normpath(args.ckpt.rsplit("checkpoints")[0]))
    sample_folder_dir = f"{args.sample_dir}/{exp_name}_{hparams}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Sampling loop
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total images to sample: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total_generated = 0

    for _ in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
        )
        
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            samples = vae.decode(samples / latents_scale + latents_bias).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, sample in enumerate(samples):
                index = (total_generated + i) * dist.get_world_size() + rank
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        
        total_generated += n

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, default="samples")
    
    # Model (using original SiT names)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--z-dims", type=str, default="")

    # Sampling
    parser.add_argument("--per-proc-batch-size", type=int, default=256)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--mode", type=str, default="sde")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    args = parser.parse_args()
    main(args)
