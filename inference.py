#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Wan 2.2 Inference Script using the same architecture as training.
Based on train.py's WanPipeline component-based approach.
"""

import argparse
import logging
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

import random
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import toml

# Import the same components as training
from models.wan import wan
from models.wan import configs as wan_configs
# from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, MAX_AREA_CONFIGS
from models.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from models.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.common import is_main_process, DTYPE_MAP
from utils.dataset import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Wan 2.2 Inference using WanPipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to TOML config file (same format as training)')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for video generation')
    parser.add_argument('--output_path', type=str, default='output_video.mp4',
                        help='Output video file path')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA weights (optional)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed (-1 for random)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=5.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--video_length', type=int, default=81,
                        help='Number of frames (must be 4n+1)')
    parser.add_argument('--sample_solver', type=str, default='unipc',
                        choices=['unipc', 'dpm++'],
                        help='Sampling solver: unipc or dpm++')
    parser.add_argument('--shift', type=float, default=12,
                        help='Noise schedule shift parameter')
    parser.add_argument('--height', type=int, default=720,
                        help='Video height')
    parser.add_argument('--width', type=int, default=1280,
                        help='Video width')
    parser.add_argument('--model_type', type=str, default='t2v-A14B',
                        choices=['t2v-A14B', 'i2v-A14B', 'ti2v-5B', 's2v-14B'],
                        help='Model type')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to Wan model checkpoint directory')
    parser.add_argument('--transformer_path', type=str, default=None,
                        help='Path to transformer weights')
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA rank (should match training)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (should match training)')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='LoRA dropout (should match training)')
    return parser.parse_args()




def setup_model(config, lora_path=None, adapter_config=None, model_type='high_noise'):
    """Setup WanPipeline model using the same approach as training.
    
    Args:
        config: Model configuration
        lora_path: Optional path to LoRA weights
        adapter_config: Optional adapter configuration
        model_type: 'high_noise' or 'low_noise' to load the appropriate model
    """
    logger.info(f"Setting up WanPipeline model ({model_type})...")

    # Create a copy of config to avoid modifying the original
    import copy
    model_config = copy.deepcopy(config)
    
    # Override transformer_path based on model_type
    base_ckpt_path = model_config['model']['ckpt_path']
    if model_type == 'high_noise':
        model_config['model']['transformer_path'] = f"{base_ckpt_path}/high_noise_model"
        logger.info(f"Loading high noise model from {base_ckpt_path}/high_noise_model")
    elif model_type == 'low_noise':
        model_config['model']['transformer_path'] = f"{base_ckpt_path}/low_noise_model"
        logger.info(f"Loading low noise model from {base_ckpt_path}/low_noise_model")
    else:
        raise ValueError(f"model_type must be 'high_noise' or 'low_noise', got {model_type}")

    # Create model using the same logic as train.py line 318-319
    # WanPipeline auto-detects and loads the appropriate config from configs.py
    # based on the model dimensions and type, not from the TOML training config
    model = wan.WanPipeline(model_config)

    # Load diffusion model using the same logic as train.py line 424
    logger.info("Loading diffusion model...")
    model.load_diffusion_model()
    
    # Set model to eval mode for inference (no gradients, no dropout, etc.)
    logger.info("Setting model to eval mode for inference...")
    model.transformer.eval()
    model.text_encoder.model.eval()
    model.vae.model.eval()
    
    # Move model to GPU for inference
    logger.info("Moving model to GPU...")
    model.transformer.to('cuda')
    model.text_encoder.model.to('cuda')
    model.vae.model.to('cuda')
    
    # Disable gradients for inference
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.text_encoder.model.parameters():
        param.requires_grad = False
    for param in model.vae.model.parameters():
        param.requires_grad = False

    # Move text encoder to CUDA (same as original text2video.py line 268)
    logger.info("Moving text encoder to CUDA...")
    model.text_encoder.model.to('cuda')

    # Configure LoRA adapter if provided (same as train.py lines 426-430)
    if adapter_config:
        logger.info("Configuring LoRA adapter...")
        model.configure_adapter(adapter_config)

        # Load LoRA weights if provided
        if lora_path:
            logger.info(f"Loading LoRA weights from {lora_path}...")
            model.load_adapter_weights(lora_path)

            # Verify LoRA weights are loaded
            if hasattr(model, 'lora_model') and model.lora_model is not None:
                lora_params = sum(p.numel() for p in model.lora_model.parameters() if p.requires_grad)
                logger.info(f"✅ LoRA weights loaded: {lora_params:,} LoRA parameters")
            else:
                logger.warning("⚠️ LoRA weights may not have loaded properly!")

    logger.info("Model setup complete!")
    return model




def prepare_inputs(model,
                   prompt,
                   seed,
                   video_length,
                   offload_model=False,
                   n_prompt="",
                   size=(1280, 720),
                   vae_stride=[4, 8, 8],
                   patch_size = (1, 2, 2),
                   vae_z_dim=16,
                   sp_size = 32,
                   device='cuda'):
    """
    Prepare inputs for inference.
    Args:
        model:
        prompt:
        seed:
        video_length:
        size:
        vae_stride:
        patch_size:
        vae_z_dim:
        batch_size:

    Returns:

    """
    logger.info("[!Inference Pipeline] Preparing inputs...")

    ######################################################################
    # Seed
    ######################################################################
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    ######################################################################
    # Shape and length
    ######################################################################
    # Calculate target shape: this is the size of the random noise,
    # which should be the same as the size after VAE compression
    target_shape = (vae_z_dim, (video_length - 1) // vae_stride[0] + 1,
                    size[1] // vae_stride[1],
                    size[0] // vae_stride[2])

    # target_shape = (16, 21, 90, 160)

    print("target_shape:", target_shape)
    print("patch size:", patch_size)
    print("sp size:", sp_size)

    # Calculating sequence length referring Wan 2.2 inference
    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (patch_size[1] * patch_size[2]) *
                        target_shape[1] / sp_size) * sp_size

    logger.info(f"Latent shape: [{target_shape[0]}, {target_shape[1]}, {target_shape[2]}, {target_shape[3]}]")
    logger.info(f"Sequence length: {seq_len}")

    ######################################################################
    # Handle text embedding. To match the training pipeline, we just get
    # the id and mask for tokens here. Embedding part is done in the
    # InitialLayer in the model later.
    ######################################################################
    logger.info("Encoding text prompt...")

    # Move text encoder to device
    model.text_encoder.model.to(device)
    
    # Tokenize the prompt. This is the same thing as training, but we don't encode text here
    ids, mask = model.text_encoder.tokenizer([prompt], return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    text_mask = mask.to(device)
    
    # For null prompt (unconditional)
    if n_prompt == "":
        n_prompt = wan_configs.wan_shared_cfg.sample_neg_prompt
        print("Negative prompts:,", n_prompt)

    ids_null, mask_null = model.text_encoder.tokenizer([n_prompt], return_mask=True, add_special_tokens=True)
    ids_null = ids_null.to(device)
    text_mask_null = mask_null.to(device)
    
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    #     context_null = model.text_encoder.model(ids_null, mask_null)

    ######################################################################
    # Model offload and noise generation
    ######################################################################
    if offload_model:
        logger.info("[!!!] Offloading model to CPU...")
        model.text_encoder.model.cpu()

    # Generate random noise
    # noise = [torch.randn(target_shape[0],
    #                     target_shape[1],
    #                     target_shape[2],
    #                     target_shape[3],
    #                     dtype=torch.float32,
    #                     device=device,
    #                     generator=seed_g)]

    noise = [torch.load('/home/ubuntu/jin-Vol/code/Wan2.2/debug_noise.pt').to(device)]
    print("noise:", len(noise), noise[0].shape)
    print(noise[0].mean(), noise[0].std(), noise[0].var())

    # Create inputs dictionary that matches WanPipeline.prepare_inputs() expected format
    inputs = {
        'latents': noise,
        'caption': prompt,
        'ids': ids,
        'text_mask': text_mask,
        'ids_null': ids_null,
        'text_mask_null': text_mask_null,
        'seq_lens': seq_len,  # Calculated sequence length
        'mask':[],  # No mask for T2V
        'y': [],  # No image for T2V
        'clip_context': [],  # No CLIP for T2V
    }

    return inputs




def run_model_forward(model, inputs):
    """
    Args:
        model:
        model_inputs:
        timestep_quantile:

    Returns:
    """

    # Get the layers for component-based processing
    layers = model.to_layers()

    # Run through the component-based architecture:
    # InitialLayer -> Transformer Layers -> FinalLayer
    x = inputs  # Pass the entire inputs tuple to each layer

    # Process through each layer
    for layer in layers:
        x = layer(x)  # Each layer expects the full inputs tuple

    # Return the predicted noise
    return x




def setup_scheduler(sample_solver, num_inference_steps, shift, device='cuda'):
    """Setup flow matching scheduler (same as original Wan 2.2)."""
    logger.info(f"Setting up {sample_solver} scheduler...")

    # TODO (JH): timestep shift is None? It was 5 in Wan 2.2 repo.
    if sample_solver == 'unipc':
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,# Should match model's training timesteps
            use_dynamic_shifting=False
        )
        sample_scheduler.set_timesteps(
            num_inference_steps, device=device, shift=shift
        )
        timesteps = sample_scheduler.timesteps

    # TODO (JH): timestep shift cannot be None here. But we aren't using this solver for now
    elif sample_solver == 'dpm++':
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            sample_scheduler,
            device=device,
            sigmas=sampling_sigmas
        )
    else:
        raise NotImplementedError("Unsupported solver.")

    logger.info(f"Scheduler setup complete. Timesteps: {len(timesteps)}")
    return sample_scheduler, timesteps




def generate_video(model, prompt, video_length, seed,
                   num_inference_steps, guidance_scale, sample_solver='unipc', shift=12):
    """Generate video using WanPipeline's flow matching schedulers."""
    logger.info("Starting video generation...")

    # Prepare inputs using the correct format
    model_inputs = prepare_inputs(model,
                                   prompt,
                                   seed,
                                   video_length,
                                   size=(1280, 720),
                                   vae_stride=[4, 8, 8],
                                   patch_size = (1, 2, 2),
                                   vae_z_dim=16,
                                   sp_size = 1,
                                   device='cuda')

    logger.info(f"Model inputs prepared: {list(model_inputs.keys())}")


    # Setup flow matching scheduler (same as original Wan 2.2)
    logger.info("Setting up scheduler...")
    sample_scheduler, timesteps = setup_scheduler(sample_solver,
                                                  num_inference_steps,
                                                  shift,
                                                  device='cuda')

    print("Seq length: ", model_inputs['seq_lens'])
    print(num_inference_steps)
    print("shift: ", shift)
    print("timesteps: ", timesteps)
    # sys.exit(0)

    # Flow matching sampling loop (same as original Wan 2.2)
    logger.info(f"Starting flow matching denoising with {num_inference_steps} steps...")

    for i, t in enumerate(tqdm(timesteps)):
        """
        This is the inputs InitialLayer is expecting:
        x, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_fea = inputs
        """
        # Convert latents to list format (same as original)
        # latent_model_input = latents if isinstance(latents, list) else [latents]

        # Convert tensor to list format expected by scheduler (same as original Wan 2.2)
        timestep = [t]
        timestep = torch.stack(timestep)

        # TODO (JH): prepare 2 sets of inputs -- might need further investigation
        # Convert latents from list to tensor if needed
        latents = model_inputs["latents"]
        if isinstance(latents, list):
            latents = torch.stack(latents)
        
        inputs_cond = [
            latents.to(torch.bfloat16),  # Force convert to bfloat16
            None,  # y (image conditioning) - None for T2V
            timestep.to(torch.bfloat16),  # Convert timestep to bfloat16 too
            model_inputs["ids"].to(torch.long),
            model_inputs["text_mask"].to(torch.long),  # Use proper attention mask, not seq_lens
            None  # clip_context - None for T2V
        ]

        print("latents shape:", latents.shape)
        print("latents dtype:", latents.dtype)
        print("timestep:", timestep.to(torch.float32))
        print("ids:", model_inputs["ids"].to(torch.long).shape)
        print("seq_lens:", torch.tensor([model_inputs["seq_lens"]], dtype=torch.long).shape)

        # sys.exit(0)

        inputs_uncond = [
            latents.to(torch.bfloat16),  # Force convert to bfloat16
            None,  # y (image conditioning) - None for T2V
            timestep.to(torch.bfloat16),  # Convert timestep to bfloat16 too
            model_inputs["ids_null"].to(torch.long),
            model_inputs["text_mask"].to(torch.long),  # Use proper attention mask, not seq_lens
            None  # clip_context - None for T2V
        ]

        # Use WanPipeline's component-based architecture
        noise_pred_cond = run_model_forward(model, inputs_cond)
        noise_pred_uncond = run_model_forward(model, inputs_uncond)

        # Apply classifier-free guidance (same as original)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Scheduler step (same as original)
        temp_x0 = sample_scheduler.step(
            noise_pred.unsqueeze(0),
            t,
            latents[0].unsqueeze(0),
            return_dict=False,
            generator=None)[0]
        latents = [temp_x0.squeeze(0)]

    x0 = latents

    logger.info("Flow matching denoising complete!")

    # Decode latents to video using VAE (same as original)
    logger.info("Decoding latents to video...")
    with torch.no_grad():
        # VAE decode expects list format (same as original)
        if isinstance(x0, list):
            # Remove all singleton dimensions to get proper 5D tensor for VAE
            x0 = [x.squeeze(0) for x in x0]
            print(f"DEBUG: After squeeze, x[0] shape: {x0[0].shape}")
        else:
            x0 = x0.squeeze()
            print(f"DEBUG: After squeeze, x shape: {x0.shape}")
        videos = model.vae.decode(x0)

    return videos[0]


def save_video(video_tensor, output_path, fps=16):
    """Save video tensor to file."""
    logger.info(f"Saving video to {output_path}...")

    # Convert tensor to numpy and normalize
    video = video_tensor.squeeze(0).cpu().numpy()  # Remove batch dimension
    video = (video + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
    video = np.clip(video, 0, 1)

    # Convert to uint8
    video = (video * 255).astype(np.uint8)

    # Save using imageio or similar
    try:
        import imageio
        # video shape: (C, F, H, W) -> (F, H, W, C) for imageio
        video = np.transpose(video, (1, 2, 3, 0))
        imageio.mimsave(output_path, video, fps=fps)
        logger.info(f"Video saved successfully to {output_path}")
    except ImportError:
        logger.error("imageio not available. Please install with: pip install imageio")
        raise


def create_config_from_args(args):
    """Create model config from command line arguments."""

    # Check if required paths are provided
    if not args.ckpt_path:
        raise ValueError("--ckpt_path is required. Provide path to Wan model checkpoint directory.")
    if not args.transformer_path:
        raise ValueError("--transformer_path is required. Provide path to transformer weights.")

    config = {
        'model': {
            'type': 'wan',
            'ckpt_path': args.ckpt_path,
            'transformer_path': args.transformer_path,
            'dtype': torch.bfloat16,  # Convert to torch dtype
            'transformer_dtype': torch.float8_e4m3fn,  # Convert to torch dtype
            'min_t': 0,
            'max_t': 0.875,
            'cache_text_embeddings': True,
        }
    }

    # Add LoRA config if provided (same structure as training config)
    if args.lora_path:
        config['adapter'] = {
            'type': 'lora',
            'rank': args.lora_rank,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'dtype': torch.bfloat16  # Convert to torch dtype
        }

    return config



if __name__ == "__main__":
    ###################################################################
    # Configuration and logging
    ###################################################################
    args = parse_args()

    logger.info("Starting Wan 2.2 inference...")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Output: {args.output_path}")

    # Load config from file if provided, otherwise create from args
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading config from {args.config}")
        config = toml.load(args.config)
        
        # Convert string dtypes to torch dtypes (same as training)
        model_config = config['model']
        model_dtype_str = model_config['dtype']
        print(f"DEBUG: model_dtype_str = {model_dtype_str}, type = {type(model_dtype_str)}")
        
        # Only convert if it's a string
        if isinstance(model_dtype_str, str):
            model_config['dtype'] = DTYPE_MAP[model_dtype_str]
        else:
            print(f"DEBUG: dtype is already converted: {model_dtype_str}")
            
        if transformer_dtype := model_config.get('transformer_dtype', None):
            if isinstance(transformer_dtype, str):
                model_config['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)
        
        # Disable text embedding caching for inference (same as training)
        config['model']['cache_text_embeddings'] = False
        
        # Set autocast dtype to match model dtype (same as training)
        from utils import common
        common.AUTOCAST_DTYPE = config['model']['dtype']
        print(f"DEBUG: AUTOCAST_DTYPE = {common.AUTOCAST_DTYPE}")

    else:
        logger.info("Creating config from command line arguments")
        config = create_config_from_args(args)
        logger.warning( "Using default model paths. Please update the paths in create_config_from_args() "
                        "or provide a config file.")


    ###################################################################
    # Setup models using training architecture (high noise and low noise)
    ###################################################################
    adapter_config = config.get('adapter', None)
    
    # Load both high noise and low noise models for two-stage inference
    logger.info("Loading high noise model...")
    model_high_noise = setup_model(config, model_type='high_noise')
    
    logger.info("Loading low noise model...")
    model_low_noise = setup_model(config, model_type='low_noise')
    
    # For now, use high noise model as the default
    model = model_high_noise
    # print(model.prepare_inputs())

    ###################################################################
    # Generate video and save output
    ###################################################################
    # Generate video
    video = generate_video(
        model=model,
        prompt=args.prompt,
        video_length=args.video_length,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        sample_solver=args.sample_solver,
        shift=args.shift)

    # Save video
    save_video(video, args.output_path)

    logger.info("Inference complete!")

