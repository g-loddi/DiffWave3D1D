import torch
import torch.nn as nn
import torch.distributed as dist
from model import Diffusion
from utils import ExponentialMovingAverage
from utils import NetmobDatasetLMDBUnified
from utils import write_real_data
from config import MODEL_CONFIGS, DATASET_CONFIG, CONFIG
import os
import pickle
from torch.utils.data import Subset
from tqdm import tqdm
import argparse


torch.cuda.empty_cache()

def load_model(model_ema_path, model_name, device="cuda"):
    """Load trained model and EMA model with DistributedDataParallel support"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Get model configuration
    model_config = MODEL_CONFIGS[model_name]
    common_config = CONFIG['common_config']

    # If running distributed, get the local rank and set the device accordingly
    if dist.is_initialized():
        local_rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device)

    print(f"Loading model on {device}...")

    # Initialize model architecture
    model = Diffusion(
        model_name=model_name,
        timesteps=common_config['timesteps'],
        image_size=model_config['image_size'],
        in_channels=common_config['in_channels'],
        sequence_length=common_config['sequence_length'],
        residual_channels=model_config['residual_channels'],
        residual_layers=model_config['residual_layers'],
        dilation_cycle_length=model_config['dilation_cycle_length'] if model_name != "csdi" else None
    ).to(device)

    # Wrap model in DistributedDataParallel if in distributed mode
    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device.index], 
            output_device=device.index,
            find_unused_parameters=True
        )

    model.eval()

    # Load Exponential Moving Average (EMA) Model
    model_ema_steps = CONFIG['training_config']['model_ema_steps']
    model_ema_decay = CONFIG['training_config']['model_ema_decay']
    epochs = CONFIG['runtime_config']['epochs']

    adjust = 1 * model_config['batch_size'] * model_ema_steps / epochs
    alpha = 1.0 - model_ema_decay
    alpha = min(1.0, alpha * adjust)

    # Initialize EMA model (using the underlying model if wrapped in DDP)
    model_ema = ExponentialMovingAverage(
        model.module if hasattr(model, "module") else model, 
        device=device, 
        decay=1.0 - alpha
    )

    # Load EMA checkpoint on the correct device
    checkpoint_ema = torch.load(model_ema_path, map_location=device)
    model_ema.load_state_dict(checkpoint_ema)
    model_ema.to(device)
    model_ema.eval()

    print(f"Model EMA successfully loaded on {device}!")
    return model_ema, model_config['batch_size']

def main():

    parser = argparse.ArgumentParser(description="Generate samples with trained models.")
    parser.add_argument('--seed', type=int, default=42, help="Seed for loading test indices (default: 42)")
    args = parser.parse_args()

    seed = args.seed

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Distributed initialization is handled by torchrun
    if dist.is_available() and dist.is_initialized():
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")

    # List of models to generate
    model_names = ['diffwave1d','diffwave3d','diffwave3d1d_64','csdi']
    
    
    # Dataset setup - this is common for all models
    dataset = NetmobDatasetLMDBUnified(
        data_dir='data',
        scaling=DATASET_CONFIG['scaling'],
        log_transform=DATASET_CONFIG['log_transform'], 
        threshold=5,
        use_precomputed=DATASET_CONFIG['use_precomputed']
    )

    # Load test indices
    split_dir = os.path.join(dataset.db_dir, "splits")
    test_path = os.path.join(split_dir, f"test_indices_seed{seed}.pkl")
    with open(test_path, "rb") as f:
        test_indices = pickle.load(f)
    test_dataset = Subset(dataset, test_indices)

    # Write real data first (only once)
    if dist.get_rank() == 0:  # Only write real data once
        write_real_data(
            dataset=test_dataset,
            batch_size=4096,
            seed = seed
        )

    # Generate for each model
    for model_name in model_names:
        if dist.get_rank() == 0:
            print(f"\nGenerating samples for model: {model_name}")
        
        # Update dataset model_name for current model
        dataset.model_name = model_name
        
        # Model generation
        model_ema_path = f'./model_ema_{model_name}_seed{seed}.pth'
        model_ema, batch_size = load_model(model_ema_path, model_name, device)

        with torch.no_grad():
            generated = model_ema.module.generate(
                model_name, test_dataset, batch_size=batch_size, seed = seed
            ) if hasattr(model_ema, "module") else model_ema.generate(
                model_name, test_dataset, batch_size=batch_size, seed = seed
            )

        # Clean up model to free GPU memory
        del model_ema
        torch.cuda.empty_cache()

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
