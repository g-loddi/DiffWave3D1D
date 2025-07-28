# Standard library imports
import os
import pickle
import time
import warnings
import argparse
import random
import  numpy as np
import copy
# Third-party imports
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

# Local imports
from model import Diffusion
from utils import (
    ExponentialMovingAverage,
    NetmobDatasetLMDBUnified,
    plot_sample,
    plot_denoised_sample
)
from config import MODEL_CONFIGS, TRAINING_CONFIG, DATASET_CONFIG, CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Training Netmob Diffusion")
    parser.add_argument('--model', type=str, default='diffwave3d')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset splitting')

    args = parser.parse_args()

    # Validate model type
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Add all config values to args
    # Runtime config
    runtime_config = CONFIG['runtime_config']
    args.ckpt = runtime_config['ckpt']
    args.n_samples = runtime_config['n_samples']
    args.no_clip = runtime_config['no_clip']
    args.cpu = runtime_config['cpu']
    args.world_size = runtime_config['world_size'] if runtime_config['world_size'] > 0 else torch.cuda.device_count()
    args.n_workers = runtime_config['n_workers']
    args.epochs = runtime_config['epochs']

    # Dataset config
    args.scaling = DATASET_CONFIG['scaling']
    args.log_transform = DATASET_CONFIG['log_transform']

    # Training config
    args.lr = TRAINING_CONFIG['learning_rate']
    args.batch_size = MODEL_CONFIGS[args.model]['batch_size']
    args.timesteps = CONFIG['common_config']['timesteps']
    args.model_ema_steps = TRAINING_CONFIG['model_ema_steps']
    args.model_ema_decay = TRAINING_CONFIG['model_ema_decay']
    args.log_freq = TRAINING_CONFIG['log_freq']
    args.sampling_freq = TRAINING_CONFIG['sampling_freq']
    args.n_evaluations = TRAINING_CONFIG['n_evaluations']
    args.evaluation_subintervals = TRAINING_CONFIG['evaluation_subintervals']

    return args

def setup_distributed_training(args):
    """Setup distributed training environment"""
    if not args.cpu and args.world_size > 1:
        dist.init_process_group(backend='nccl')
        args.local_rank = int(os.environ["LOCAL_RANK"])  
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device("cpu") if args.cpu else torch.device("cuda")
    print(f"Using device: {device}")
    return device

def setup_wandb(args, metadata):
    """Setup Weights & Biases logging"""
    is_main = (not hasattr(dist, "get_rank")) or (not args.world_size > 1) or (dist.get_rank() == 0)
    
    if not is_main:
        return None, None, None, None

    run = wandb.init(
        project="netmob_diffusion", 
        config=vars(args),
        tags=[f"{k}={v}" for k, v in metadata.items()],
        mode='online'
    )
    
    # Create necessary directories
    os.makedirs(f'{run.dir}/samples_plots', exist_ok=True)
    os.makedirs(f'{run.dir}/denoising_plots', exist_ok=True)

    # Create artifacts
    samples_plots = wandb.Artifact('sampled_time_series', type='image')
    denoising_plots = wandb.Artifact('denoising_time_series', type='image')
    samples_plots.metadata = metadata
    denoising_plots.metadata = metadata

    print('TRAINING - ', vars(args))
    return run, samples_plots, denoising_plots, is_main

def setup_dataset(args):
    """Setup dataset and dataloader"""
    dataset = NetmobDatasetLMDBUnified(
        data_dir='data',
        scaling=args.scaling,
        log_transform=args.log_transform, 
        threshold=5,
        use_precomputed=DATASET_CONFIG['use_precomputed'], 
        model_name=args.model
    )
    
    # Load or create train/test splits
    seed = args.seed
    split_dir = os.path.join(dataset.db_dir, "splits")
    train_path = os.path.join(split_dir, f"train_indices_seed{seed}.pkl")
    test_path = os.path.join(split_dir, f"test_indices_seed{seed}.pkl")
    
    if not os.path.exists(train_path):
        os.makedirs(split_dir, exist_ok=True)
        random.seed(seed)
        anchors = list(dataset.anchor_index.keys())
        random.shuffle(anchors)
        split_idx = int(len(anchors) * 0.7)
        train_anchors = anchors[:split_idx]
        test_anchors = anchors[split_idx:]

        train_indices = [idx for anchor in train_anchors for idx in dataset.anchor_index[anchor]]
        test_indices = [idx for anchor in test_anchors for idx in dataset.anchor_index[anchor]]

        with open(train_path, "wb") as f:
            pickle.dump(train_indices, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_indices, f)
    else:
        with open(train_path, "rb") as f:
            train_indices = pickle.load(f)
        with open(test_path, "rb") as f:
            test_indices = pickle.load(f)

    train_dataset = Subset(dataset, train_indices)
    print("Total dataset size:", len(dataset))
    print("Training dataset size:", len(train_dataset))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.n_workers,
        sampler=train_sampler
    )

    return train_dataloader, dataset

def setup_model(args, device):
    """Setup model and optimizer"""
    model = Diffusion(
        model_name=args.model,
        timesteps=CONFIG['common_config']['timesteps'],
        image_size=MODEL_CONFIGS[args.model]['image_size'],
        in_channels=CONFIG['common_config']['in_channels'],
        sequence_length=CONFIG['common_config']['sequence_length'],
        residual_channels=MODEL_CONFIGS[args.model]['residual_channels'],
        residual_layers=MODEL_CONFIGS[args.model]['residual_layers'],
        dilation_cycle_length=MODEL_CONFIGS[args.model]['dilation_cycle_length'] if args.model != "csdi" else None
    ).to(device)

    if not args.cpu and args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank, 
            find_unused_parameters=False
        )

    base_model = model.module if hasattr(model, "module") else model

    # Setup EMA
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(base_model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
    loss_fn = nn.MSELoss(reduction='mean')

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    return model, model_ema, optimizer, scheduler, loss_fn

def main(args):
    seed = args.seed
    random.seed(seed)           # Python's random
    np.random.seed(seed)        # NumPy
    torch.manual_seed(seed)     # PyTorch CPU
    torch.cuda.manual_seed(seed)        # PyTorch GPU (single)
    torch.cuda.manual_seed_all(seed)    # PyTorch GPU (all)
    torch.backends.cudnn.deterministic = True   # For reproducibility

    warnings.filterwarnings("ignore", message="find_unused_parameters", category=UserWarning)
    torch.cuda.empty_cache()

    # Setup distributed training
    device = setup_distributed_training(args)

    # Setup metadata
    metadata = {
        "model": args.model,
        "timesteps": args.timesteps,
        "seed": args.seed
    }

    # Setup wandb
    run, samples_plots, denoising_plots, is_main = setup_wandb(args, metadata)

    # Setup dataset and dataloader
    train_dataloader, dataset = setup_dataset(args)

    # Setup model and optimizer
    model, model_ema, optimizer, scheduler, loss_fn = setup_model(args, device)

    global_steps = 0
    last_log_time = time.time()

    for epoch in range(args.epochs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        model.train()
        
        for j, (target, image) in enumerate(train_dataloader):
            # Move data to device
            image = image.to(device).unsqueeze(1)
            noise = torch.randn_like(image).to(device)
            
            # Move target tensors to device
            for key in ['position', 'DoW', 'service']:
                if target[key] is not None:
                    target[key] = target[key].to(device)

            # Forward pass
            pred = model(image, noise, target)
            loss = loss_fn(pred, noise)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model.module if hasattr(model, "module") else model)
            
            global_steps += 1

            # Logging
            if is_main and j % args.log_freq == 0:
                elapsed_time = time.time() - last_log_time
                print(
                    f"Epoch[{epoch + 1}/{args.epochs}], Step[{j}/{len(train_dataloader)}], "
                    f"loss:{loss.detach().cpu().item():.5f}, lr:{scheduler.get_last_lr()}, "
                    f"time:{elapsed_time:.2f}s"
                )
                last_log_time = time.time()

            running_loss += loss.detach().cpu().item()

        scheduler.step()
        epoch_loss = running_loss / len(train_dataloader)

        # Evaluation and sampling
        if is_main:
            # Log epoch loss
            log_dict = {"train_loss/avg_train_loss": epoch_loss, "epoch": epoch}
            
            # Evaluate subinterval losses
            if epoch % args.sampling_freq == 0 or epoch > args.epochs - 5:
                model.eval()
                subinterval_train_losses = []
                subintervals = torch.linspace(1, args.timesteps, steps=args.evaluation_subintervals + 1, device=device).long()
                
                for k in range(len(subintervals) - 1):
                    t_start, t_end = subintervals[k], subintervals[k + 1]
                    subinterval_losses = []

                    for _ in range(args.n_evaluations):
                        t_sample = torch.randint(t_start, t_end, (image.shape[0],), device=device)
                        # Handle both distributed and non-distributed cases
                        if hasattr(model, 'module'):
                            pred_noise = model.module.evaluate(image, noise, t_sample, target)
                        else:
                            pred_noise = model.evaluate(image, noise, t_sample, target)
                        sampled_loss = loss_fn(pred_noise, noise)
                        subinterval_losses.append(sampled_loss.item())

                    avg_loss = sum(subinterval_losses) / len(subinterval_losses)
                    subinterval_train_losses.append(avg_loss)
                    log_dict[f"train_loss/subinterval{k}"] = avg_loss

                wandb.log(log_dict)

                # Generate samples and plots
                model_ema.eval()
                synthetic_sample, real_alike_samples, metadata = model_ema.module.sampling(
                    args.n_samples,
                    dataset,
                    n_real_samples=100,
                    clipped_reverse_diffusion=True,
                    device=device
                )
                plot_sample(
                    synthetic_sample,
                    real_alike_samples,
                    metadata,
                    save_path=f'{run.dir}/samples_plots',
                    filename=f"sampling_epoch{epoch:04}.png"
                )

                # Generate denoising plots
                os.makedirs(f"{run.dir}/denoising_plots/epoch{epoch:04}", exist_ok=True)
                sample_metadata, sample_x = copy.deepcopy(random.choice(dataset))
                sample_x = sample_x.unsqueeze(0).unsqueeze(1).to(device)
                
                for key in ['position', 'DoW', 'service']:
                    if sample_metadata[key] is not None:
                        sample_metadata[key] = torch.tensor(sample_metadata[key]).unsqueeze(0).to(device)

                for t in range(1, args.timesteps, args.timesteps//10):
                    real_sample, noised_sample, denoised_sample = model_ema.module.debug_plot(
                        sample_x,
                        sample_metadata,
                        t,
                        device=device
                    )
                    plot_denoised_sample(
                        real_sample,
                        noised_sample,
                        denoised_sample,
                        save_path=f"{run.dir}/denoising_plots/epoch{epoch:04}",
                        filename=f"{t:04}.png"
                    )

    # Save checkpoints and artifacts
    if is_main:
        model_dir = os.path.join(run.dir, 'model')
        os.makedirs(model_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(model_dir, 'model_seed.pth'))
        torch.save(model_ema.state_dict(), os.path.join(model_dir, 'model_ema.pth'))
        
        # Save model_ema locally based on model_name
        model_name = model.module.model_name if hasattr(model, "module") else model.model_name
        torch.save(model_ema.state_dict(), f'model_ema_{model_name}_seed{args.seed}.pth')

        # Log artifacts
        artifact = wandb.Artifact('model', type='model')
        artifact.add_dir(model_dir)
        run.log_artifact(artifact)

        # Create new artifacts for the plots
        samples_plots = wandb.Artifact('sampled_time_series', type='image')
        denoising_plots = wandb.Artifact('denoising_time_series', type='image')
        
        # Add all plot files to the artifacts
        samples_plots.add_dir(f'{run.dir}/samples_plots', name='samples_plots')
        denoising_plots.add_dir(f'{run.dir}/denoising_plots', name='denoising_plots')
        
        # Log the artifacts
        run.log_artifact(samples_plots)
        run.log_artifact(denoising_plots)
            
        run.finish()
        
    # Clean up distributed process group
    if not args.cpu and args.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    main(args)