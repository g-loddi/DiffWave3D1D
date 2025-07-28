import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from torchsummary import summary

from residual_blocks import get_residual_block, Conv3d, Conv2d, Conv1d, silu

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 128)
        self.projection2 = nn.Linear(128, 128)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class PositionEmbedding(nn.Module):
    def __init__(self, max_height, max_width):
        super().__init__()
        self.register_buffer('embedding_i', self._build_embedding(max_height), persistent=False)
        self.register_buffer('embedding_j', self._build_embedding(max_width), persistent=False)

    def forward(self, i, j):
        emb_i = self._get_embedding(i, self.embedding_i)
        emb_j = self._get_embedding(j, self.embedding_j)
        emb_i = emb_i.permute(0, 3, 1, 2).contiguous()
        emb_j = emb_j.permute(0, 3, 1, 2).contiguous()
        x = torch.cat([emb_i, emb_j], dim=1)
        return x

    def _get_embedding(self, t, table):
        if t.dtype in [torch.int32, torch.int64]:
            emb = table[t]
        else:
            emb = self._lerp_embedding(t, table)
        return emb

    def _lerp_embedding(self, t, table):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low_idx = torch.clamp(low_idx, 0, table.shape[0] - 1)
        high_idx = torch.clamp(high_idx, 0, table.shape[0] - 1)
        low = table[low_idx]
        high = table[high_idx]
        weight = (t - low_idx).unsqueeze(-1)
        return low + (high - low) * weight

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        table = steps * 10.0**(dims * 4.0 / 63.0)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class TimeEmbedding(nn.Module):
    def __init__(self, d_model=128, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.device = device

    def forward(self, pos):
        pos = pos.to(self.device)
        pe = torch.zeros(pos.shape[0], pos.shape[1], self.d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, self.d_model, 2).to(self.device) / self.d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

# class MetadataConvolution(nn.Module):
#     def __init__(self, d_meta):
#         super().__init__()
#         self.conv1 = Conv2d(d_meta, 512, kernel_size=1)
#         self.conv2 = Conv2d(512, 512, kernel_size=1)

#     def forward(self, x):
#         x = silu(self.conv1(x))
#         x = silu(self.conv2(x))
#         return x

class DiffWave(nn.Module):
    def __init__(self,
                 input_channels,
                 residual_layers,
                 residual_channels,
                 dilation_cycle_length,
                 noise_schedule,
                 block_type,  # Options: "diffwave3d", "diffwave3d1d", "diffwave1d", "csdi"
                 device='cuda'):
        super().__init__()
        
        self.block_type = block_type
        # Use appropriate input projection based on block type
        if block_type == "csdi":
            self.input_projection = Conv2d(input_channels, residual_channels, 1)
        else:
            self.input_projection = Conv3d(input_channels, residual_channels, 1)
            
        self.diffusion_embedding = DiffusionEmbedding(len(noise_schedule))
        self.device = device

        self.position_embedding = PositionEmbedding(max_height=409, max_width=346)
        self.DoW_embedding = nn.Embedding(7, 16)
        self.service_embedding = nn.Embedding(3, 16)
        
        # Only initialize time embedding for CSDI blocks
        if block_type == "csdi":
            self.time_embedding = TimeEmbedding(device=device)
            total_cond_dims = 256 + 16 + 16 + 128  # position (256) + DoW (16) + service (16) + time (128)
        else:
            self.time_embedding = None
            total_cond_dims = 256 + 16 + 16  # position (256) + DoW (16) + service (16)

        # Create residual blocks based on the specified type
        self.residual_layers = nn.ModuleList([
            get_residual_block(
                block_type,
                residual_channels=residual_channels,
                dilation=(2**(i % dilation_cycle_length),1,1) if block_type in ["diffwave3d", "diffwave3d1d", "diffwave1d","diffwave3d1d_64"] else None,
                nheads=8 if block_type == "csdi" else None,
                cond_dims=total_cond_dims,  # Includes time embedding for CSDI blocks
                diffusion_dim=128  # From diffusion embedding output
            )
            for i in range(residual_layers)
        ])

        self.skip_projection = Conv3d(residual_channels, residual_channels, 1)
        self.output_projection = Conv3d(residual_channels, input_channels, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, diffusion_step, condition=None):
        B, C, T, H, W = x.shape
        
        # Apply input projection and reshape based on block type
        if self.block_type == "csdi":
            # For CSDI: reshape to [B, C, H*W, T]
            x = x.reshape(B, C, T, H * W).permute(0, 1, 3, 2)
            x = self.input_projection(x)  # Conv2d
        else:
            # For other blocks: keep shape [B, C, T, H, W]
            x = self.input_projection(x)  # Conv3d
            
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)

        # Prepare position embeddings
        position = None
        DoW = None
        service = None
        time_embed = None
        
        if condition['position'] is not None:
            position = condition['position']
            position = self.position_embedding(position[:, 0, :, :], position[:, 1, :, :])
        if condition['DoW'] is not None:
            DoW = self.DoW_embedding(condition['DoW']).unsqueeze(-1).unsqueeze(-1)
            DoW = DoW.expand(-1, -1, position.shape[2], position.shape[3])
        
        if condition['service'] is not None:
            service = self.service_embedding(condition['service']).unsqueeze(-1).unsqueeze(-1)
            service = service.expand(-1, -1, position.shape[2], position.shape[3])

        # Add time embedding for CSDI
        if self.time_embedding is not None:
            timestamps = torch.arange(T).unsqueeze(0).repeat(B, 1)
            time_embed = self.time_embedding(timestamps)
            time_embed = time_embed.permute(0, 2, 1).unsqueeze(3).unsqueeze(3)
            time_embed = time_embed.expand(-1, -1, -1, H, W)

        # Prepare metadata based on block type
        if self.block_type == "csdi":
            # For CSDI: reshape and concatenate all metadata
            total_pixels = H * W
            
            # Reshape position tensor
            position = position.reshape(B, position.shape[1], 1, H, W)  # Add time dimension
            position = position.expand(-1, -1, T, -1, -1)  # Expand along time
            position = position.reshape(B, position.shape[1], T, total_pixels).permute(0, 1, 3, 2)
            
            # Reshape DoW and service tensors
            DoW = DoW.reshape(B, DoW.shape[1], 1, H, W)
            DoW = DoW.expand(-1, -1, T, -1, -1)
            DoW = DoW.reshape(B, DoW.shape[1], T, total_pixels).permute(0, 1, 3, 2)
            
            service = service.reshape(B, service.shape[1], 1, H, W)
            service = service.expand(-1, -1, T, -1, -1)
            service = service.reshape(B, service.shape[1], T, total_pixels).permute(0, 1, 3, 2)
            
            # Reshape time embedding
            time_embed = time_embed.reshape(B, time_embed.shape[1], T, total_pixels).permute(0, 1, 3, 2)
            
            concat_metadata = torch.cat([position, time_embed, DoW, service], dim=1)
        else:
            # For other blocks: standard concatenation
            position = position.unsqueeze(2)
            DoW = DoW.unsqueeze(2)
            service = service.unsqueeze(2)
            concat_metadata = torch.cat([position, DoW, service], dim=1)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, concat_metadata)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        
        # Handle output projection based on block type
        if self.block_type == "csdi":
            # Reshape back to [B, C, T, H, W] for output projection
            x = x.permute(0, 1, 3, 2).reshape(B, -1, T, H, W)
            
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x

if __name__ == "__main__":
    from model import Diffusion
    from config import MODEL_CONFIGS, TRAINING_CONFIG, DATASET_CONFIG, CONFIG

    model_name_list = ["diffwave3d1d", "diffwave3d", "diffwave1d", "csdi", "diffwave3d1d_64"]
    for model_name in model_name_list:
        model = Diffusion(
            model_name=model_name,
            timesteps=CONFIG['common_config']['timesteps'],
            image_size=MODEL_CONFIGS[model_name]['image_size'],
            in_channels=CONFIG['common_config']['in_channels'],
            sequence_length=CONFIG['common_config']['sequence_length'],
            residual_channels=MODEL_CONFIGS[model_name]['residual_channels'],
            residual_layers=MODEL_CONFIGS[model_name]['residual_layers'],
            dilation_cycle_length=MODEL_CONFIGS[model_name]['dilation_cycle_length'] if model_name != "csdi" else None
        )
        print("-"*100)
        print(f"Model Summary for {model_name}:")
        print("-"*100)
        summary(model=model, input_size=(64, 1, 96, 16, 16), batch_size=32, device="cpu")
