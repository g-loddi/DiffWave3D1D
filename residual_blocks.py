import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

def Conv3d(*args, **kwargs):
    layer = nn.Conv3d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d(*args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

class BidirectionalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        self.conv_forward = Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_reverse = Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x):
        forward_out = self.conv_forward(x)
        reversed_x = x.flip(dims=[2])
        reverse_out = self.conv_reverse(reversed_x)
        reverse_out = reverse_out.flip(dims=[2])
        return forward_out + reverse_out

class DiffWave3DResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, diffusion_dim, cond_dims):
        super().__init__()
        self.dilated_conv = BidirectionalConv3d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(diffusion_dim, residual_channels)        
        self.conditions_projection = Conv3d(cond_dims, 2 * residual_channels, 1)
        self.output_projection = Conv3d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, embedded_conditions=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_step    
        conditions = self.conditions_projection(embedded_conditions)
        y = self.dilated_conv(y) + conditions

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class DiffWave1DResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, diffusion_dim, cond_dims):
        super().__init__()
        self.dilated_conv = BidirectionalConv3d(residual_channels, 2 * residual_channels, (3,1,1), padding=(dilation[0],0,0), dilation=dilation)
        self.diffusion_projection = nn.Linear(diffusion_dim, residual_channels)        
        self.conditions_projection = Conv3d(cond_dims, 2 * residual_channels, 1)
        self.output_projection = Conv3d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, embedded_conditions=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_step        
        conditions = self.conditions_projection(embedded_conditions)
        y = self.dilated_conv(y) + conditions

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class DiffWave3D1DResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, diffusion_dim, cond_dims):
        super().__init__()
        self.dilated_conv_3d = BidirectionalConv3d(residual_channels // 2, residual_channels, 3, padding=dilation, dilation=dilation)
        self.dilated_conv_1d = BidirectionalConv3d(residual_channels // 2, residual_channels, (3,1,1), padding=(dilation[0],0,0), dilation=dilation)
        self.diffusion_projection = nn.Linear(diffusion_dim, residual_channels)
        self.conditions_projection = Conv3d(cond_dims, 2 * residual_channels, 1)
        self.output_projection = Conv3d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, embedded_conditions=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y1, y2 = torch.chunk(x, 2, dim=1)
        diffusion_step1, diffusion_step2 = torch.chunk(diffusion_step, 2, dim=1)
        y1 = y1 + diffusion_step1
        y2 = y2 + diffusion_step2
        conditions = self.conditions_projection(embedded_conditions)
        conditions1, conditions2 = torch.chunk(conditions, 2, dim=1)
        y1 = self.dilated_conv_3d(y1) + conditions1
        y2 = self.dilated_conv_1d(y2) + conditions2

        gate1, filter1 = torch.chunk(y1, 2, dim=1)
        y1 = torch.sigmoid(gate1) * torch.tanh(filter1)

        gate2, filter2 = torch.chunk(y2, 2, dim=1)
        y2 = torch.sigmoid(gate2) * torch.tanh(filter2)

        y = torch.cat([y1, y2], dim=1)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class CSDIResidualBlock(nn.Module):
    def __init__(self, cond_dims, residual_channels, nheads, diffusion_dim):
        super().__init__()
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=residual_channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=residual_channels)

        self.mid_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
        self.diffusion_projection = nn.Linear(diffusion_dim, residual_channels)    
        self.conditions_projection = Conv1d(cond_dims, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward_time(self, y, base_shape):
        B, residual_channels, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, residual_channels, K, L).permute(0, 2, 1, 3).reshape(B * K, residual_channels, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, residual_channels, L).permute(0, 2, 1, 3).reshape(B, residual_channels, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, residual_channels, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, residual_channels, K, L).permute(0, 3, 1, 2).reshape(B * L, residual_channels, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, residual_channels, K).permute(0, 2, 3, 1).reshape(B, residual_channels, K * L)
        return y

    def forward(self, x, diffusion_step, embedded_conditions=None):
        B, residual_channels, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, residual_channels, K * L)
        
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step 
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _, _ = embedded_conditions.shape
        embedded_conditions = embedded_conditions.reshape(B, cond_dim, K*L)
        conditions = self.conditions_projection(embedded_conditions)
        y = y + conditions

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / sqrt(2.0), skip

def get_residual_block(block_type, **kwargs):
    """Factory function to get the appropriate residual block based on type."""
    if block_type == "diffwave3d":
        return DiffWave3DResidualBlock(
            residual_channels=kwargs['residual_channels'],
            dilation=kwargs['dilation'],
            diffusion_dim=kwargs.get('diffusion_dim', 512),
            cond_dims=kwargs.get('cond_dims', 512)
        )
    elif block_type == "diffwave3d1d":
        return DiffWave3D1DResidualBlock(
            residual_channels=kwargs['residual_channels'],
            dilation=kwargs['dilation'],
            diffusion_dim=kwargs.get('diffusion_dim', 512),
            cond_dims=kwargs.get('cond_dims', 512)
        )
    elif block_type == "diffwave3d1d_64":
        return DiffWave3D1DResidualBlock(
            residual_channels=kwargs['residual_channels'],
            dilation=kwargs['dilation'],
            diffusion_dim=kwargs.get('diffusion_dim', 512),
            cond_dims=kwargs.get('cond_dims', 512)
        )
    elif block_type == "diffwave1d":
        return DiffWave1DResidualBlock(
            residual_channels=kwargs['residual_channels'],
            dilation=kwargs['dilation'],
            diffusion_dim=kwargs.get('diffusion_dim', 512),
            cond_dims=kwargs.get('cond_dims', 512)
        )
    elif block_type == "csdi":
        return CSDIResidualBlock(
            cond_dims=kwargs['cond_dims'],
            residual_channels=kwargs['residual_channels'],
            nheads=kwargs['nheads'],
            diffusion_dim=kwargs.get('diffusion_dim', 128)  # Default to 128 for CSDI
        )
    else:
        raise ValueError(f"Unknown residual block type: {block_type}") 

def compute_receptive_field(residual_layers, dilation_cycle_length):
    """
    Compute the receptive field for a given number of layers and dilation cycle length.
    
    Args:
        residual_layers: Number of residual layers
        dilation_cycle_length: Length of the dilation cycle
        
    Returns:
        Dictionary containing receptive fields for each dimension (temporal, height, width)
    """
    # Initialize receptive field for each dimension
    rf_temporal = 1  # Temporal dimension
    rf_height = 1    # Height dimension
    rf_width = 1     # Width dimension
    
    # Compute receptive field for each layer
    for i in range(residual_layers):
        dilation = 2 ** (i % dilation_cycle_length)
        
        # For temporal dimension (affected by all convolutions)
        rf_temporal += 2 * (3 - 1) * dilation
        
        # For spatial dimensions (only affected by 3D convolutions)
        rf_height += 2 * (3 - 1)  # No dilation in spatial dimensions
        rf_width += 2 * (3 - 1)   # No dilation in spatial dimensions
    
    return {
        'temporal': rf_temporal,
        'height': rf_height,
        'width': rf_width
    }

def get_receptive_field(block_type, residual_layers, dilation_cycle_length):
    """
    Get the receptive field for a specific block type.
    
    Args:
        block_type: Type of residual block ("diffwave3d", "diffwave1d", "diffwave3d1d")
        residual_layers: Number of residual layers
        dilation_cycle_length: Length of the dilation cycle
        
    Returns:
        Dictionary containing receptive fields for each dimension
    """
    rf = compute_receptive_field(residual_layers, dilation_cycle_length)
    
    if block_type == "diffwave3d":
        return rf
    elif block_type == "diffwave1d":
        return {
            'temporal': rf['temporal'],
            'height': 1,  # No spatial receptive field for 1D
            'width': 1
        }
    elif block_type == "diffwave3d1d":
        # For 3D1D, we have two parallel paths with the same receptive field
        return rf
    else:
        raise ValueError(f"Unknown block type: {block_type}")

# Example usage:
if __name__ == "__main__":
    # Compute receptive fields for each model configuration
    model_configs = {
        "diffwave3d": {"residual_layers": 6, "dilation_cycle_length": 6},
        "diffwave1d": {"residual_layers": 4, "dilation_cycle_length": 4},
        "diffwave3d1d": {"residual_layers": 6, "dilation_cycle_length": 6}
    }
    
    print("Receptive Fields:")
    print("-" * 50)
    for model_name, config in model_configs.items():
        rf = get_receptive_field(model_name, config["residual_layers"], config["dilation_cycle_length"])
        print(f"{model_name}:")
        print(f"  Temporal: {rf['temporal']}")
        print(f"  Height: {rf['height']}")
        print(f"  Width: {rf['width']}")
        print("-" * 50) 