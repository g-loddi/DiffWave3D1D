import torch.nn as nn
import torch
import math
from diffwave import DiffWave
from tqdm import tqdm
import random
import numpy as np
import copy
import torch.distributed as dist
import os
from torch.utils.data import DataLoader
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq




class Diffusion(nn.Module):
    def __init__(self, model_name, timesteps, image_size, in_channels, sequence_length, residual_channels, residual_layers, dilation_cycle_length, device='cuda'):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.model_name = model_name
        betas = self._cosine_variance_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

        # Initialize the unified DiffWave model with appropriate block type
        self.model = DiffWave(
            input_channels=in_channels,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
            noise_schedule=np.linspace(1e-4, 0.05, timesteps).tolist(),
            block_type=model_name,  # Use model_name as block_type
            device=device
        )

    def forward(self,x,noise,target=None):
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x = x_t, diffusion_step = t, condition = target)
        return pred_noise
    
    @torch.no_grad()
    def evaluate(self,x,noise,t,target=None):
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x = x_t, diffusion_step = t, condition = target)
        return pred_noise

    @torch.no_grad()
    def sampling(self,n_samples,dataset, n_real_samples, clipped_reverse_diffusion=True,device="cuda"):
        anchor_index = dataset.anchor_index
        service_index = dataset.service_index
        DoW_index = dataset.DoW_index
        metadata, x = copy.deepcopy(random.choice(dataset))         #create copy to avoid in-place modification of dataset when unsqueezing
        metadata['position'] = torch.tensor(metadata['position']).unsqueeze(0).to(device)
        metadata['DoW'] = torch.tensor(metadata['DoW']).unsqueeze(0).to(device)
        metadata['service'] = torch.tensor(metadata['service']).unsqueeze(0).to(device)

        # Get indices of samples with same anchor (position)
        idx_same_anchor = anchor_index[tuple(list(metadata['anchor']))]
        
        # Get indices of samples with same service
        service_key = metadata['service'].item()  # Convert tensor to scalar
        # Convert numeric service index to string
        service_mapping = {v: k for k, v in dataset.SERVICE_MAPPING.items()}
        service_str = service_mapping.get(service_key, "Unknown")
        idx_same_service = service_index[service_str]
        
        # Get indices of samples with same day of week
        dow_key = metadata['DoW'].item()  # Convert tensor to scalar
        idx_same_dow = DoW_index[dow_key]
        
        # Find intersection of all three sets of indices
        matching_indices = set(idx_same_anchor).intersection(set(idx_same_service)).intersection(set(idx_same_dow))
        
        # Convert back to list and get the real samples
        real_alike_x = [dataset[idx][1] for idx in list(matching_indices)[:n_real_samples]]
        
        # All models now use 3D tensors
        x_t=torch.randn((n_samples,1,self.sequence_length,self.image_size,self.image_size)).to(device)

        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise,metadata)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise,metadata)

        # Prepare metadata for return
        metadata_dict = {
            'position': metadata['position'].squeeze().cpu().numpy()[:, 0, 0],  # Get first column of first row from each matrix
            'DoW': metadata['DoW'].squeeze().cpu().numpy(),
            'service': service_str  # Use the string service name instead of numeric index
        }

        return x_t.squeeze(0), real_alike_x, metadata_dict
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1}) 
        buf_shape = (x_0.shape[0], 1, 1, 1, 1)
        x_t = self.sqrt_alphas_cumprod.gather(-1,t).reshape(buf_shape)*x_0+ \
            self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(buf_shape)*noise
        return x_t

    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise,target=None):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred = self.model(x = x_t, diffusion_step = t, condition = target)

        buf_shape = (x_t.shape[0], 1, 1, 1, 1)
        alpha_t=self.alphas.gather(-1,t).reshape(buf_shape)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(buf_shape)
        beta_t=self.betas.gather(-1,t).reshape(buf_shape)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(buf_shape)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(buf_shape)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self,x_t,t,noise,target): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred = self.model(x = x_t, diffusion_step = t, condition = target)

        buf_shape = (x_t.shape[0], 1, 1, 1, 1)
        alpha_t=self.alphas.gather(-1,t).reshape(buf_shape)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(buf_shape)
        beta_t=self.betas.gather(-1,t).reshape(buf_shape)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt((1-alpha_t_cumprod)/alpha_t_cumprod)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(buf_shape)
            mean = (torch.sqrt(alpha_t_cumprod_prev) * beta_t * x_0_pred + torch.sqrt(alpha_t) * (1. - alpha_t_cumprod_prev) * x_t) / (1. - alpha_t_cumprod)
            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            mean = x_0_pred
            std = 0.0

        return mean + std * noise
    
    @torch.no_grad()
    def debug_plot(self,x,metadata,t,device):
        t = torch.tensor([t]).to(device)
        noise=torch.randn_like(x).to(device)

        x_t=self._forward_diffusion(x,t,noise)
        pred = self.model(x = x_t, diffusion_step = t, condition = metadata)
            
        buf_shape = (x_t.shape[0], 1, 1, 1, 1)
        alpha_t=self.alphas.gather(-1,t).reshape(buf_shape)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(buf_shape)
        beta_t=self.betas.gather(-1,t).reshape(buf_shape)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt((1-alpha_t_cumprod)/alpha_t_cumprod)*pred
        return x.squeeze(), x_t.squeeze(), x_0_pred.squeeze()

    @torch.no_grad()
    def generate(self, model_name, dataset, batch_size, seed, cpu=False):
        torch.manual_seed(seed)
        np.random.seed(seed)  # If you use numpy at any point

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = torch.device("cpu") if cpu else torch.device("cuda")
        if dist.is_initialized():
            print(f"Process {dist.get_rank()} is using GPU {torch.cuda.current_device()}")
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            print(f"Non-distributed run using device {torch.cuda.current_device()}")
            local_rank = 0
            world_size = 1

        # Only rank 0 (main process) prints progress bar
        is_main_process = (local_rank == 0)

        # Create a sampler for distributed runs.
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=sampler
        )

        filename = f'generated_data_{model_name}_seed{seed}_rank_{local_rank}.parquet'
        writer = None  # ParquetWriter instance

        for metadata, x in tqdm(dl, desc="Generating samples", disable=not is_main_process):
            # Move conditioning variables to the proper device.
            metadata['position'] = torch.as_tensor(metadata['position'],device=device)
            metadata['DoW'] = torch.as_tensor(metadata['DoW'],device=device)
            metadata['service'] = torch.as_tensor(metadata['service'],device=device)
         
            metadata['position'] = metadata['position'].to(device)
            metadata['DoW'] = metadata['DoW'].to(device)
            metadata['service'] = torch.tensor(metadata['service']).to(device)

            # Add a channel dimension.
            x = x.unsqueeze(1)
            # Create a fixed initial noise sample for the batch.
            x_init = torch.randn_like(x).to(device)
            x_t = x_init.clone()

            for t in range(self.timesteps - 1, -1, -1):
                noise = torch.randn_like(x_t).to(device)
                t_tensor = torch.tensor([t for _ in range(x.shape[0])]).to(device)
                x_t = self._reverse_diffusion_with_clip(x_t, t_tensor, noise, metadata)

            # Remove the channel dimension.
            x = x.squeeze(1)
            x_t = x_t.squeeze(1)

            # If the data is in patch form (4D tensor), flatten to individual cells.
            if len(x_t.shape) == 4:
                this_batch_size = x_t.shape[0]
                cells_in_patch = x_t.shape[2] * x_t.shape[3]
                x_t = x_t.permute(0, 2, 3, 1).reshape(this_batch_size * cells_in_patch, 96)
                metadata['position'] = metadata['position'].permute(0, 2, 3, 1).reshape(this_batch_size * cells_in_patch, 2)
                metadata['DoW'] = metadata['DoW'].repeat_interleave(cells_in_patch)
                metadata['city'] = [item for item in metadata['city'] for _ in range(cells_in_patch)]
                metadata['date'] = [item for item in metadata['date'] for _ in range(cells_in_patch)]
                metadata['service'] = [item for item in metadata['service'] for _ in range(cells_in_patch)]

            # Build rows for the current batch.
            rows = []
            # Loop over each sample in the batch.
            for j in range(x_t.shape[0]):
                city = str(metadata['city'][j])      # Ensure city is a string.
                service = str(metadata['service'][j])  # Ensure service is a string.
                date = str(metadata['date'][j])        # Ensure date is a string.

                # Convert the position to a tuple and then split it.
                position = metadata['position'][j]
                pos_tuple = tuple(map(int, position.tolist()))
                if len(pos_tuple) < 2:
                    raise ValueError("Position tuple must contain at least 2 elements.")
                pos_0, pos_1 = pos_tuple[0], pos_tuple[1]

                # Get the generated data as a NumPy array of length 96.
                gen_data = x_t[j].cpu().numpy()
                row = {
                    'city': city,
                    'service': service,
                    'date': date,
                    'position_0': pos_0,
                    'position_1': pos_1,
                }
                # Add each generated value as a separate column.
                for i in range(96):
                    row[str(i)] = gen_data[i]
                rows.append(row)

            # Convert the batch rows to a DataFrame.
            df_batch = pd.DataFrame(rows)

            # Convert DataFrame to a PyArrow table and write.
            table = pa.Table.from_pandas(df_batch)
            if writer is None:
                writer = pq.ParquetWriter(filename, table.schema)
            writer.write_table(table)
            # Optionally, clear variables that are no longer needed
            del df_batch, table, rows

        if writer is not None:
            writer.close()

        return None
    
    