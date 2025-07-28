DiffWave3D1D  –  code overview
=================================
Official implementation of the paper: A Conditional Generative Diffusion Model for Spatio-Temporal Data

Accepted at ECAI 2025
Authors: Giulio, Alessandro Betti, Fabio Pinelli

File descriptions
--------------------

├── ./data   						# raw data folder
├── requirements.txt 				# to install dependencies
├── train_parallel_conditioned.py   # main training launcher 
├── generate.py                     # runs generation 
├── diffwave.py                     # unified denoiser backbone 
├── residual_blocks.py              # individual residual blocks built by diffwave.py
├── model.py                        # diffusion model
│
├── config.py                       # tiny helper that loads model_config.json 
├── model_config.json               # single source for all hyper-parameters
├── utils.py                        # shared utilities:
│     • NetMob LMDB dataset class
│     • ExponentialMovingAverage
│
├── compute_wessertain.py           # Wasserstein-distance evaluation  
├── compute_correlations.py         # hop-1 / hop-2 spatial correlations
└── TSTR_evaluation.py              # Train-on-Synthetic / Test-on-Real 

Quick start
-----------

1.  Install dependencies:
      $ pip install -r requirements.txt          # or install packages manually

2.  Train a model (example: hybrid 3D1D, 4 GPUs):
      $ torchrun --nproc_per_node=4 train_parallel_conditioned.py --model diffwave3d1d_64 --use_precomputed false 
	  You can set use_precomputed to true after the first run

3.  Generate synthetic samples:
      $ python generate.py 

4.  Evaluate/compute metrics:
      $ python compute_wessertain.py
      $ python compute_correlations.py
      $ python TSTR_evaluation.py

Attribution
-----------

This project includes code adapted from:

- [DiffWave](https://github.com/lmnt-com/diffwave) – Apache License 2.0
- [CSDI](https://github.com/ermongroup/CSDI) – MIT License
- [MNISTDiffusion](https://github.com/bot66/MNISTDiffusion) – MIT License

