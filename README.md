# Multiplex Virtual Staining

Paired image-to-image translation for virtual IHC staining: brightfield IHC + Hematoxylin to DAPI, Lap2, and Marker fluorescence channels.

## Task

Given a paired dataset of brightfield IHC images and their corresponding fluorescence channels (from the [DeepLIIF](https://github.com/nadeemlab/DeepLIIF) dataset), we train models to predict three target fluorescence channels:

- **DAPI** (nuclear stain)
- **Lap2** (cytoplasmic marker)
- **Marker** (protein of interest)

**Input**: IHC brightfield RGB (3ch) + Hematoxylin channel (1ch) = 4 channels
**Output**: 3 fluorescence channels (DAPI, Lap2, Marker)

All images are 512x512 and paired (registered). Data is normalized to [0, 1] for JiT and [-1, 1] for V3 GAN.

## Models

### 1. Pix2Pix V3 GAN (Baseline)

Conditional GAN with attention-gated U-Net generator and PatchGAN discriminator.

```
Architecture:
  Encoder:  ConvNeXt-Base (pretrained, frozen)
            Multi-scale features at 1/4, 1/8, 1/16, 1/32 resolution
  Decoder:  4 DecoderBlocks with attention-gated skip connections
            Each block: Upsample -> Conv -> BN -> ReLU + AttentionGate(skip)
  Output:   4 independent heads (Hematoxylin, DAPI, Lap2, Marker)
  Discriminator: PatchGAN70 (70x70 receptive field)

Training:
  Loss = lambda_gan * GAN_loss + lambda_l1 * L1_loss + lambda_perc * VGG_perceptual
  Optimizer: Adam (lr=2e-4, betas=[0.5, 0.999])
  Input: IHC RGB (3ch), normalized to [-1, 1]
  Output: 4 channels in [-1, 1]
```

Key files:
- `src/multiplex/models/v3_generator.py` - V3Generator
- `src/multiplex/models/encoder.py` - ConvNeXt encoder
- `src/multiplex/models/discriminator.py` - PatchGAN70
- `src/multiplex/training/trainer.py` - GAN training loop
- `src/multiplex/training/losses.py` - MultiLoss, DiscriminatorLoss

### 2. JiT Flow Matching

Diffusion transformer (DiT-style) with flow matching for conditional I2I translation. Based on [JiT (arXiv:2511.13720)](https://arxiv.org/abs/2511.13720).

```
Architecture:
  Source Encoder: Frozen ConvNeXt-Base -> multi-scale features -> cross-attention keys
  Patch Embed:    Image patches (32x32) -> BottleneckPatchEmbed -> hidden_size=768
  Timestep:       Sinusoidal + MLP -> AdaLN-Zero modulation
  Transformer:    12 ConditionalJiTBlocks
                  Each block: LayerNorm -> Self-Attention -> Cross-Attention -> SwiGLU FFN
                  Self-attn + Cross-attn use QK-Norm + Flash Attention 2
                  AdaLN-Zero: scale/shift/gate from timestep embedding
  MarkerGNN:      GATv2 graph over marker nodes for inter-channel reasoning (optional)
  Unpatchify:     Linear projection back to pixel space

Flow Matching:
  Forward:  x_t = (1-t) * noise + t * x_1    (linear interpolation)
  Training: predict x_1 directly (x_prediction) with L1 loss
  Sampling: Euler ODE solver, 50 steps (or Heun, 25 steps)
  Timestep distribution: logit-normal (mean=0, std=1)

Auxiliary Losses:
  BioLoss (lambda=0.05): Nuclear consistency + spatial coherence
  LPIPS (lambda=0.1): Perceptual loss at all timesteps
```

Key files:
- `src/multiplex/models/jit/conditional_jit.py` - Main JiT model
- `src/multiplex/models/jit/blocks.py` - Transformer blocks with AdaLN-Zero
- `src/multiplex/models/jit/attention.py` - Self + Cross attention with QK-Norm
- `src/multiplex/models/jit/source_encoder.py` - Frozen ConvNeXt source encoder
- `src/multiplex/models/jit/embeddings.py` - Patch embed, timestep embed, RMSNorm
- `src/multiplex/models/marker_gnn.py` - MarkerGNN (GATv2)
- `src/multiplex/training/flow_matching.py` - FlowMatchingLoss + ODE samplers
- `src/multiplex/training/jit_trainer.py` - Training loop
- `src/multiplex/training/bio_losses.py` - BioLoss suite
- `scripts/train_jit.py` - Training entry point

## Results (250K steps, 4x L40S)

| Model | PSNR | SSIM | LPIPS |
|-------|------|------|-------|
| V3 GAN (attention) | **26.74** | **0.7560** | **0.2841** |
| JiT FM (no LPIPS) | 20.35 | 0.5978 | 0.5429 |
| JiT FM (bridge+bio) | 20.34 | 0.5999 | 0.5383 |
| JiT FM (bridge+bio+p16) | 20.40 | 0.5805 | 0.5719 |

### Known Issues

The JiT flow matching model produces outputs with compressed dynamic range (near-black/dim) on sparse fluorescence targets. The V3 GAN significantly outperforms it on all metrics. Investigating root causes:
- Dynamic range compression during ODE integration
- MSE loss averaging over mostly-dark pixels (switched to L1)
- Potential issues in flow matching formulation for sparse targets

## Training

### JiT Flow Matching

```bash
# Single GPU
python scripts/train_jit.py experiment_name=my_run max_steps=10000

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 scripts/train_jit.py \
    experiment_name=my_run max_steps=250000 batch_size=4

# Slurm
sbatch slurm/train_jit.sbatch              # 1x L40S, 10K steps
sbatch slurm/train_jit_multigpu.sbatch     # 4x L40S, 250K steps
```

### V3 GAN

```bash
python scripts/train_deepliif.py experiment_name=v3_baseline use_attention=true
```

See `configs/train_jit.yaml` and `configs/train_deepliif.yaml` for all hyperparameters.

## Data

Expects the [DeepLIIF](https://github.com/nadeemlab/DeepLIIF) dataset in `data/deepliif/` with structure:

```
data/deepliif/
  train/
    <tissue_id>_<channel>.png   # Channels: IHC, Hematoxylin, DAPI, Lap2, Marker, Seg
  val/
  test/
```

## Installation

```bash
pip install -e .
```

Requires Python >= 3.9, PyTorch >= 2.0 with CUDA.
