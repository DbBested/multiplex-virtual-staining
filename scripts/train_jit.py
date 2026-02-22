#!/usr/bin/env python3
"""JiT flow matching training script.

Hydra-based entry point for training ConditionalJiT with flow matching.
Handles dataset loading, model creation, W&B initialization, and training
orchestration via JiTTrainer.

Supports both single-GPU and multi-GPU (DDP via torchrun) training:
    python scripts/train_jit.py                              # single GPU
    python scripts/train_jit.py max_steps=10000 batch_size=2 # override
    torchrun --nproc_per_node=4 scripts/train_jit.py         # 4-GPU DDP

Reference:
    Phase 21 Plan 02: JiT Training Pipeline
    Phase 25 Plan 02: Multi-GPU DDP Training
"""

import logging
import os
import sys

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DistributedSampler

logger = logging.getLogger(__name__)


def setup_distributed():
    """Setup DDP if running via torchrun.

    Detects distributed training from environment variables set by torchrun.
    Returns (0, 1, 0) for single-GPU training (no env vars).

    Returns:
        Tuple of (rank, world_size, local_rank).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


@hydra.main(version_base=None, config_path="../configs", config_name="train_jit")
def train(cfg: DictConfig) -> None:
    """Train ConditionalJiT with flow matching.

    Args:
        cfg: Hydra config loaded from configs/train_jit.yaml.
    """
    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. JiT training requires a GPU.")
        sys.exit(1)

    if rank == 0:
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"PyTorch: {torch.__version__}")
        if world_size > 1:
            print(f"DDP: {world_size} GPUs")
        print()

    # ------------------------------------------------------------------
    # Config summary
    # ------------------------------------------------------------------
    if rank == 0:
        print("=" * 60)
        print("JiT Flow Matching Training")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)
        print()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    from multiplex.data.deepliif_dataset import DeepLIIFDataset

    train_dataset = DeepLIIFDataset(
        root_dir=cfg.data_root,
        split=cfg.train_split,
        normalize_to="0_1",
        target_channels=list(cfg.target_channels),
    )
    val_dataset = DeepLIIFDataset(
        root_dir=cfg.data_root,
        split=cfg.val_split,
        normalize_to="0_1",
        target_channels=list(cfg.target_channels),
    )

    # DataLoader with optional DistributedSampler
    train_sampler = None
    shuffle_train = True
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle_train = False  # sampler handles shuffling

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Verify dataset returns hematoxylin for conditioning
    sample = train_dataset[0]
    assert 'hematoxylin' in sample, "Dataset must return 'hematoxylin' key for FiLM conditioning"
    if rank == 0:
        print(f"Source: ihc {sample['ihc'].shape} + hematoxylin {sample['hematoxylin'].shape}")
        print(f"Target: {sample['targets'].shape} ({list(cfg.target_channels)})")

        print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
        if world_size > 1:
            print(f"Effective batch size: {cfg.batch_size} x {world_size} = {cfg.batch_size * world_size}")
        print()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from multiplex.models.jit import ConditionalJiT

    model = ConditionalJiT(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_chans=cfg.in_chans,
        out_chans=cfg.out_chans,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        freeze_encoder=cfg.freeze_encoder,
        # MarkerGNN (Phase 23)
        use_marker_gnn=cfg.use_marker_gnn,
        marker_gnn_node_dim=cfg.marker_gnn_node_dim,
        marker_gnn_heads=cfg.marker_gnn_heads,
        marker_gnn_layers=cfg.marker_gnn_layers,
        marker_gnn_dropout=cfg.marker_gnn_dropout,
        marker_gnn_bio_prior=cfg.marker_gnn_bio_prior,
        # Gradient checkpointing (Phase 25)
        use_gradient_checkpointing=getattr(cfg, 'gradient_checkpointing', False),
        nonzero_init=getattr(cfg, 'nonzero_init', False),
        # CNN decoder + bridge matching
        use_conv_upsample=getattr(cfg, 'use_conv_upsample', False),
        use_cnn_decoder=getattr(cfg, 'use_cnn_decoder', False),
        use_bridge=getattr(cfg, 'use_bridge', False),
        source_chans=4,  # IHC RGB (3) + Hematoxylin (1)
    ).to(device)

    # Keep reference to unwrapped model for EMA and checkpointing
    unwrapped_model = model

    use_static_graph = getattr(cfg, 'ddp_static_graph', True)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            static_graph=use_static_graph,
            find_unused_parameters=not use_static_graph,
        )

    if rank == 0:
        total_params = sum(p.numel() for p in unwrapped_model.parameters())
        trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"Parameters: {total_params / 1e6:.1f}M total | "
              f"{trainable_params / 1e6:.1f}M trainable | "
              f"{frozen_params / 1e6:.1f}M frozen")
        print(f"MarkerGNN: {'enabled' if cfg.use_marker_gnn else 'disabled'}")
        print(f"Conditioning: p_ihc_only={cfg.p_ihc_only}, p_ihc_h={cfg.p_ihc_h}")
        if world_size > 1:
            print(f"DDP: static_graph={use_static_graph}, gradient_checkpointing={getattr(cfg, 'gradient_checkpointing', False)}")
        print()

    # ------------------------------------------------------------------
    # W&B (rank 0 only)
    # ------------------------------------------------------------------
    wandb_run = None
    if rank == 0:
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.experiment_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            print(f"W&B run: {wandb_run.url}")
        except Exception as e:
            print(f"W&B not available or init failed: {e}")
            print("Continuing without W&B logging.")
        print()

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    from multiplex.training.jit_trainer import JiTTrainer

    trainer = JiTTrainer(
        model=model,
        config=cfg,
        device=device,
        wandb_run=wandb_run,
        rank=rank,
        world_size=world_size,
        unwrapped_model=unwrapped_model if world_size > 1 else None,
        train_sampler=train_sampler,
    )

    # Resume from checkpoint if specified
    if cfg.resume_from is not None:
        resumed_step = trainer.load_checkpoint(cfg.resume_from)
        if rank == 0:
            print(f"Resumed from step {resumed_step}")
    if rank == 0:
        print()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.train(train_loader, val_loader, cfg.max_steps)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if wandb_run is not None:
        wandb_run.finish()
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    train()
