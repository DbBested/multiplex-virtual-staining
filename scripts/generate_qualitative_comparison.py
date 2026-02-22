#!/usr/bin/env python
"""Generate qualitative comparison figures: V3 GAN vs JiT Flow Matching.

Creates publication-ready side-by-side comparisons showing:
    Input IHC | Ground Truth | V3 GAN | JiT Flow Matching

For each target channel (DAPI, Lap2, Marker) across multiple samples.

Usage:
    python scripts/generate_qualitative_comparison.py
    python scripts/generate_qualitative_comparison.py --num-samples 6 --split test
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multiplex.data.deepliif_dataset import DeepLIIFDataset
from multiplex.data.transforms import get_val_transforms
from multiplex.models import create_generator, CrossMarkerConfig
from multiplex.models.jit.conditional_jit import ConditionalJiT
from multiplex.training.flow_matching import euler_sample
from multiplex.training.ema import EMA

JIT_CHANNELS = ["DAPI", "Lap2", "Marker"]


def load_v3_model(checkpoint_path: str, device: str):
    """Load V3 GAN model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "generator_state_dict" in ckpt:
        state_dict = ckpt["generator_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    has_cross_marker = any("cross_marker_attention" in k for k in state_dict.keys())
    has_output_refinement = any("output_refinement" in k for k in state_dict.keys())
    has_attention = any("decoders" in k and "attention" in k for k in state_dict.keys())

    if has_cross_marker or has_output_refinement:
        cross_marker_config = CrossMarkerConfig(
            num_markers=4, use_stage1=has_cross_marker, stage1_embed_dim=1024,
            stage1_num_heads=8, use_stage2=has_output_refinement,
            stage2_hidden_dim=64, stage2_num_heads=4,
        )
        use_attention = True
    elif has_attention:
        cross_marker_config = None
        use_attention = True
    else:
        cross_marker_config = None
        use_attention = False

    model = create_generator(
        in_channels=3, num_markers=4, pretrained=False,
        use_attention=use_attention, cross_marker_config=cross_marker_config,
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def load_jit_model(checkpoint_path: str, device: str):
    """Load JiT model with EMA weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    state_dict_check = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", {}))
    has_marker_gnn = any("marker_gnn" in k for k in state_dict_check.keys())

    # Detect decoder type from state dict
    has_decoder = any("decoder." in k for k in state_dict_check.keys())
    has_conv_upsample = cfg.get("use_conv_upsample", False) or (
        has_decoder
        and any("decoder.output_proj" in k for k in state_dict_check.keys())
        and not any("decoder.input_proj" in k for k in state_dict_check.keys())
    )
    has_progressive = has_decoder and not has_conv_upsample

    model = ConditionalJiT(
        img_size=cfg.get("img_size", 512),
        patch_size=cfg.get("patch_size", 32),
        in_chans=cfg.get("in_chans", 3),
        out_chans=cfg.get("out_chans", 3),
        hidden_size=cfg.get("hidden_size", 768),
        depth=cfg.get("depth", 12),
        num_heads=cfg.get("num_heads", 12),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        freeze_encoder=cfg.get("freeze_encoder", True),
        use_marker_gnn=has_marker_gnn,
        marker_gnn_node_dim=cfg.get("marker_gnn_node_dim", 192),
        marker_gnn_heads=cfg.get("marker_gnn_heads", 4),
        marker_gnn_layers=cfg.get("marker_gnn_layers", 2),
        use_conv_upsample=has_conv_upsample,
        use_cnn_decoder=has_progressive,
    )

    state_dict = ckpt.get("model_state_dict", {})
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    ema_sd = ckpt.get("ema_state_dict", {})
    if ema_sd:
        for k, v in ema_sd.items():
            state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    prediction_type = cfg.get("prediction", "x_prediction")
    return model, prediction_type


def to_numpy_01(t):
    """Tensor [0,1] → numpy [0,255] uint8."""
    return (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def to_numpy_11(t):
    """Tensor [-1,1] → numpy [0,255] uint8."""
    return (((t.clamp(-1, 1).cpu().numpy() + 1) / 2) * 255).astype(np.uint8)


def to_numpy_normalized(t):
    """Tensor → numpy [0,255] uint8 with per-channel min/max normalization."""
    arr = t.cpu().float().numpy()
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return (((arr - vmin) / (vmax - vmin)) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-checkpoint", default="outputs/checkpoints/v3_8636958/best_model.pt")
    parser.add_argument("--jit-checkpoint", default="checkpoints/jit/v5_pixelgen/best_model.pt")
    parser.add_argument("--data-root", default="data/deepliif_tissue")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--euler-steps", type=int, default=50)
    parser.add_argument("--output-dir", default="samples/comparison")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    project_dir = Path(__file__).parent.parent

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading V3 GAN...")
    v3_path = project_dir / args.v3_checkpoint
    v3_model = load_v3_model(str(v3_path), device)

    print("Loading JiT Flow Matching...")
    jit_path = project_dir / args.jit_checkpoint
    jit_model, jit_pred_type = load_jit_model(str(jit_path), device)
    print(f"  Prediction type: {jit_pred_type}")

    # Load datasets (V3 uses [-1,1], JiT uses [0,1])
    v3_dataset = DeepLIIFDataset(
        root_dir=str(project_dir / args.data_root),
        split=args.split,
        transform=get_val_transforms(image_size=512),
        normalize_to="-1_1",
    )
    jit_dataset = DeepLIIFDataset(
        root_dir=str(project_dir / args.data_root),
        split=args.split,
        transform=get_val_transforms(image_size=512),
        normalize_to="0_1",
        target_channels=JIT_CHANNELS,
    )

    indices = np.linspace(0, len(jit_dataset) - 1, args.num_samples, dtype=int)

    # =========================================================================
    # Figure 1: Main comparison grid
    # Layout: rows = samples, columns = Input | GT DAPI | GT Lap2 | GT Marker |
    #                                     V3 DAPI | V3 Lap2 | V3 Marker |
    #                                     JiT DAPI | JiT Lap2 | JiT Marker
    # =========================================================================
    n_samples = len(indices)
    fig, axes = plt.subplots(
        n_samples, 10, figsize=(30, 3.2 * n_samples),
        gridspec_kw={"wspace": 0.02, "hspace": 0.08},
    )
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_headers = [
        "Input IHC",
        "GT\nDAPI", "GT\nLap2", "GT\nMarker",
        "V3 GAN\nDAPI", "V3 GAN\nLap2", "V3 GAN\nMarker",
        "JiT FM\nDAPI", "JiT FM\nLap2", "JiT FM\nMarker",
    ]

    for row, idx in enumerate(indices):
        idx = int(idx)

        # V3 prediction
        v3_sample = v3_dataset[idx]
        v3_input = v3_sample["ihc"].unsqueeze(0).to(device)
        with torch.no_grad():
            v3_pred = v3_model(v3_input)[0].cpu()  # (4, H, W) in [-1,1]

        # JiT prediction
        jit_sample = jit_dataset[idx]
        jit_source = torch.cat([jit_sample["ihc"], jit_sample["hematoxylin"]], dim=0)
        jit_source = jit_source.unsqueeze(0).to(device)
        avail = torch.ones(1, dtype=torch.long, device=device)
        with torch.no_grad():
            jit_pred = euler_sample(
                jit_model, jit_source, avail,
                num_steps=args.euler_steps,
                img_size=512, out_chans=3,
                prediction_type=jit_pred_type,
            )[0]  # (3, H, W) in [0,1]

        # Ground truth from JiT dataset (0,1 range)
        gt = jit_sample["targets"]  # (3, H, W) in [0,1]
        ihc_rgb = jit_sample["ihc"]  # (3, H, W) in [0,1]

        # Col 0: Input IHC
        axes[row, 0].imshow(to_numpy_01(ihc_rgb).transpose(1, 2, 0))
        axes[row, 0].axis("off")

        # Col 1-3: Ground truth channels
        for ch in range(3):
            axes[row, ch + 1].imshow(to_numpy_01(gt[ch]), cmap="gray", vmin=0, vmax=255)
            axes[row, ch + 1].axis("off")

        # Col 4-6: V3 GAN predictions (channels 1-3 = DAPI, Lap2, Marker)
        for ch in range(3):
            axes[row, ch + 4].imshow(to_numpy_11(v3_pred[ch + 1]), cmap="gray", vmin=0, vmax=255)
            axes[row, ch + 4].axis("off")

        # Col 7-9: JiT predictions
        for ch in range(3):
            axes[row, ch + 7].imshow(to_numpy_01(jit_pred[ch]), cmap="gray", vmin=0, vmax=255)
            axes[row, ch + 7].axis("off")

    # Column headers
    for j, header in enumerate(col_headers):
        axes[0, j].set_title(header, fontsize=11, fontweight="bold", pad=8)

    # Add group brackets/labels
    fig.text(0.35, 0.98, "Ground Truth", ha="center", fontsize=13,
             fontweight="bold", color="#2c7fb8")
    fig.text(0.62, 0.98, "V3 GAN", ha="center", fontsize=13,
             fontweight="bold", color="#d95f02")
    fig.text(0.88, 0.98, "JiT Flow Matching", ha="center", fontsize=13,
             fontweight="bold", color="#7570b3")

    save_path = out_dir / "v3_vs_jit_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    # =========================================================================
    # Figure 2: Per-channel close-up (single sample, larger)
    # Layout: 3 rows (channels) x 4 cols (Input, GT, V3, JiT)
    # =========================================================================
    best_idx = int(indices[0])

    v3_sample = v3_dataset[best_idx]
    v3_input = v3_sample["ihc"].unsqueeze(0).to(device)
    with torch.no_grad():
        v3_pred = v3_model(v3_input)[0].cpu()

    jit_sample = jit_dataset[best_idx]
    jit_source = torch.cat([jit_sample["ihc"], jit_sample["hematoxylin"]], dim=0)
    jit_source = jit_source.unsqueeze(0).to(device)
    avail = torch.ones(1, dtype=torch.long, device=device)
    with torch.no_grad():
        jit_pred = euler_sample(
            jit_model, jit_source, avail,
            num_steps=args.euler_steps,
            img_size=512, out_chans=3,
            prediction_type=jit_pred_type,
        )[0]

    gt = jit_sample["targets"]
    ihc_rgb = jit_sample["ihc"]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15),
                              gridspec_kw={"wspace": 0.05, "hspace": 0.08})

    col_headers_2 = ["Input IHC", "Ground Truth", "V3 GAN", "JiT Flow Matching"]

    for ch_idx, ch_name in enumerate(JIT_CHANNELS):
        # Input IHC (same for all rows)
        axes[ch_idx, 0].imshow(to_numpy_01(ihc_rgb).transpose(1, 2, 0))
        axes[ch_idx, 0].axis("off")
        axes[ch_idx, 0].set_ylabel(ch_name, fontsize=14, fontweight="bold",
                                    rotation=0, ha="right", va="center", labelpad=15)

        # Ground truth
        axes[ch_idx, 1].imshow(to_numpy_01(gt[ch_idx]), cmap="gray", vmin=0, vmax=255)
        axes[ch_idx, 1].axis("off")

        # V3 GAN (channel offset: 1=DAPI, 2=Lap2, 3=Marker)
        axes[ch_idx, 2].imshow(to_numpy_11(v3_pred[ch_idx + 1]), cmap="gray", vmin=0, vmax=255)
        axes[ch_idx, 2].axis("off")

        # JiT
        axes[ch_idx, 3].imshow(to_numpy_01(jit_pred[ch_idx]), cmap="gray", vmin=0, vmax=255)
        axes[ch_idx, 3].axis("off")

    for j, header in enumerate(col_headers_2):
        axes[0, j].set_title(header, fontsize=14, fontweight="bold", pad=12)

    save_path = out_dir / "v3_vs_jit_detail.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    # =========================================================================
    # Figure 3: Contrast-enhanced comparison (per-channel normalized)
    # Shows structural content independent of intensity range.
    # Layout: 3 rows (channels) x 5 cols (Input, GT, V3, JiT Raw, JiT Normalized)
    # =========================================================================
    fig, axes = plt.subplots(3, 5, figsize=(25, 15),
                              gridspec_kw={"wspace": 0.05, "hspace": 0.08})

    col_headers_3 = ["Input IHC", "Ground Truth", "V3 GAN",
                     "JiT FM (raw)", "JiT FM (enhanced)"]

    for ch_idx, ch_name in enumerate(JIT_CHANNELS):
        axes[ch_idx, 0].imshow(to_numpy_01(ihc_rgb).transpose(1, 2, 0))
        axes[ch_idx, 0].axis("off")
        axes[ch_idx, 0].set_ylabel(ch_name, fontsize=14, fontweight="bold",
                                    rotation=0, ha="right", va="center", labelpad=15)

        gt_ch = gt[ch_idx]
        axes[ch_idx, 1].imshow(to_numpy_01(gt_ch), cmap="gray", vmin=0, vmax=255)
        axes[ch_idx, 1].axis("off")

        v3_ch = v3_pred[ch_idx + 1]
        axes[ch_idx, 2].imshow(to_numpy_11(v3_ch), cmap="gray", vmin=0, vmax=255)
        axes[ch_idx, 2].axis("off")

        jit_ch = jit_pred[ch_idx]
        axes[ch_idx, 3].imshow(to_numpy_01(jit_ch), cmap="gray", vmin=0, vmax=255)
        axes[ch_idx, 3].axis("off")

        # Contrast-enhanced (per-channel min/max normalized)
        axes[ch_idx, 4].imshow(to_numpy_normalized(jit_ch), cmap="gray", vmin=0, vmax=255)
        axes[ch_idx, 4].axis("off")

        # Print per-channel stats
        jit_arr = jit_ch.cpu().float()
        gt_arr = gt_ch.cpu().float()
        print(f"  {ch_name}: JiT [{jit_arr.min():.4f}, {jit_arr.max():.4f}] "
              f"mean={jit_arr.mean():.4f} | GT [{gt_arr.min():.4f}, {gt_arr.max():.4f}] "
              f"mean={gt_arr.mean():.4f}")

    for j, header in enumerate(col_headers_3):
        axes[0, j].set_title(header, fontsize=14, fontweight="bold", pad=12)

    save_path = out_dir / "v3_vs_jit_enhanced.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
