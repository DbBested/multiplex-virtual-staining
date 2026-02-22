"""
Ablation experiment runner for systematic evaluation.

This module provides the AblationRunner class for running ablation experiments
and collecting metrics for component contribution analysis.
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from multiplex.evaluation.ablations.ablation_config import (
    ABLATION_CONFIGS,
    AblationConfig,
    get_ablation_config,
)
from multiplex.evaluation.metrics import BatchedMetricAccumulator, MarkerMetrics
from multiplex.models.discriminator import PatchGAN70
from multiplex.models.generator import AttentionUNetGenerator
from multiplex.training.config import TrainConfig
from multiplex.training.losses import DiscriminatorLoss, MultiLoss
from multiplex.training.trainer import Trainer


logger = logging.getLogger(__name__)


class AblationRunner:
    """Runner for ablation experiments.

    Trains models with different ablation configurations and collects
    evaluation metrics for comparison.

    Attributes:
        base_config: Base TrainConfig for settings not varied in ablations.
        output_dir: Directory for saving results and checkpoints.
        device: Training device (cuda or cpu).

    Example:
        >>> runner = AblationRunner(base_config, output_dir="/scratch/ablations")
        >>> results = runner.run_experiment(ablation_config)
        >>> print(results["mean"]["PSNR"])
    """

    def __init__(
        self,
        base_config: TrainConfig,
        output_dir: str,
        device: str = "cuda",
    ):
        """Initialize AblationRunner.

        Args:
            base_config: Base training configuration for common settings.
            output_dir: Directory for saving ablation results.
            device: Training device. Default "cuda".
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info(f"AblationRunner initialized: output_dir={output_dir}, device={self.device}")

    def _create_generator(self, ablation: AblationConfig) -> AttentionUNetGenerator:
        """Create generator with ablation-specific settings.

        Args:
            ablation: Ablation configuration.

        Returns:
            AttentionUNetGenerator with appropriate settings.
        """
        # Determine number of markers for output
        num_markers = 5 if ablation.use_multitask_heads else 1

        return AttentionUNetGenerator(
            in_channels=1,
            num_markers=num_markers,
            pretrained=ablation.use_pretrained,
            use_attention=ablation.use_attention_gates,
            dropout_p=0.0,  # No dropout for ablation training
        )

    def _create_discriminator(self, ablation: AblationConfig) -> PatchGAN70:
        """Create discriminator.

        Args:
            ablation: Ablation configuration.

        Returns:
            PatchGAN70 discriminator.
        """
        # Discriminator input: 1 (BF) + num_markers
        num_markers = 5 if ablation.use_multitask_heads else 1
        return PatchGAN70(in_channels=1 + num_markers)

    def _create_loss(self, ablation: AblationConfig) -> MultiLoss:
        """Create loss function with ablation-specific settings.

        Args:
            ablation: Ablation configuration.

        Returns:
            MultiLoss configured for this ablation.
        """
        return MultiLoss(
            lambda_l1=100.0,
            lambda_perc=10.0 if ablation.use_adversarial else 0.0,
            lambda_gan=1.0 if ablation.use_adversarial else 0.0,
            device=str(self.device),
            use_patchnce=ablation.use_patchnce,
        )

    def run_experiment(
        self,
        ablation: AblationConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run a single ablation experiment.

        Trains a model with the ablation configuration and evaluates
        on validation/test sets.

        Args:
            ablation: Ablation configuration to test.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            test_loader: Test DataLoader (uses val_loader if None).

        Returns:
            Dictionary with:
            - 'config': Ablation configuration dict
            - 'metrics': Per-marker and mean metrics (PSNR, SSIM, LPIPS)
            - 'train_time': Training duration in seconds
        """
        logger.info(f"Starting ablation experiment: {ablation.name}")
        logger.info(f"  Description: {ablation.description}")

        # Create experiment directory
        exp_dir = self.output_dir / ablation.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create models
        generator = self._create_generator(ablation).to(self.device)
        discriminator = self._create_discriminator(ablation).to(self.device)

        # Create training config
        train_config = ablation.to_train_config(
            experiment_name=f"ablation_{ablation.name}",
            checkpoint_dir=str(exp_dir / "checkpoints"),
        )

        # Create trainer
        trainer = Trainer(
            generator=generator,
            discriminator=discriminator,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            rank=0,
            world_size=1,
        )

        # Train
        start_time = datetime.now()
        trainer.fit()
        train_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        eval_loader = test_loader if test_loader is not None else val_loader
        metrics = self._evaluate(generator, eval_loader, ablation)

        # Save results
        results = {
            "name": ablation.name,
            "description": ablation.description,
            "config": asdict(ablation),
            "metrics": metrics,
            "train_time_seconds": train_time,
            "timestamp": datetime.now().isoformat(),
        }

        results_path = exp_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Ablation {ablation.name} complete: PSNR={metrics['mean'].get('psnr', 0):.2f}")

        return results

    def _evaluate(
        self,
        generator: AttentionUNetGenerator,
        data_loader: DataLoader,
        ablation: AblationConfig,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate generator on dataset.

        Args:
            generator: Trained generator.
            data_loader: DataLoader for evaluation.
            ablation: Ablation configuration.

        Returns:
            Dictionary with per-marker and mean metrics.
        """
        generator.eval()

        # Use metric accumulator
        accumulator = BatchedMetricAccumulator(device=str(self.device))

        with torch.no_grad():
            for batch in data_loader:
                bf = batch["bf"].to(self.device)
                markers_real = batch["markers"].to(self.device)

                markers_pred = generator(bf)

                # Handle shared head case: expand single output to 5 channels
                if not ablation.use_multitask_heads and markers_pred.shape[1] == 1:
                    markers_pred = markers_pred.expand(-1, 5, -1, -1)

                accumulator.update(markers_pred, markers_real)

        return accumulator.compute()

    def run_all(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        ablations: Optional[List[AblationConfig]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all ablation experiments.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            test_loader: Test DataLoader (optional).
            ablations: List of ablation configs to run. Defaults to all standard configs.

        Returns:
            Dictionary mapping ablation name to results.
        """
        if ablations is None:
            ablations = ABLATION_CONFIGS

        all_results = {}
        for ablation in ablations:
            try:
                results = self.run_experiment(
                    ablation, train_loader, val_loader, test_loader
                )
                all_results[ablation.name] = results
            except Exception as e:
                logger.error(f"Ablation {ablation.name} failed: {e}")
                all_results[ablation.name] = {"error": str(e)}

        # Save combined results
        combined_path = self.output_dir / "all_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)

        return all_results


def run_single_ablation(
    ablation_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    device: str = "cuda",
    test_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """Run a single ablation experiment by name.

    Convenience function for running one ablation from CLI.

    Args:
        ablation_name: Name of ablation configuration.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        output_dir: Output directory.
        device: Training device.
        test_loader: Optional test DataLoader.

    Returns:
        Ablation results dictionary.

    Raises:
        ValueError: If ablation name not found.
    """
    ablation = get_ablation_config(ablation_name)
    if ablation is None:
        available = [c.name for c in ABLATION_CONFIGS]
        raise ValueError(f"Unknown ablation: {ablation_name}. Available: {available}")

    # Create a minimal base config
    base_config = TrainConfig(experiment_name=f"ablation_{ablation_name}")

    runner = AblationRunner(
        base_config=base_config,
        output_dir=output_dir,
        device=device,
    )

    return runner.run_experiment(
        ablation, train_loader, val_loader, test_loader
    )
