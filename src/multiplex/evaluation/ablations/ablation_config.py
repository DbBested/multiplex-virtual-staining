"""
Ablation configuration dataclasses and standard ablation experiments.

This module defines the AblationConfig dataclass for systematic ablation studies
and provides 7 standard configurations for evaluating component contributions:
1. full - all components enabled (baseline)
2. no_pretrain - random initialization instead of pretrained encoder
3. shared_head - single output head instead of per-marker heads
4. no_patchnce - disable PatchNCE contrastive loss
5. no_adversarial - disable adversarial (GAN) loss
6. no_attention - standard U-Net without attention gates
7. l1_only - minimal baseline with only L1 loss
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from multiplex.training.config import TrainConfig


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment.

    Each ablation tests the contribution of a specific component by
    disabling it and measuring the impact on model performance.

    Attributes:
        name: Unique identifier for this ablation experiment.
        description: Human-readable description of what this ablation tests.
        use_pretrained: Whether to use pretrained ImageNet weights for encoder.
        use_multitask_heads: Whether to use separate output heads per marker.
        use_patchnce: Whether to use PatchNCE contrastive loss.
        use_adversarial: Whether to use adversarial (GAN) loss.
        use_attention_gates: Whether to use attention gates in decoder.
        epochs: Number of training epochs (keep consistent across ablations).
        batch_size: Batch size per GPU.
        lr: Learning rate.

    Example:
        >>> config = AblationConfig(
        ...     name="no_attention",
        ...     description="Standard U-Net without attention gates",
        ...     use_attention_gates=False,
        ... )
    """

    name: str
    description: str
    use_pretrained: bool = True
    use_multitask_heads: bool = True
    use_patchnce: bool = True
    use_adversarial: bool = True
    use_attention_gates: bool = True
    epochs: int = 100
    batch_size: int = 4
    lr: float = 0.0002

    def to_train_config(
        self,
        experiment_name: Optional[str] = None,
        checkpoint_dir: str = "/tmp/ablation_checkpoints",
    ) -> TrainConfig:
        """Convert AblationConfig to TrainConfig for training.

        Maps ablation component toggles to TrainConfig parameters:
        - use_pretrained -> passed to generator constructor (not in TrainConfig)
        - use_multitask_heads -> affects num_markers in generator
        - use_patchnce -> TrainConfig.use_patchnce
        - use_adversarial -> TrainConfig.lambda_gan (0 if disabled)
        - use_attention_gates -> passed to generator constructor
        - epochs -> TrainConfig.max_epochs
        - batch_size -> TrainConfig.batch_size
        - lr -> TrainConfig.lr

        Args:
            experiment_name: Name for the experiment. Defaults to ablation name.
            checkpoint_dir: Directory for checkpoints.

        Returns:
            TrainConfig with appropriate settings for this ablation.
        """
        return TrainConfig(
            experiment_name=experiment_name or f"ablation_{self.name}",
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            # Loss weights
            lambda_l1=100.0,
            lambda_perc=10.0 if self.use_adversarial else 0.0,  # Keep perceptual with GAN
            lambda_gan=1.0 if self.use_adversarial else 0.0,
            # PatchNCE
            use_patchnce=self.use_patchnce,
            lambda_nce=1.0 if self.use_patchnce else 0.0,
            # Checkpointing
            checkpoint_dir=checkpoint_dir,
        )


# Standard ablation configurations for systematic evaluation
ABLATION_CONFIGS: List[AblationConfig] = [
    # 1. Full model (baseline) - all components enabled
    AblationConfig(
        name="full",
        description="Full model with all components enabled (baseline)",
        use_pretrained=True,
        use_multitask_heads=True,
        use_patchnce=True,
        use_adversarial=True,
        use_attention_gates=True,
    ),
    # 2. No pretrained encoder - random initialization
    AblationConfig(
        name="no_pretrain",
        description="Random encoder initialization (no ImageNet pretraining)",
        use_pretrained=False,
        use_multitask_heads=True,
        use_patchnce=True,
        use_adversarial=True,
        use_attention_gates=True,
    ),
    # 3. Shared output head - single head instead of per-marker
    AblationConfig(
        name="shared_head",
        description="Shared output head instead of per-marker heads",
        use_pretrained=True,
        use_multitask_heads=False,
        use_patchnce=True,
        use_adversarial=True,
        use_attention_gates=True,
    ),
    # 4. No PatchNCE loss
    AblationConfig(
        name="no_patchnce",
        description="Disable PatchNCE contrastive loss",
        use_pretrained=True,
        use_multitask_heads=True,
        use_patchnce=False,
        use_adversarial=True,
        use_attention_gates=True,
    ),
    # 5. No adversarial loss
    AblationConfig(
        name="no_adversarial",
        description="Disable adversarial (GAN) loss",
        use_pretrained=True,
        use_multitask_heads=True,
        use_patchnce=True,
        use_adversarial=False,
        use_attention_gates=True,
    ),
    # 6. No attention gates - standard U-Net
    AblationConfig(
        name="no_attention",
        description="Standard U-Net without attention gates",
        use_pretrained=True,
        use_multitask_heads=True,
        use_patchnce=True,
        use_adversarial=True,
        use_attention_gates=False,
    ),
    # 7. L1 only - minimal baseline
    AblationConfig(
        name="l1_only",
        description="Minimal baseline with only L1 reconstruction loss",
        use_pretrained=True,
        use_multitask_heads=True,
        use_patchnce=False,
        use_adversarial=False,
        use_attention_gates=True,
    ),
]


def get_ablation_config(name: str) -> Optional[AblationConfig]:
    """Get ablation config by name.

    Args:
        name: Ablation configuration name.

    Returns:
        AblationConfig if found, None otherwise.
    """
    for config in ABLATION_CONFIGS:
        if config.name == name:
            return config
    return None
