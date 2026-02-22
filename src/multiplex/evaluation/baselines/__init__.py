"""Baseline methods for image translation comparison.

This module provides wrappers for baseline methods (pix2pix, CycleGAN, CUT, SPADE)
to enable fair comparison with the main multiplex virtual staining model.
"""

from .registry import (
    BaselineConfig,
    BaselineMethod,
    BaselineRegistry,
    register_baseline,
)

# Import baseline implementations to trigger registration
from . import pix2pix
from . import cyclegan
from . import cut
from . import spade
from . import reggan
from . import nicegan

__all__ = [
    "BaselineConfig",
    "BaselineMethod",
    "BaselineRegistry",
    "register_baseline",
]
