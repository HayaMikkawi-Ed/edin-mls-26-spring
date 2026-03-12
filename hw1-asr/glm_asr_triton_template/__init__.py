"""
Template Baseline - Triton Student Assignment
Performance: TBD (Torch baseline with Triton kernels available)

Key Characteristics:
- Pure Torch tensor operations
- Triton kernels for core ops (student TODOs)
"""

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from . import layers
from . import model
from . import rope
from . import conv
from . import weight_loader

import layers as _layers_direct
_layers_direct.Linear.BACKEND = "triton"
_layers_direct.MLP.FUSED = False
_layers_direct.EncoderMLP.FUSED = False
