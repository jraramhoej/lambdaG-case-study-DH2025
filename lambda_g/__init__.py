"""Minimal LambdaG package for authorship verification.
"""

from .model import LambdaGModel
from .posnoise import POSnoiseProcessor

__all__ = ["LambdaGModel", "POSnoiseProcessor"]
