"""Auramaur — Polymarket prediction market trading bot."""

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

__version__ = "0.1.0"
