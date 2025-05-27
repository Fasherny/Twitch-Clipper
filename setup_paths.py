#!/usr/bin/env python3
"""
BeastClipper Path Setup
Sets up the Python path to include all necessary directories
"""

import os
import sys
from pathlib import Path

# Get the root directory
ROOT_DIR = Path(__file__).resolve().parent

# Add the git/BeastClipperDRAFT directory to the path
BEAST_CLIPPER_DIR = ROOT_DIR / "git" / "BeastClipperDRAFT"
if BEAST_CLIPPER_DIR.exists():
    sys.path.insert(0, str(BEAST_CLIPPER_DIR))
    print(f"Added {BEAST_CLIPPER_DIR} to Python path")
else:
    print(f"Warning: {BEAST_CLIPPER_DIR} does not exist")

# Add the root directory to the path (for modules in root)
sys.path.insert(0, str(ROOT_DIR))
print(f"Added {ROOT_DIR} to Python path") 