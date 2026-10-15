"""Re-export of the repo-root device_setup.detect_device for library-internal use.

Imported lazily (inside functions, not at module top) by callers within the
alpha_analysis package to avoid a circular/early import of device_setup relative
to package initialization.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_setup import detect_device  # noqa: E402
