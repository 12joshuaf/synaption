# Thin Python wrapper so `import synaption` works and hides the .pyd detail
import sys, os

# Point at the freshly built binary next to your Python venv/scripts
_bin_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "bin", "Release")
sys.path.insert(0, os.path.abspath(_bin_dir))

from synaption_core import Tensor, Network  # noqa

__all__ = ["Tensor", "Network"]