import sys, os

_here = os.path.dirname(__file__)
_candidates = [
    os.path.join(_here, "..", "..", "build", "bin", "Debug"),
    os.path.join(_here, "..", "..", "build", "bin", "Release"),
]
for _dir in _candidates:
    _abs = os.path.abspath(_dir)
    if os.path.isdir(_abs):
        sys.path.insert(0, _abs)

from synaption_core import Tensor, Network, Activation, Loss, SGD  # noqa

__all__ = ["Tensor", "Network", "Activation", "Loss", "SGD"]