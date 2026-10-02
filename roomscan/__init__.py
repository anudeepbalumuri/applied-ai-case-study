"""roomscan: phone capture -> dimensioned, stitched floor plan with damage, rules and scope."""
import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.environ.setdefault("HF_HOME", _os.path.join(_ROOT, "weights", "hf"))
_os.environ.setdefault("TORCH_HOME", _os.path.join(_ROOT, "weights", "torch"))
_os.environ.setdefault("GLOG_minloglevel", "2")
