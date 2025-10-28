from dataclasses import dataclass
from typing import Optional
import re

def clean_name(name: str) -> str:
    # Normalize parameter names for grouping/plotting
    return re.sub(r"\.+", ".", name.strip())

def layer_from_param(param_name: str) -> str:
    # Heuristic: take all but last token (e.g., "encoder.layers.0.attn.q_proj.weight" -> "encoder.layers.0.attn.q_proj")
    toks = param_name.split(".")
    return ".".join(toks[:-1]) if len(toks) > 1 else param_name

@dataclass
class GradVizConfig:
    log_every_steps: int = 1         # record every N steps
    keep_zero_grads: bool = False    # include zero gradients in logs
    max_rows_in_memory: int = 1_000_000
