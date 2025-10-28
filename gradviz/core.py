from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
import io
import math
import time
import csv
import os

import torch
import numpy as np
import pandas as pd

from .hooks import GradHookManager
from .utils import GradVizConfig, clean_name, layer_from_param

@dataclass
class _Row:
    step: int
    epoch: Optional[int]
    param: str
    layer: str
    grad_norm: float

class GradViz:
    """
    Minimal gradient visualizer:
    - Attach gradient hooks to a model
    - On every backward, logs per-parameter gradient L2 norms
    - Save to CSV; visualize with built-in plotting utilities
    """
    def __init__(self, model: torch.nn.Module, config: Optional[GradVizConfig] = None):
        self.model = model
        self.cfg = config or GradVizConfig()
        self._mgr = GradHookManager(model)
        self._rows: List[_Row] = []
        self._step_count = 0
        self._epoch: Optional[int] = None
        self._enabled = False

    def set_epoch(self, epoch: int):
        """Optionally call this in your training loop at the start of each epoch."""
        self._epoch = int(epoch)

    def attach(self):
        """Register gradient hooks."""
        if self._enabled:
            return
        def _collect(param_name: str, grad_tensor: torch.Tensor):
            self._on_grad(param_name, grad_tensor)
        self._mgr.attach(_collect)
        self._enabled = True

    def detach(self):
        self._mgr.detach()
        self._enabled = False

    def step(self):
        """
        Call AFTER each optimizer.step() (or after each backward if you prefer).
        This advances the internal step counter and optionally throttles logging frequency.
        """
        self._step_count += 1

    def _on_grad(self, param_name: str, grad_tensor: torch.Tensor):
        # Skip if not logging this step
        if self.cfg.log_every_steps > 1 and (self._step_count % self.cfg.log_every_steps != 0):
            return

        if grad_tensor is None:
            return
        # Compute L2 norm; handle sparse
        try:
            if grad_tensor.is_sparse:
                g = grad_tensor.coalesce().values()
            else:
                g = grad_tensor
            norm = float(torch.linalg.vector_norm(g).detach().cpu().item())
        except Exception:
            return

        if not self.cfg.keep_zero_grads and norm == 0.0:
            return

        pname = clean_name(param_name)
        layer = layer_from_param(pname)
        self._rows.append(_Row(step=self._step_count, epoch=self._epoch, param=pname, layer=layer, grad_norm=norm))

        # Memory guard
        if len(self._rows) > self.cfg.max_rows_in_memory:
            # Drop earliest half if we exceed memory (simple ring buffer)
            self._rows = self._rows[len(self._rows)//2 :]

    # ---------- Export ----------
    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            cols = ["step", "epoch", "param", "layer", "grad_norm"]
            return pd.DataFrame(columns=cols)
        data = {
            "step": [r.step for r in self._rows],
            "epoch": [r.epoch for r in self._rows],
            "param": [r.param for r in self._rows],
            "layer": [r.layer for r in self._rows],
            "grad_norm": [r.grad_norm for r in self._rows],
        }
        return pd.DataFrame(data)

    def save(self, path: str):
        """
        Save collected gradients to CSV.
        """
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)

    # ---------- Quick Plots ----------
    def plot(self, by: str = "layer", topk: int = 20, figsize=(10,6), show=True, savepath: Optional[str]=None):
        """
        Quick line plots of gradient norm over steps.
        by: "layer" or "param"
        """
        from .plotting import plot_lines_by_key
        df = self.to_dataframe()
        if df.empty:
            raise RuntimeError("No gradients recorded yet. Did you call attach() and run training?")
        ax = plot_lines_by_key(df, by=by, topk=topk, figsize=figsize)
        if savepath:
            import matplotlib.pyplot as plt
            plt.savefig(savepath, bbox_inches="tight", dpi=200)
        if show:
            import matplotlib.pyplot as plt
            plt.show()

    def heatmap(self, by: str = "layer", at_step: Optional[int] = None, figsize=(10,6), show=True, savepath: Optional[str]=None):
        """
        Heatmap of gradient norms for a specific step (closest available).
        """
        from .plotting import plot_heatmap_at_step
        df = self.to_dataframe()
        if df.empty:
            raise RuntimeError("No gradients recorded yet.")
        ax = plot_heatmap_at_step(df, by=by, at_step=at_step, figsize=figsize)
        if savepath:
            import matplotlib.pyplot as plt
            plt.savefig(savepath, bbox_inches="tight", dpi=200)
        if show:
            import matplotlib.pyplot as plt
            plt.show()

    # ---------- Context manager ----------
    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.detach()
