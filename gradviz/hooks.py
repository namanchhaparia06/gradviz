from typing import Any, Dict, List, Optional, Tuple
import torch

class GradHookManager:
    """
    Registers .register_hook(...) on tensors param.grad to collect gradient norms.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._handles: List[Any] = []
        self._params: List[Tuple[str, torch.nn.Parameter]] = []
        self._collect_fn = None

    def _iter_named_params(self):
        # include only params that require grad
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                yield n, p

    def attach(self, collect_fn):
        """
        collect_fn signature: (param_name: str, grad: torch.Tensor) -> None
        """
        self.detach()  # safety
        self._collect_fn = collect_fn
        self._params = list(self._iter_named_params())

        for name, param in self._params:
            # We hook on the gradient AFTER it’s computed (grad hook on tensor)
            def _make_hook(nm):
                def _hook(grad):
                    if grad is None:
                        return
                    self._collect_fn(nm, grad)
                return _hook
            h = param.register_hook(_make_hook(name))
            self._handles.append(h)

    def detach(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._collect_fn = None
        self._params.clear()
