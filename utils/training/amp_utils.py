"""
AMP (Automatic Mixed Precision) utilities.

Provides:
- AmpWandbCallback: logs AMP-related metrics (loss scale, precision) to W&B
  when available, with graceful fallbacks when Lightning or W&B are absent.
"""

from typing import Optional

try:
    from pytorch_lightning.callbacks import Callback  # type: ignore
except Exception:  # pragma: no cover - fallback when Lightning not installed
    class Callback:  # type: ignore
        pass


class AmpWandbCallback(Callback):
    """
    Lightweight callback to log AMP loss scale and precision to W&B.

    Attempts to introspect Lightning's precision plugin to read the
    underlying torch.cuda.amp GradScaler scale (when using fp16 mixed).
    If not available, logs only enabled/precision flags.
    """

    def __init__(self, enabled: bool, precision: str):
        super().__init__()
        self.enabled = enabled
        self.precision = precision

    def _get_loss_scale(self, trainer) -> Optional[float]:
        try:
            strategy = getattr(trainer, 'strategy', None)
            if strategy is None:
                return None
            pp = getattr(strategy, 'precision_plugin', None)
            if pp is None:
                return None
            scaler = getattr(pp, 'scaler', None)
            if scaler is None:
                return None
            # torch.cuda.amp.GradScaler supports get_scale()
            if hasattr(scaler, 'get_scale'):
                return float(scaler.get_scale())
        except Exception:
            return None
        return None

    def on_train_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        try:
            import wandb  # type: ignore
            if not getattr(wandb, 'run', None):
                return
            log = {
                'amp/enabled': 1 if self.enabled else 0,
                'amp/precision': self.precision,
            }
            scale = None
            if self.enabled and (self.precision in ('16', '16-mixed', '16_true')):
                scale = self._get_loss_scale(trainer)
                if scale is not None:
                    log['amp/loss_scale'] = float(scale)
            # Use epoch as step if available
            step = getattr(trainer, 'current_epoch', None)
            wandb.log(log, step=step)
        except Exception:
            # W&B not installed or logging unavailable; ignore
            pass


def compute_effective_precision(requested_precision: str,
                                use_amp: Optional[bool],
                                cuda_available: bool,
                                use_gpu: bool) -> str:
    """
    Decide final precision string based on AMP flag, device availability,
    and requested default.

    Returns one of: '32', '16', 'bf16' (we keep existing requested value
    when use_amp is None).
    """
    if use_amp is None:
        return requested_precision
    if use_amp and cuda_available and use_gpu:
        return '16'
    return '32'
