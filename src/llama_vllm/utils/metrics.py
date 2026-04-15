"""Dual WandB + TensorBoard metrics logging."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsTracker:
    """Track and report training/evaluation metrics."""

    run_name: str
    output_dir: str
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    project: str = "llama-vllm"
    _step: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)
    _wandb_run: Any = field(default=None, init=False)
    _tb_writer: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if "wandb" in self.report_to:
            self._init_wandb()
        if "tensorboard" in self.report_to:
            self._init_tensorboard()

    def _init_wandb(self) -> None:
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.project,
                name=self.run_name,
                dir=self.output_dir,
                resume="allow",
            )
            logger.info(f"WandB initialized: {self._wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed. Skipping WandB logging.")
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.output_dir, "tensorboard", self.run_name)
            self._tb_writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard logs: {tb_dir}")
        except ImportError:
            logger.warning("tensorboard not installed. Skipping TensorBoard logging.")
        except Exception as e:
            logger.warning(f"TensorBoard init failed: {e}")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics."""
        step = step if step is not None else self._step
        self._step = step

        # Console
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"[Step {step}] {metric_str}")

        # WandB
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception:
                pass

        # TensorBoard
        if self._tb_writer is not None:
            try:
                for k, v in metrics.items():
                    self._tb_writer.add_scalar(k, v, global_step=step)
            except Exception:
                pass

        self._step += 1

    def log_throughput(self, tokens: int, elapsed: Optional[float] = None) -> float:
        """Compute and log tokens per second."""
        elapsed = elapsed or (time.time() - self._start_time)
        tps = tokens / max(elapsed, 1e-6)
        self.log({"throughput/tokens_per_sec": tps})
        return tps

    def finish(self) -> None:
        """Close all logging backends."""
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
        logger.info("Metrics tracking finished.")

