"""
Detailed evaluation of the model, its predictions, and the losses.
By: Florian Wiesner
Date: 2025-05-01
"""

from pathlib import Path
from typing import Optional
import logging
import math

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from physicsflow.models.flow_matching.flow_matching_model import FlowMatchingOutput
from physicsflow.train.utils.run_utils import (
    base_attribute,
    compute_metrics,
    reduce_all_losses,
)
from physicsflow.train.utils.logger import setup_logger


class Evaluator:
    """Thorough evaluation of the model on the full dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    dataloader : DataLoader
        Dataloader to evaluate on
    metrics : dict[str, torch.nn.Module]
        Dictionary of metrics to evaluate
    eval_dir : Path
        Directory to save evaluation results
    eval_fraction : float, optional
        Fraction of validation data to use for evaluation (0, 1], by default 1.0
    amp : bool, optional
        Whether to use automatic mixed precision, by default True
    amp_precision : torch.dtype, optional
        Precision to use for AMP, by default torch.bfloat16
    time_bin_count : int, optional
        Number of bins used for t-binned velocity metrics, by default 5
    rollout_num_steps : int, optional
        Number of ODE steps for rollout sample metric, by default 32
    rollout_method : str, optional
        Integration method used for rollout sample metric, by default "euler"
    global_rank : int, optional
        Global rank for distributed training, by default 0
    local_rank : int, optional
        Local rank for distributed training, by default 0
    world_size : int, optional
        World size for distributed training, by default 1
    logger : logging.Logger, optional
        Logger to use, by default None
    """

    BUILTIN_REL_L2_KEY = "velocity_rel_l2"
    BUILTIN_COSINE_ERROR_KEY = "velocity_cosine_error"
    BUILTIN_MMD_KEY = "rollout_mmd_rbf"

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        metrics: dict[str, torch.nn.Module],
        eval_dir: Path,
        eval_fraction: float = 1.0,
        amp: bool = True,
        amp_precision: torch.dtype = torch.bfloat16,
        time_bin_count: int = 5,
        rollout_num_steps: int = 32,
        rollout_method: str = "euler",
        global_rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or setup_logger("Evaluator", rank=global_rank)
        self.global_rank = global_rank
        if not (0.0 < eval_fraction <= 1.0):
            raise ValueError(
                f"eval_fraction must be in the range (0, 1], got {eval_fraction}."
            )
        if time_bin_count <= 0:
            raise ValueError(f"time_bin_count must be positive, got {time_bin_count}.")
        if rollout_num_steps <= 0:
            raise ValueError(
                f"rollout_num_steps must be positive, got {rollout_num_steps}."
            )
        if rollout_method not in {"euler", "midpoint"}:
            raise ValueError(
                "rollout_method must be one of ['euler', 'midpoint'], "
                f"got {rollout_method}."
            )
        self.eval_fraction = eval_fraction
        self.time_bin_count = time_bin_count
        self.rollout_num_steps = rollout_num_steps
        self.rollout_method = rollout_method
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.ddp_enabled = dist.is_initialized()

        self.model = model
        self.model.eval()
        self.model.to(self.device)

        self.dataloader = dataloader
        self.metrics = metrics
        self.eval_dir = eval_dir
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = amp
        self.amp_precision = amp_precision

    def log_msg(self, msg: str):
        """Log a message."""
        self.logger.info(f"{msg}")

    @staticmethod
    def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    @classmethod
    def _per_sample_relative_l2(
        cls, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        pred_flat = cls._flatten_batch(pred.float())
        target_flat = cls._flatten_batch(target.float())
        err_norm = torch.linalg.vector_norm(pred_flat - target_flat, dim=1)
        target_norm = torch.linalg.vector_norm(target_flat, dim=1)
        return err_norm / (target_norm + eps)

    @classmethod
    def _per_sample_cosine_error(
        cls, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        pred_flat = cls._flatten_batch(pred.float())
        target_flat = cls._flatten_batch(target.float())
        cosine = torch.nn.functional.cosine_similarity(
            pred_flat, target_flat, dim=1, eps=eps
        )
        return 1.0 - cosine

    @classmethod
    def _relative_l2_metric(
        cls, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return cls._per_sample_relative_l2(pred, target).mean()

    @classmethod
    def _cosine_error_metric(
        cls, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return cls._per_sample_cosine_error(pred, target).mean()

    @staticmethod
    def _mmd_rbf_metric(
        x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Compute an RBF-kernel MMD^2 between two sample sets."""
        x_flat = x.reshape(x.shape[0], -1).float()
        y_flat = y.reshape(y.shape[0], -1).float()

        n_x = x_flat.shape[0]
        n_y = y_flat.shape[0]
        if n_x == 0 or n_y == 0:
            return torch.tensor(float("nan"), device=x.device)

        z = torch.cat([x_flat, y_flat], dim=0)
        if z.shape[0] < 2:
            return torch.tensor(0.0, device=x.device)

        dist_zz = torch.cdist(z, z).pow(2)
        off_diag_mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        bandwidth = torch.median(dist_zz[off_diag_mask])
        bandwidth = torch.clamp(bandwidth, min=eps)
        gamma = 1.0 / (2.0 * bandwidth)

        k_xx = torch.exp(-gamma * torch.cdist(x_flat, x_flat).pow(2))
        k_yy = torch.exp(-gamma * torch.cdist(y_flat, y_flat).pow(2))
        k_xy = torch.exp(-gamma * torch.cdist(x_flat, y_flat).pow(2))

        if n_x > 1 and n_y > 1:
            mmd2 = (
                (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n_x * (n_x - 1))
                + (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n_y * (n_y - 1))
                - 2.0 * k_xy.mean()
            )
        else:
            # Fallback for tiny batches where unbiased MMD is undefined.
            mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

        return torch.clamp(mmd2, min=0.0)

    def _time_bin_metric_sums(
        self, pred: torch.Tensor, target: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel_l2 = self._per_sample_relative_l2(pred, target)
        cosine_error = self._per_sample_cosine_error(pred, target)

        bin_indices = torch.clamp(
            (t.float() * self.time_bin_count).long(),
            min=0,
            max=self.time_bin_count - 1,
        )

        rel_l2_sum = torch.zeros(self.time_bin_count, device=self.device)
        cosine_error_sum = torch.zeros(self.time_bin_count, device=self.device)
        counts = torch.zeros(self.time_bin_count, device=self.device)

        for bin_idx in range(self.time_bin_count):
            mask = bin_indices == bin_idx
            if mask.any():
                rel_l2_sum[bin_idx] = rel_l2[mask].sum()
                cosine_error_sum[bin_idx] = cosine_error[mask].sum()
                counts[bin_idx] = mask.sum().float()

        return rel_l2_sum, cosine_error_sum, counts

    def _rollout_mmd_metric(
        self, target: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        sample_fn = base_attribute(self.model, "sample")
        generated = sample_fn(
            shape=tuple(target.shape),
            cond=cond,
            num_steps=self.rollout_num_steps,
            method=self.rollout_method,
        )
        return self._mmd_rbf_metric(generated, target)

    def _time_bin_key(self, metric_key: str, bin_idx: int) -> str:
        start_pct = int(round(100 * bin_idx / self.time_bin_count))
        end_pct = int(round(100 * (bin_idx + 1) / self.time_bin_count))
        return f"{metric_key}_t{start_pct:02d}_{end_pct:02d}"

    @torch.inference_mode()
    def eval(self) -> dict[str, torch.Tensor]:
        """Evaluate the model on the full dataset.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of metrics (losses) averaged over the full dataset.
        """
        total_metrics = {}
        for metric_name, _ in self.metrics.items():
            total_metrics[metric_name] = torch.tensor(0.0, device=self.device)
        total_metrics[self.BUILTIN_REL_L2_KEY] = torch.tensor(0.0, device=self.device)
        total_metrics[self.BUILTIN_COSINE_ERROR_KEY] = torch.tensor(
            0.0, device=self.device
        )
        total_metrics[self.BUILTIN_MMD_KEY] = torch.tensor(0.0, device=self.device)
        batch_average_metric_names = list(total_metrics.keys())

        t_bin_rel_l2_sum = torch.zeros(self.time_bin_count, device=self.device)
        t_bin_cosine_error_sum = torch.zeros(self.time_bin_count, device=self.device)
        t_bin_counts = torch.zeros(self.time_bin_count, device=self.device)

        n_total_batches = len(self.dataloader)
        if n_total_batches == 0:
            raise ValueError("Validation dataloader is empty.")
        n_batches = min(
            n_total_batches,
            max(1, math.ceil(n_total_batches * self.eval_fraction)),
        )
        log_interval = max(1, 10 ** math.floor(math.log10(max(1, n_batches // 100))))
        model_input_dtype = next(self.model.parameters()).dtype

        for i, data in enumerate(self.dataloader):
            if (i + 1) % log_interval == 0 or i == 0:
                self.log_msg(f"Batch {i + 1}/{n_batches}")

            # Move all tensors in dict to device
            data = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }
            t1 = data["output_fields"]
            cond = data["constant_scalars"]
            if t1.is_floating_point():
                t1 = t1.to(dtype=model_input_dtype)
            if cond is not None and cond.is_floating_point():
                cond = cond.to(dtype=model_input_dtype)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_precision,
                enabled=self.use_amp,
            ):
                output: FlowMatchingOutput = self.model(t1, cond)
                rollout_mmd = self._rollout_mmd_metric(target=t1, cond=cond)

            current_metrics = compute_metrics(
                output.predicted_velocity, output.target_velocity, self.metrics
            )
            current_metrics[self.BUILTIN_REL_L2_KEY] = self._relative_l2_metric(
                output.predicted_velocity, output.target_velocity
            )
            current_metrics[self.BUILTIN_COSINE_ERROR_KEY] = self._cosine_error_metric(
                output.predicted_velocity, output.target_velocity
            )
            current_metrics[self.BUILTIN_MMD_KEY] = rollout_mmd

            (batch_rel_l2_sum, batch_cosine_error_sum, batch_counts) = (
                self._time_bin_metric_sums(
                    output.predicted_velocity, output.target_velocity, output.t
                )
            )
            t_bin_rel_l2_sum += batch_rel_l2_sum
            t_bin_cosine_error_sum += batch_cosine_error_sum
            t_bin_counts += batch_counts

            current_metrics = reduce_all_losses(current_metrics)
            for metric_name, metric_value in current_metrics.items():
                total_metrics[metric_name] += metric_value.float()

            if i + 1 >= n_batches:
                break

        for metric_name in batch_average_metric_names:
            total_metrics[metric_name] /= n_batches

        if self.ddp_enabled:
            dist.all_reduce(t_bin_rel_l2_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_bin_cosine_error_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_bin_counts, op=dist.ReduceOp.SUM)

        for bin_idx in range(self.time_bin_count):
            rel_l2_key = self._time_bin_key(self.BUILTIN_REL_L2_KEY, bin_idx)
            cosine_key = self._time_bin_key(self.BUILTIN_COSINE_ERROR_KEY, bin_idx)
            count = t_bin_counts[bin_idx]
            if count.item() > 0:
                total_metrics[rel_l2_key] = t_bin_rel_l2_sum[bin_idx] / count
                total_metrics[cosine_key] = t_bin_cosine_error_sum[bin_idx] / count
            else:
                total_metrics[rel_l2_key] = torch.tensor(
                    float("nan"), device=self.device
                )
                total_metrics[cosine_key] = torch.tensor(
                    float("nan"), device=self.device
                )

        return total_metrics
