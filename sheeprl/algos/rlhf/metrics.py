from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
from torchmetrics import Metric
from torchmetrics.wrappers import Running


class LastValueMetric(Metric):
    last_value: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("last_value", default=torch.tensor(torch.nan), dist_reduce_fx="mean")

    def update(self, value: Union[torch.Tensor, float]):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.last_value = value

    def compute(self):
        return self.last_value

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(last_value: {self.last_value}, device: {self.device})"


class StatsMetric(Metric):
    mean: torch.Tensor
    std: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("mean", default=torch.tensor(torch.nan), dist_reduce_fx="mean")
        self.add_state("std", default=torch.tensor(torch.nan), dist_reduce_fx="mean")
        self.add_state("min", default=torch.tensor(torch.nan), dist_reduce_fx="mean")
        self.add_state("max", default=torch.tensor(torch.nan), dist_reduce_fx="mean")

    def update(self, value: torch.Tensor):
        self.mean = value.mean()
        self.std = value.std()
        self.min = value.min()
        self.max = value.max()

    def compute(self):
        return {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max}


class MetricManager:
    def __init__(self, log_interval: int):
        self.log_interval = log_interval
        for metric_name, type_val in self.__annotations__.items():
            if issubclass(type_val, Metric):
                setattr(self, metric_name, type_val())
            elif issubclass(type_val, Running):
                setattr(self, metric_name, type_val(window=self.log_interval))
            else:
                raise ValueError(
                    f"Expected type of {metric_name} instance of `torchmetrics.wrappers.Running`, but got {type_val}"
                )

    def to(self, device):
        for metric_name, _ in self.__annotations__.items():
            metric = getattr(self, metric_name)
            metric.to(device)
        return self

    def __str__(self) -> str:
        out = []
        for metric_name, metric_class in self.__annotations__.items():
            if issubclass(metric_class, Metric):
                metric = getattr(self, metric_name)
                out.append(f"{metric.__str__()}")
        return "\n".join(out)

    def format_metric_name(self, metric_name: str):
        words = metric_name.split("_")
        if len(words) == 1:
            return f"info/{metric_name}"
        context = metric_name.split("_")[0]
        name = "_".join(metric_name.split("_")[1:])
        return f"{context}/{name}"

    def compute_all(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        metrics = {}
        for metric_name, _ in self.__annotations__.items():
            formatted_metric_name = self.format_metric_name(metric_name)
            if exclude is not None and formatted_metric_name in exclude:
                continue
            metric = getattr(self, metric_name)
            if not metric.update_called:
                continue
            computed_value = metric.compute()
            if isinstance(computed_value, dict):
                for k, v in computed_value.items():
                    if not torch.isnan(v):
                        metrics[f"{formatted_metric_name}_{k}"] = v.item()
            elif isinstance(computed_value, torch.Tensor):
                if not torch.isnan(computed_value):
                    metrics[formatted_metric_name] = computed_value.item()
            else:
                raise ValueError(f"Invalid type for {metric_name}: {type(computed_value)}")
        return metrics

    def log_all(self, fabric: L.Fabric, step: int, metrics_dict: Dict[str, Any], exclude: Optional[List[str]] = None):
        for formatted_metric_name, metric_value in metrics_dict.items():
            if exclude is not None and formatted_metric_name in exclude:
                continue
            fabric.log(formatted_metric_name, metric_value, step=step)


@torch.inference_mode()
def reward_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor):
    tp = torch.count_nonzero(chosen_rewards > rejected_rewards)
    total = chosen_rewards.shape[0]
    acc = tp / total
    return acc


# TODO: Replace some of them with RunningMean when the bug is fixed in `torchmetrics`
# in this way, we will have less noisy data in the logs


class SFTMetricManager(MetricManager):
    train_loss: LastValueMetric
    val_loss: LastValueMetric
    info_grad_norm: LastValueMetric
    info_lr: LastValueMetric
    info_tokens_per_seconds: LastValueMetric
    info_padding_percentage: LastValueMetric
    info_time: LastValueMetric


class RMMetricManager(MetricManager):
    train_loss: LastValueMetric
    train_acc: LastValueMetric
    val_loss: LastValueMetric
    val_acc: LastValueMetric
    info_lr: LastValueMetric
    info_time: LastValueMetric
    info_reward_margin: LastValueMetric
    info_choosen_reward: LastValueMetric
    info_rejected_reward: LastValueMetric
    info_grad_norm: LastValueMetric


class DPOMetricManager(MetricManager):
    train_loss: LastValueMetric
    train_acc: LastValueMetric
    val_loss: LastValueMetric
    val_acc: LastValueMetric
    info_lr: LastValueMetric
    info_time: LastValueMetric
    info_choosen_reward: LastValueMetric
    info_rejected_reward: LastValueMetric
    info_reward_margin: LastValueMetric
    info_grad_norm: LastValueMetric


class PPOMetricManager(MetricManager):
    train_actor_loss: LastValueMetric
    train_critic_loss: LastValueMetric
    train_reward_mean: LastValueMetric
    train_kl_div_mean: LastValueMetric
    info_lr: LastValueMetric
    info_ppo_time: LastValueMetric
    info_rollout_time: LastValueMetric
    info_kl_coeff: LastValueMetric
    info_actor_grad_norm: LastValueMetric
    info_critic_grad_norm: LastValueMetric
    debug_reward_scores: StatsMetric
    debug_advantages: StatsMetric
    debug_returns: StatsMetric
