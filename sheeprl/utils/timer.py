# timer.py
from __future__ import annotations

import time
from contextlib import ContextDecorator, contextmanager
from typing import Dict, Iterator, Optional, Type, Union

import torch
from torchmetrics import Metric, SumMetric


class TimerError(Exception):
    """A custom exception used to report errors in use of timer class"""


class timer(ContextDecorator):
    """A timer class to measure the time of a code block."""

    disabled: bool = False
    timers: Dict[str, Metric] = {}
    _start_time: Optional[float] = None

    def __init__(self, name: str, metric: Optional[Type[Metric]] = None, **kwargs) -> None:
        """Add timer to dict of timers after initialization"""
        self.name = name
        if not timer.disabled and self.name is not None and self.name not in self.timers.keys():
            self.timers.setdefault(self.name, metric(**kwargs) if metric is not None else SumMetric(**kwargs))

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.name:
            self.timers[self.name].update(elapsed_time)

        return elapsed_time

    @classmethod
    def to(cls, device: Union[str, torch.device] = "cpu") -> None:
        """Create a new timer on a different device"""
        if cls.timers:
            for k, v in cls.timers.items():
                cls.timers[k] = v.to(device)

    @classmethod
    def reset(cls) -> None:
        """Reset all timers"""
        for timer in cls.timers.values():
            timer.reset()
        cls._start_time = None

    @classmethod
    def compute(cls) -> Dict[str, torch.Tensor]:
        """Reduce the timers to a single value"""
        reduced_timers = {}
        if cls.timers:
            for k, v in cls.timers.items():
                reduced_timers[k] = v.compute().item()
        return reduced_timers

    def __enter__(self):
        """Start a new timer as a context manager"""
        if not timer.disabled:
            self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        if not timer.disabled:
            self.stop()


def phase_timer(name: str) -> timer:
    """The timer of a phase of the iterations (`Time/env_interaction_time`, `Time/train_time`), timed by every process
    on its own: the processes play and train at the same time, and summing their times (`metric.sync_on_compute`)
    would divide the speeds of the run (`Time/sps_*`) by their number."""
    return timer(name, SumMetric, sync_on_compute=False)


@contextmanager
def training_timer(device: Union[str, torch.device]) -> Iterator[None]:
    """The timer of the training phase (`Time/train_time`), which waits for the GPU at its end: the GPU runs the
    training after the CPU has launched it, and without waiting the training would be timed with the interaction that
    follows (whose first copy to the CPU waits for it)."""
    with phase_timer("Time/train_time"):
        yield
        if not timer.disabled and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
