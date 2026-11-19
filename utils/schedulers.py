"""
Linear Warmup with Cosine Annealing Learning Rate Scheduler (without restarts).
Source: https://github.com/nguhcv/cl-tad/blob/main/utils/linearwarmup_cosineLR.py
Khuyến nghị bởi GS. Geoffrey Hinton trong SimCLR (ICML 2020, Section 4).
"""

from __future__ import annotations
import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Schedules learning rate with linear warmup followed by cosine annealing decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Số epoch thực hiện linear warmup (ví dụ: 5 hoặc 10).
            max_epochs (int): Tổng số epoch của quá trình huấn luyện.
            warmup_start_lr (float): Tốc độ học khởi đầu cho warmup (mặc định 0.0).
            eta_min (float): Tốc độ học tối thiểu khi kết thúc cosine decay (mặc định 0.0).
            last_epoch (int): Chỉ số epoch cuối cùng (mặc định -1).
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler.
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            div = max(1, self.warmup_epochs - 1)
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / div
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            div = max(1, self.warmup_epochs - 1)
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / div
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]


# ==============================================================================
# Unit Test tự kiểm chứng
# ==============================================================================

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("Testing LinearWarmupCosineAnnealingLR (utils/schedulers.py)...")
    print("=" * 60)

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    warmup_epochs = 5
    max_epochs = 20
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=1e-5,
        eta_min=1e-6,
    )

    print(f"Base LR: 1e-3 | Warmup Epochs: {warmup_epochs} | Max Epochs: {max_epochs}\n")
    lrs = []
    for epoch in range(max_epochs):
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        if epoch in [0, warmup_epochs - 1, warmup_epochs, max_epochs // 2, max_epochs - 1]:
            print(f"  Epoch {epoch:2d}: LR = {current_lr:.6f}")
        optimizer.step()
        scheduler.step()

    # Kiểm tra tính chất: Sau warmup, LR phải đạt đỉnh base_lr và cuối cùng giảm dần
    assert lrs[warmup_epochs] >= lrs[0], "Warmup LR should increase!"
    assert lrs[-1] <= lrs[warmup_epochs], "Cosine decay should decrease LR!"

    print("\n" + "=" * 60)
    print("[PASSED] LinearWarmupCosineAnnealingLR verified successfully!")
    print("=" * 60)