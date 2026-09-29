"""Initializing a policy from another one (curriculum, 3-DOF -> 6-DOF).

Observations always start with the same quantities (the aircraft's own state, then the
previous action); only the task-specific quantities, at the end, change. We therefore copy:

* every parameter with the same shape (3-DOF -> 6-DOF on the same task: full copy);
* for a first layer where only the number of inputs differs, the columns of the shared
  inputs (new inputs start with zero weights: initially the policy ignores them and
  behaves like the old one);
* nothing else (output layers for a different action space are reinitialized).

The exploration standard deviation (``log_std``) is **not** copied: the source policy
reduced its exploration for its own task; reusing it as is prevents discovering the new
one (observed on the sustained turn: the agent coming from stabilization no longer dared
to bank). The new experiment's initial standard deviation is kept.

SAC critics, whose input is the concatenated (observation, action), are only copied if
the shapes match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass
class TransferReport:
    copied: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{len(self.copied)} tensors copied, {len(self.partial)} partially "
            f"(shared inputs), {len(self.skipped)} reinitialized"
        )


def transfer_weights(source: BaseAlgorithm, target: BaseAlgorithm) -> TransferReport:
    """Copies the compatible weights of ``source`` into ``target`` (in place)."""
    report = TransferReport()
    src = source.policy.state_dict()
    dst = target.policy.state_dict()
    for name, tensor in dst.items():
        if name not in src or name == "log_std":
            report.skipped.append(name)
            continue
        s = src[name]
        if s.shape == tensor.shape:
            dst[name] = s.clone()
            report.copied.append(name)
        elif (
            s.ndim == 2
            and tensor.ndim == 2
            and s.shape[0] == tensor.shape[0]
            and "critic" not in name
            and "qf" not in name
        ):
            n = min(s.shape[1], tensor.shape[1])
            new = torch.zeros_like(tensor)
            new[:, :n] = s[:, :n]
            dst[name] = new
            report.partial.append(name)
        else:
            report.skipped.append(name)
    target.policy.load_state_dict(dst)
    return report


def resolve_model_path(init_from: str, runs_dir: str | Path = "runs") -> Path:
    """``init_from``: path to a ``.zip``, or name of an already trained experiment
    (then the best model of the best seed of its latest run is used)."""
    path = Path(init_from)
    if path.suffix == ".zip" or path.exists():
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    from jetfighter_rl.training.train import best_model_of  # deferred import (avoids a cycle)

    return best_model_of(init_from, runs_dir)
