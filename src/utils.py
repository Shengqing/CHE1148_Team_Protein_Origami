from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device

    @staticmethod
    def auto() -> "DeviceConfig":
        return DeviceConfig(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))