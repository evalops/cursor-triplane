from __future__ import annotations

from .config import TrainConfig


def main():
    from . import train as _train

    _train.main()


__all__ = ["TrainConfig", "main"]
