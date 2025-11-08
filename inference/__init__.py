from __future__ import annotations


def bootstrap_ray(*args, **kwargs):
    from .serve import bootstrap_ray as _bootstrap_ray

    return _bootstrap_ray(*args, **kwargs)


__all__ = ["bootstrap_ray"]
