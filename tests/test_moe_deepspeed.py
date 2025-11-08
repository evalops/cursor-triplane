import pytest

pytest.importorskip("torch")

from trainer.moe_deepspeed import HAS_DEEPSPEED, build_moe


def test_build_moe_guard():
    if not HAS_DEEPSPEED:
        with pytest.raises(RuntimeError):
            build_moe(64, 128, 2, 1.25)
    else:  # pragma: no cover - exercised only when deepspeed is available
        layer = build_moe(64, 128, 2, 1.25)
        assert layer.num_experts == 2
