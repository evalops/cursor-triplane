import subprocess
from pathlib import Path

import pytest

SCRIPTS = [
    Path("scripts/firecracker/build_base.sh"),
    Path("scripts/firecracker/create_template.sh"),
    Path("scripts/firecracker/launch_envs.sh"),
]


@pytest.mark.parametrize("script_path", SCRIPTS)
def test_firecracker_scripts_shellcheck(script_path):
    full_path = Path(__file__).resolve().parents[1] / script_path
    result = subprocess.run(["bash", "-n", str(full_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_launch_envs_contains_expected_flags():
    script = (Path(__file__).resolve().parents[1] / "scripts/firecracker/launch_envs.sh").read_text()
    assert "--snapshot" in script
    assert "--copy-files" in script
    assert "--cmd" in script
