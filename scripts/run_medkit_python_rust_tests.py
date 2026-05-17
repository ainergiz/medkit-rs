"""Run medkit-python Rust tests with an explicit PyO3 interpreter config."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def main() -> None:
    config_path = Path("target/ci-pyo3-config.txt")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_pyo3_config(), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONHOME"] = sys.base_prefix
    env["PYO3_CONFIG_FILE"] = str(config_path.resolve())
    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "medkit-python",
            "--no-default-features",
            "--lib",
            "--locked",
        ],
        check=True,
        env=env,
    )


def _pyo3_config() -> str:
    lib_name = (sysconfig.get_config_var("LDLIBRARY") or "").removeprefix("lib")
    for suffix in (".dylib", ".so", ".a", ".dll"):
        lib_name = lib_name.removesuffix(suffix)
    version = sysconfig.get_config_var("VERSION") or f"{sys.version_info.major}.{sys.version_info.minor}"
    return "\n".join(
        [
            "implementation=CPython",
            f"version={version}",
            "shared=true",
            "abi3=true",
            f"lib_name={lib_name}",
            f"lib_dir={sysconfig.get_config_var('LIBDIR') or ''}",
            f"executable={sys.executable}",
            "pointer_width=64",
            "build_flags=",
            "suppress_build_script_link_lines=false",
            "",
        ]
    )


if __name__ == "__main__":
    main()
