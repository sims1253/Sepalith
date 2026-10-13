#!/usr/bin/env python3
"""Exec the reviewed diagnostic submitter with the existing private Hub token."""

import os
from pathlib import Path
from huggingface_hub import get_token


token = get_token()
if not token:
    raise SystemExit("Existing private Hugging Face login is unavailable")
sensitive = ("TOKEN", "SECRET", "CREDENTIAL", "PASSWORD", "PASSWD", "API_KEY", "ACCESS_KEY", "PRIVATE_KEY", "AUTH")
environment = {key: value for key, value in os.environ.items()
               if not any(marker in key.upper() for marker in sensitive)}
environment.update(HF_TOKEN=token, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
python = "/home/m0hawk/.local/share/uv/tools/anyscale/bin/python"
script = str(Path(__file__).with_name("root_submit_diagnostic.py"))
os.execve(python, [python, "-I", "-B", script], environment)
