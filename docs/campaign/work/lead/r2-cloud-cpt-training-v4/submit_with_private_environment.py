"""Resolve the existing Hub login without writing a credential file."""
import os
from pathlib import Path
from huggingface_hub import get_token
token=get_token()
if not token:raise SystemExit('Existing Hugging Face login is unavailable')
env=dict(os.environ,HF_TOKEN=token,PYTHONDONTWRITEBYTECODE='1')
python='/home/m0hawk/.local/share/uv/tools/anyscale/bin/python';script=str(Path(__file__).with_name('submit_job.py'))
os.execve(python,[python,'-B',script],env)
