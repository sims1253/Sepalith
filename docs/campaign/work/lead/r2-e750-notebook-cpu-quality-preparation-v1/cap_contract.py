"""Exact output-cap contract shared by the frozen DEV transport and profile gate."""
import json
from pathlib import Path
ALLOWED_OUTPUT_CAPS=(192,384,768)
CONTEXT_SIZE=4096

def validate_cap(value):
    if type(value) is not int or value not in ALLOWED_OUTPUT_CAPS:
        raise ValueError('output cap must be exactly one of 192,384,768')
    return value

def load_profile(path=None):
    p=Path(path) if path is not None else Path(__file__).with_name('profile.json')
    profile=json.loads(p.read_text())
    cap=validate_cap(profile.get('output'))
    mp=profile.get('model_profile',{})
    if mp.get('maxOutputTokens')!=cap:
        raise ValueError('model profile/output cap mismatch')
    if profile.get('context')!=CONTEXT_SIZE or mp.get('contextSize')!=CONTEXT_SIZE:
        raise ValueError('context must remain 4096')
    if profile.get('case_deadline_seconds')!=5:
        raise ValueError('paired DEV case deadline must remain 5 seconds')
    return profile,cap
