# DAT-10 semantic 12–26 target-free preparation

This packet runs the reviewed `r2-semantic763-context-admission-v1/prepare_inputs.py`
over every supported-context candidate in semantic shards 12–26. It writes
target-free full pre-edit prediction inputs, complete target bodies in a
separate training sidecar, and explicit preparation holds. It preserves the
recorded Unicode/EOL form and cursor geometry. No row is admitted for
training.

The output is `/mnt/e/sepalith/campaign-20260915/data-work/Semantic10952-preparation-v1`.
The launch wrapper uses the pinned campaign venv, two CPU cores (8 and 10),
nice 10, idle I/O, no CUDA, and a 2,700-second bound. It runs an independent
post-write verifier that re-applies every complete target into the recorded
pre-edit buffer and checks the pinned post-edit source hash. Holds remain
explicit and are never silently dropped.

Root-owned source queue and replay outputs are read-only inputs. The packet
does not edit reviewed preparation code or semantic queue artifacts.
