# PRE-04 Anyscale large-VRAM admission v3

This packet resolves the inference error in v2: saved compute configs and GPU
fleet snapshots are not a complete account catalog. It performs only read-only
GETs, filters the live fleet snapshot for three target AWS shapes, refreshes the
allowlisted credit fields, and records official documentation separately.

Run the targeted account probe with the installed Anyscale environment:

```bash
/home/m0hawk/.local/share/uv/tools/anyscale/bin/python \
  targeted_account_probe.py --output fresh-account-evidence.json
```

The result cannot authorize a paid launch. Empty fleet results mean only that no
matching node was present in the current snapshot. Exact hosted entitlement,
future capacity, and all-in target-shape pricing remain absent from the read-only
SDK surface.
