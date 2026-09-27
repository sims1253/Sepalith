# Cloud CPT live observer

`live_observer.py` performs a read-only observation of the exact owned Anyscale job `prodjob_neqwkumat4i3rii5yl8af241ks` and run `4f899bbc6e9d46c0a88985d64e6d40e2`. It binds the provider job, job run, and cluster before reading phase events. It lists only the exact private artifact prefix and reads node-local telemetry through Anyscale's SSH key and head-node APIs.

The node connection follows Anyscale's documented topology: a transient in-memory `ssh-agent` authenticates to the Ubuntu head host, then the observer proxies to the Ray endpoint on port 5020. The private key is passed to `ssh-add` on stdin, is never written to a file, and the agent is terminated after the read. The observer sends a fixed Python reader that can only match the exact run directory. Returned records allowlist optimizer timing, trainer metrics, resource counters, staging identity, and training-log file metadata. It does not return prompts, examples, configuration, tokens, IP addresses, credentials, or arbitrary log tails.

Run a fresh observation with:

```text
python3 -B live_observer.py --output FRESH_OUTPUT.json
```

The provider's `RUNNING` state and the `training START` event are reported but never treated as optimizer evidence. `first_optimizer_update_observed` becomes true only when the node-local `telemetry.jsonl` contains an `optimizer_step` record.
