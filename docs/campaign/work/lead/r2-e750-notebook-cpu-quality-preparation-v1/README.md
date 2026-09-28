# E750 notebook CPU quality preparation

The accepted E750 Q8 was copied to a fresh notebook campaign directory, verified by full remote SHA-256, published atomically, and made read-only. The b4 model, rollback profile, ordinary editor profile, and their ports were not changed.

The corrected DEV75 panel was rendered and tokenized once with the accepted PRM-03 protocol and tokenizer. All cap arms reuse the exact prompt text and integer IDs. Prompt lengths are 132–2619 tokens including manual BOS, so every row fits 192, 384, and 768 within context 4096. Targets do not affect cap selection.

The prepared suite runs fresh CPU llama-server processes serially on CPUs 0 and 2, `-t 2 -tb 2 -ngl 0`, one isolated loopback port and fresh run directories. Each request has a 120-second offline quality deadline. This deliberately avoids treating notebook prefill/decode slowness as model-quality failure. It supplies no production latency evidence and does not modify the accepted five-second production profiles.

The runner verifies the staged model-integrity receipt and immutable stat identity, all runtime and tokenizer hashes, the packet and panel, root admission, preserved b4/editor profiles, server PID/start tick, native `/tokenize` parity, manual BOS, prompt budget, canonical EOS, cap/transport/mechanical classifications, and all 75 IDs. An incomplete response stops that arm with explicit unattempted IDs. Cleanup signals only the owned server process group and verifies the port is free.

No model server, evaluator, editor, or generated R was run during preparation. Root review and an exact filled admission are required before `launch_remote.sh`.
