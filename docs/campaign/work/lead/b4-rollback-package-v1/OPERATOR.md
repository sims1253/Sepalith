# Use the b4 recovery profile

Run these commands in terminals on the notebook. The b4 model and its matched extension are installed in a separate profile. The desktop GPU is not required.

1. Stop the desktop LAN session with its `daily_lan.py stop RUN_DIRECTORY` command. Wait for the session terminal receipt to confirm resource release. This also closes the notebook port forward.
2. Start the b4 CPU server in a notebook terminal:

```sh
env -u GGML_BACKEND_PATH -u GGML_CUDA_GRAPH_OPT CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server -m /home/m0hawk/.local/share/sepalith-campaign-20260915/models/b4/packaging_b4-Q8_0.gguf --host 127.0.0.1 --port 18403 -t 6 -tb 6 --threads-http 2 --parallel 1 -c 8192 -b 256 -ub 256 -ngl 0 -lv 4
```

Keep this terminal open. If port 18403 is occupied, identify the existing session and stop it through its owner before continuing.

3. Open the dedicated recovery editor from a second notebook terminal:

```sh
/usr/bin/code --user-data-dir /home/m0hawk/.local/share/sepalith-b4-rollback/user-data --extensions-dir /home/m0hawk/.local/share/sepalith-b4-rollback/extensions --new-window
```

4. Open an R file. Run **Sepalith: Start server** from the Command Palette to attach the extension to the running server. Accept a suggestion with the editor's inline-suggestion command.
5. To stop recovery serving, close the recovery editor and press Ctrl+C in the server terminal.

The recovery profile uses the legacy b4 renderer. Keep it paired with b4; the R2 model uses a different renderer. Existing everyday editor profiles are preserved.

The installed profile passed a real editor attachment, inline acceptance, saved-buffer check, and R syntax parse on 14 September. This proves the recovery path works. It does not establish representative latency or perfect suggestion quality. The CPU server can be slower than desktop GPU serving.

Evidence: `docs/campaign/receipts/REL-08-b4-recovery-attachment-root-readout.json`. Model and runtime pins: `rollback-spec.json`. The specification records preparation-time status; the later installation and attachment receipts establish the completed checks.
