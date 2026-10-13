# Cloud CPT parent merge preparation v1

This packet prepares a root-selected cloud CPT LoRA checkpoint as a native BF16 candidate for the full-weight trainer. Candidate 1585 is fully bound to the root-verified local readback. Candidate 1902 is deliberately rejected until its own readback and root verification exist.

The runtime verifies the immutable full checkpoint inventory, its CPT source/data/schedule/parent identity, the exact adapter configuration and all 588 adapter tensors. It loads the verified CPT250 merged base on CPU. Each of 294 LoRA matrices is accumulated in FP32 and cast to BF16 once, then adapters are unloaded without applying a second merge. It never loads the cloud checkpoint tokenizer. It copies the native tokenizer artifacts byte-for-byte and verifies tokenizer.json SHA `3e065a...`, vocabulary 130560, BOS 0, EOS/PAD 1, and native EOG `[1,130073]`.

The scoped E output uses a hidden sibling directory, file and directory fsync, atomic rename, reopen, and a full output weight hash. Estimated peak host RSS is 12 GiB; the binding requires at least 18 GiB available before root starts it. The output weight is approximately 5.03 GB. The estimate derives from the 5.03 GB BF16 base, 0.20 GB adapter, one FP32 matrix update at a time, tokenizer/framework overhead, and prior reviewed streaming merge behavior. Root should retain its host-memory guard because this preparation does not load the model.

Run `root-commands.json` only after root compares checkpoint 1585 and 1902 on the same heldout panel and creates an exact admission receipt. The merged artifact remains an unselected candidate.
