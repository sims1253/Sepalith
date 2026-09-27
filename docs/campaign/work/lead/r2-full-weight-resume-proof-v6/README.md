# Full-weight resume proof v6

This fresh proof keeps the v2 two-update cosine horizon and the v3 complete protocol source closure. It adds the reviewed, narrow Transformers 5.5 EOS-alignment repair after Trainer construction and at `on_train_begin`. The repair accepts only the native EOS form or the observed canonical alignment, then proves the tokenizer vocabulary and model embedding identities are unchanged.

The proof asserts BOS/EOS/PAD, the full vocabulary mapping, native model and generation EOG IDs, and embedding object/storage identity after load, after Trainer construction, at train begin, first loss, before every checkpoint, and after training. Each serialized checkpoint must contain byte-exact `tokenizer.json` and native BOS/EOG/PAD in both model and generation configs.

`seal_checkpoint` flushes files and computes a SHA-256 inventory for every file. The callback uses that returned manifest instead of immediately rereading the same 17+ GiB checkpoint. Independent full-byte verification remains mandatory before resume and at terminal lane acceptance.

Preparation only. Root must allocate the exclusive CUDA lease and run the commands in `root-commands.json` in order. The output root must be fresh.
