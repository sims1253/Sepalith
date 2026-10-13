# Deferred after DEV rejection

Root rejected target-only checkpoint 25: 24/43 edits, 23/32 no-ops and 8 false positives against floors 26/25/5; exact finish was 0/6. These numbers are root-reported. This worker did not read evaluation records. No export is admitted.

The files in this directory are an unfinished private source draft. Do not launch its commands. The two Python files parse successfully, but behavioral tests and complete launch review were cancelled. The selection template deliberately fails admission. Nothing here selects a checkpoint or changes a production artifact.

The accepted merge script saves a tokenizer through Transformers, which normalizes its bytes. Theta0's accepted parent manifest records a subsequent packaging correction. The draft copies the pinned theta0 tokenizer.json and tokenizer_config.json after saving, then retains the existing vocabulary and token-ID checks. This preserves both original files byte for byte. This change was not exercised against a real model.

The pinned training source records the sampler as a frozen sequential draw schedule with consumed_draws = global_step * 16. A future accepted full step-25 candidate would need campaign-manifest and campaign-state full/step/identity checks, trainer global_step 25, and an observed sampler consumed_draws 400 with matching schedule and split. The draft retains the entire recipe identity, including max_steps 200 and the policy string containing pilot50; renaming these would change checkpoint identity.

The concrete recipe uses the existing merged theta0 BF16 base and a fresh LoRA, keeps the accepted 294-module FP32 merge / single final BF16 rounding verification, and specifies F16 then Q8_0 with q8_0 output/embedding tensors through b10453. It produces only fresh private paths. New weights and runtime hashes are unknown and deliberately absent. The original parent and all prior exports stay in place.

The dependency inventory pins 176 code, library and metadata files, including convert_hf_to_gguf.py, every local conversion Python module, every local gguf Python module, frozen training source files, the quantizer and its bundled shared libraries. It does not hash model weights or any checkpoint file. Python package metadata records torch 2.11.0, transformers 5.5.0, peft 0.20.0, safetensors 0.8.0, numpy 2.2.6, sentencepiece 0.2.2 and tokenizers 0.22.2; metadata hashes are not an attestation of all installed package bytes or system libraries.

The draft resource proposal is two CPU threads, 900 seconds all-in, three 240-second command bounds, 180 seconds for hashing/cleanup, 24 GiB host-free and 16 GiB WSL-available admission, a 12 GiB host-free stop threshold, and 120 GiB free native storage including a 20 GiB artifact reserve. These are unadmitted conservative proposals, not observed current resources or a tested memory ceiling. An outer host guard and root terminal/resource review would still be required; this packet does not implement that launcher.

For a different, independently accepted candidate, root must commission a new bounded review and behavioral tests before adapting this draft. GGUF architecture, tensor/tokenizer/EOG closure, integer-ID/manual-BOS parity and actual development quality remain required after any future export. No inference launch, quality result, or native proof is claimed here.
