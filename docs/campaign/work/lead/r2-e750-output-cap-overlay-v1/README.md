# E750 output-cap overlay v1

Fresh isolated preparation from the accepted PRM-03 extension snapshot and E750 native DEV packet. The accepted artifacts remain unchanged.

The managed extension permits exactly `192`, `384`, or `768` in `modelProfile.maxOutputTokens`. That manifest value initializes `NativeCampaignClient`, which bounds `n_predict`, enforces prompt plus output at 4096 tokens, and retains canonical EOS and parser/application behavior. Every other primary identity remains fixed.

The three native arms each carry a profile with the same model, tokenizer, renderer, runtime, corrected DEV75 panel, five-second case deadline, and launch geometry. `output` and `model_profile.maxOutputTokens` must agree. The transport reads that profile value for prompt budgeting, request `n_predict`, overflow checks, cap-hit classification, and result reporting. The canonical profile digest and output cap are repeated in root launch approval and native-process admission.

A limit/length/truncated response without canonical EOS is a cap hit and protocol failure. Canonical EOS at or before the inclusive cap is complete. Native EOG 130073 is a noncanonical terminal failure and is not mislabeled as a cap hit unless the server also reports a limit condition.

No model server, editor, benchmark, CUDA work, model staging, sealed-final access, or paid cloud action occurred. The VSIX is a review artifact. Root must review one arm at a time, create a fresh admission from its template, and use a fresh run root and port. Results are comparable only when all 75 corrected DEV cases complete with exact IDs and 43 edit/32 no-op denominators.
