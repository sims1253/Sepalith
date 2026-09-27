# C09. Night 3: finish arm A, export, publish

- **Where:** PC nightly window for training; daytime for export and upload
- **Needs:** C08 selected an arm
- **Produces:** SFT arm A at update 1,071, or at the last gate that passed; a
  GGUF export; public weights; evaluation results
- **Effort:** one night, plus about half a day of agent time

## Goal

Complete the selected arm with automatic gates, then make the result usable by
the Kaggle census and by the editor.

## Night

Resume the winning arm from its gate-268 checkpoint. Gates at 536, 803 and
1,071 run automatically under R1. If the deadline arrives first, the trainer
saves and the next night resumes. The runner re-enqueues the job for that.

## Day after completion

1. **Final numbers.** Record `valid`, `cap`, `edit`, `nofp` and `score250` at
   every gate, plus the CPT 2K validation loss at the final gate, as a
   forgetting check against the parent's 0.94197.
2. **Export.** Export GGUF Q8_0 and Q4_K_M with the existing export path.
   Check that the converter handles `LlamaForCausalLM` with vocabulary 130,560
   and both end-of-generation ids. Smoke-test both files with llama.cpp on
   DEV75: counts must be within 2 of the bf16 run for Q8_0. Record Q4_K_M
   drift without gating on it.
3. **Publish.** Create a public model repository
   `scholzmx/sepalith-2b-edit-sft` with a folder `arm-a/`. Put the final
   weights, config, tokenizer, GGUF files and a model card in it, plus a
   results table and the lineage: CPT 11,586, schedule v2 hash, LR. Do not
   upload optimizer state.
4. **Notebook latency (optional).** If the user's notebook is reachable, run
   the llama.cpp latency check there on Q4_K_M. Otherwise leave a note.
5. Write `status/C09.json`. C12 can start once the weights are public and
   C11 has landed.
