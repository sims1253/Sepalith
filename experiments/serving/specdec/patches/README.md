# ngram-simple cascade patch (b10453) — tested, NOT recommended

`ngram-simple-cascade.patch` makes `common_ngram_simple_draft` try the
configured lookup size `size_n` first, then progressively shorter n-grams
down to unigrams, returning the first non-empty draft.

- Sim ceiling on the frozen 221-trace set: 2.056 tok/step pooled /
  1.861 median vs 1.779 / 1.500 for stock n2m4 (+16% / +24%).
- Measured wall (60-trace subset, cap 256, paired, load ~9):
  **1.443x median — a tie with stock n2m4 (1.40-1.50x), not a win.**
  Accept rate drops 0.42 -> 0.29 and mean draft length doubles (1.26 ->
  2.62), inflating verify batches with wasted rows; the CPU verify-cost
  law (report §3) eats the ceiling. Kept as evidence that closes the
  patch direction by measurement, not just oracle simulation.

Build (work-dir fork; never modify the pinned install):

    cp -r ~/.local/opt/llama.cpp-b10453/src <workdir>/llama-fork-src
    cd <workdir>/llama-fork-src
    patch -p0 < <this dir>/ngram-simple-cascade.patch
    cmake -B build-cpu -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON \
          -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    nice -n 10 cmake --build build-cpu --target llama-server -j 8

Run with the same flags as the champion (the cascade is built into
`size_n=2`; unigram fallback is automatic).
