# Varlen numerical discrepancy diagnosis

The v3 and v4 reports reproduce the same packed-versus-standalone discrepancy. V4 routes every standalone member through a one-member `packed_seq_lengths` call. Its hidden and sampled-logit maxima are unchanged from v3, and its loss-delta differs by only `1.862645149230957e-08`. This rules out the metadata argument itself.

The reports establish exact cross-document isolation and a common 9,681-token loss denominator. They do not establish within-block numerical parity. Sampled gradients have median relative L2 5.19%, mean 8.86%, and maximum 24.22%, concentrated most strongly in early attention projections. That is too large to treat packing as a transparent throughput change.

The next probe holds inputs and weights fixed and locates the first divergence in layer 0. It compares split and concatenated dense projections, then compares one block-diagonal xformers call with sixteen one-block calls using the same precomputed Q/K/V. An ordinary padded 16-row batch gives an empirical control for numerical changes caused by normal batching. A sampled pre-optimizer Aurora/Muon update comparison connects gradient differences to the actual optimizer without changing model state.

The packet contains CPU-only report validation and probe helpers. No CUDA process ran and no production source changed.
