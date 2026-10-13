# Full-weight update-scale audit v1

This packet compares deterministic, bounded safetensors slices from the
root-verified cloud1902 parent and the rejected representative checkpoint-24.
It reads 417,792 logical tensor bytes across early, middle, and late layers.
It does not recalculate either five-gigabyte model hash or load optimizer state.

The primary finding is broad hidden-matrix movement: 0.616% geometric-mean
relative RMS, with sampled values from 0.438% to 1.069%. The current optimizer
normalizes hidden directions and then applies an explicit LR times
`0.2*sqrt(max(shape))`; it has no layer-index or depth scaling. The active 10x
lower-LR pilot is the narrow test of the leading scale hypothesis.

This is diagnostic evidence only. Slice norms are not whole-tensor norms and
do not establish an optimizer winner.
