# Frozen 4,435 candidate token review

This independent CPU review re-encodes every stored prompt, target, and their
concatenation with the pinned tokenizer through the existing
`TokenizerAdapter` contract: `add_special_tokens=False`,
`split_special_tokens=True`, and backend `encode_special_tokens=True`.

For each row it checks exact stored IDs, the prompt/target boundary, target
start, terminal suffix, BOS/EOS uniqueness, native-control exclusion, strict
protocol validation, and the assigned 16K or 32K context limit. The detailed
ledger contains counts only and does not copy prompt or target text.

`target_token_count` includes the five-token `>>>>>>> UPDATED` terminal and
excludes the manually appended protocol EOS. The selector reserves
`BOS + prompt + generation_reserve`; a full training sequence needs one more
token for EOS. All 4,435 rows fit both their actual sequence and the stricter
fixed-reserve-plus-EOS ceiling. The separately recovered row also passes:
433 prompt + 1,675 target + BOS + EOS = 2,110 actual tokens. Its 2,048-token
reserve gate uses 2,482 tokens without EOS and 2,483 with EOS.

No context was selected with target or gold data, and no candidate was edited.
